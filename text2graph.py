

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertTokenizerFast, BertModel
from spacy.tokens import Doc
import spacy
import networkx as nx
import networkx
from collections import defaultdict
import logging
import sys
import traceback
import time
from joblib import Parallel, delayed
import warnings
from spacy.tokens import Doc
import nltk
import os
import re
import multiprocessing
from spacy.lang.xx import MultiLanguage

import utils

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#TOKENIZER_FILE = "/home/avaldez/projects/Autextification2024/inputs/bert-base-uncased-vocab.txt"
TOKENIZER_FILE = "/home/avaldez/projects/Autextification2024/inputs/bert-base-multilingual-cased.txt"


class BTokenizer:
    def __init__(self, vocab, vocab_file, lowercase=True):
        self.vocab = vocab
        self._tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=lowercase)
        #self._tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    def __call__(self, text):
        tokens = self._tokenizer.encode(text)
        #tokens = self._tokenizer(text, return_tensors='pt')
        words = []
        spaces = []
        for i, (text, (start, end)) in enumerate(zip(tokens.tokens, tokens.offsets)):
            words.append(text)
            if i < len(tokens.tokens) - 1:
                # If next start != current end we assume a space in between
                next_start, next_end = tokens.offsets[i + 1]
                spaces.append(next_start > end)
            else:
                spaces.append(True)
        return Doc(self.vocab, words=words, spaces=spaces)


#tokenizer_file = "/home/avaldez/projects/Autextification2024/inputs/bert-base-multilingual-cased.txt"
#nlp = spacy.blank("en")
#nlp.tokenizer = BTokenizer(nlp.vocab, tokenizer_file)
#doc = nlp("Justin Drew Bieber is a Canadian singer, songwriter, and actor.")
#print(nlp.pipe_names)
#print(doc.text, [token.text for token in doc])


 
class Text2Graph():
    def __init__(self, 
                graph_type, 
                apply_prep=True,
                parallel_exec=False, 
                window_size=1,
                language='en', 
                steps_preprocessing={}
            ):
        """Constructor method
        """
        self.apply_prep = apply_prep
        self.window_size = window_size
        self.graph_type = graph_type
        self.parallel_exec = parallel_exec

        # scpay model
        #self.nlp = MultiLanguage()
        self.nlp = spacy.blank("xx")
        self.nlp.max_length = 10000000000


    def _get_entities(self, doc_instance) -> list:  
        nodes = []
        for token in doc_instance:
            if token.text in ['[CLS]', '[SEP]', '[UNK]']:
                continue
            node = (str(token.text), {}) # (word, {'node_attr': value})
            nodes.append(node)

        logger.debug("Nodes: %s", nodes)
        return nodes


    def _get_relations(self, doc) -> list:  
        d_cocc = defaultdict(int)
        text_doc_tokens, edges = [], []
        for token in doc:
            if token.text in ['[CLS]', '[SEP]', '[UNK]']:
                continue
            text_doc_tokens.append(token.text)
        for i in range(len(text_doc_tokens)):
            word = text_doc_tokens[i]
            next_word = text_doc_tokens[i+1 : i+1 + self.window_size]
            for t in next_word:
                key = (word, t)
                d_cocc[key] += 1
        for key, value in d_cocc.items():
            edge = (key[0], key[1], {'freq': value})  # (word_i, word_j, {'edge_attr': value})
            edges.append(edge) 

        logger.debug("Edges: %s", edges)
        return edges
    
    
    def _text_normalize(self, text: str) -> list:
        prep_text = text.lower() # text to lower case
        prep_text = re.sub(r'\s+', ' ', prep_text).strip() # remove blank spaces
        prep_text = re.compile('<.*?>').sub(r'', prep_text) # remove html tags

        return prep_text


    def _nlp_pipeline(self, docs: list, params = {'get_multilevel_lang_features': False}):
        doc_tuples = []

        self.nlp.tokenizer = BTokenizer(self.nlp.vocab, TOKENIZER_FILE)
        Doc.set_extension("multilevel_lang_info", default=[], force=True)
        for doc, context in list(self.nlp.pipe(docs, as_tuples=True, n_process=4, batch_size=1000)):
            if params['get_multilevel_lang_features'] == True:
                doc._.multilevel_lang_info = self.get_multilevel_lang_features(doc)
            doc_tuples.append((doc, context))
        return doc_tuples


    def _build_graph(self, nodes: list, edges: list) -> networkx:
        if self.graph_type == 'DiGraph':
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


    def _transform_pipeline(self, doc_instance: tuple) -> list:
        output_dict = {
            'doc_id': doc_instance['id'], 
            'context': doc_instance['context'],
            'graph': None, 
            'number_of_edges': 0, 
            'number_of_nodes': 0, 
            'status': 'success'
        }
        try:
            # get_entities
            nodes = self._get_entities(doc_instance['doc'])
            # get_relations
            edges = self._get_relations(doc_instance['doc'])
            # build graph
            graph = self._build_graph(nodes, edges)
            output_dict['number_of_edges'] += graph.number_of_edges()
            output_dict['number_of_nodes'] += graph.number_of_nodes()
            output_dict['graph'] = graph
        except Exception as e:
            logger.error('Error: %s', str(e))
            logger.error('Error Detail: %s', str(traceback.format_exc()))
            output_dict['status'] = 'fail'
        finally:
            return output_dict
    

    def transform(self, corpus_texts) -> list:
        logger.info("Init transformations: Text to Co-Ocurrence Graph")
        logger.info("Transforming %s text documents...", len(corpus_texts))
        prep_docs, corpus_output_graph, delayed_func = [], [], []

        logger.debug("Preprocessing")
        for doc_data in corpus_texts:
            if self.apply_prep == True:
                doc_data['doc'] = self._text_normalize(doc_data['doc'])
            prep_docs.append(
                (doc_data['doc'], {'id': doc_data['id'], "context": doc_data['context']})
            )

        logger.debug("Transform_pipeline")
        docs = self._nlp_pipeline(prep_docs)

        if self.parallel_exec == True: 
            for input_text in corpus_texts:
                logger.debug('--- Processing doc %s ', str(input_text['id'])) 
                delayed_func.append(
                    utils.joblib_delayed(funct=self._transform_pipeline, params=input_text) 
                )
            num_proc = multiprocessing.cpu_count() // 2
            corpus_output_graph = utils.joblib_parallel(delayed_func, num_proc=num_proc, process_name='transform_cooocur_graph')

        else:
            for doc, context in list(docs):
                corpus_output_graph.append(
                    self._transform_pipeline(
                        {
                            'id': context['id'], 
                            'doc': doc,
                            'context': context['context']
                        }
                    )
                )

            logger.info("Done transformations")
        
        return corpus_output_graph
