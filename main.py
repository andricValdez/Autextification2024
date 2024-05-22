

import os
import sys
import joblib
import time
import numpy as np
import pandas as pd 
import logging
import traceback 
from tqdm import tqdm
import torch
import networkx as nx
import scipy as sp
from scipy.sparse import coo_array 
import gensim
from torch_geometric.data import DataLoader, Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import warnings
from polyglot.detect import Detector

import utils
import baselines
import gnn
import text2graph
import node_feat_init

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#************************************* MAIN
def main():
    dataset_name = 'autext24_small' # autext24, autext24_small

    # ****************************** READ DATASET
    autext_dataset = utils.read_dataset(dir_path=utils.ROOT_DIR + '/Autextification2024/dataset/train_dataset/subtask_1/subtask_1_lang.jsonl')
    autext_dataset.loc[autext_dataset['label'] == 'human', 'label'] = 1
    autext_dataset.loc[autext_dataset['label'] == 'generated', 'label'] = 0
    print(autext_dataset.info())

    human_autext_dataset = shuffle(autext_dataset.loc[autext_dataset['label'] == 1])
    machine_autext_dataset = shuffle(autext_dataset.loc[autext_dataset['label'] == 0])

    # identiy lang
    '''
    autext_dataset['lang'] = None
    autext_dataset['lang_code'] = None
    autext_dataset['lang_confidence'] = None
    autext_dataset = utils.lang_identify(dataframe=autext_dataset) # must contain 'text' column
    print(autext_dataset.info())
    print(autext_dataset['label'].value_counts())
    print(autext_dataset['lang'].value_counts())
    #print(autext_dataset.head(5))

    autext_dataset_null_lang = autext_dataset[autext_dataset['lang'].isna()]
    autext_dataset.to_csv(utils.OUTPUTS_PATH + 'autext_dataset.csv')
    autext_dataset_null_lang.to_csv(utils.OUTPUTS_PATH + 'autext_dautext_dataset_null_langataset.csv')
    utils.save_json(autext_dataset.to_dict('records'), file_path='/home/avaldez/projects/Autextification2024/dataset/train_dataset/subtask_1/subtask_1_lang.jsonl')
    return
    '''
    # ****************************** DATA METRICS
    '''
    print(40*'*', 'Human Texts')
    utils.text_metrics(human_autext_dataset)

    print(40*'*', 'Machine Texts')
    utils.text_metrics(machine_autext_dataset)
    '''

    # ****************************** DATASET PARTITION
    print(40*'*', 'Dataset Distro-Partition')
    train_human, test_human = train_test_split(human_autext_dataset, test_size=0.2)
    train_machine, test_machine = train_test_split(machine_autext_dataset, test_size=0.2)

    print("train_human:   ", len(train_human), '  | test_human:   ', len(test_human))
    print("train_machine: ", len(train_machine), ' | test_machine: ', len(test_machine))

    train_dataset_df = pd.concat([train_human, train_machine])
    test_dataset_df = pd.concat([test_human, test_machine])

    print("train_dataset_df: ", train_dataset_df.shape)
    print("test_dataset_df:  ", test_dataset_df.shape)

    # ****************************** BASELINES
    '''
    print(40*'*', 'Train and Test ML baseline models')
    models = ['LinearSVC','MultinomialNB','LogisticRegression','SGDClassifier','xgboost']
    for model in models:
        print(20*'*', 'model: ', model)
        baselines.main(
            train_set=train_dataset_df, 
            test_set=test_dataset_df, 
            algo_ml=model
        )
    '''

    # ****************************** GRAPH NEURAL NETWORK
    test_corpus = [
        {'id': 1, 'doc': "Artificial Intelligence is the ability of a digital computer or computer-controlled robot to perform tasks commonly associated with intelligent beings", "context": {"target": 1}},
        {'id': 2, 'doc': "Natural language processing refers to the branch of computer science that focus on the ability of computers to understand text and spoken words in much the same way human beings can", "context": {"target": 1}},
    ]

    train_text_docs = utils.process_autext24_dataset(train_dataset_df)
    test_text_docs = utils.process_autext24_dataset(test_dataset_df)

    # fine tunning LLM
    #node_feat_init.llm_fine_tuning(train_text_docs, test_text_docs)
    #return

    t2g_instance = text2graph.Text2Graph(
        graph_type='Graph', 
        apply_prep=True, 
        parallel_exec=False, 
        window_size=5
    )

    cut_off = 50
    gnn.graph_neural_network(
        dataset=f'autext24_{cut_off}perc', # autext24_1perc, autext24_10perc, autext24_20perc, autext24_50perc, autext24_100perc
        graph_trans=False, 
        nfi='llm', # llm, w2v
        cut_off=cut_off, 
        t2g_instance=t2g_instance,
        train_text_docs=train_text_docs, 
        test_text_docs=test_text_docs
    )
    return

    outputs_path = '/home/avaldez/projects/Autextification2024/outputs/'
    #corpus_graphs = t2g.transform(train_text_docs[:])
    graphs_train_data = utils.t2g_transform(train_text_docs, t2g_instance, cut_off=cut_off)
    utils.save_data(graphs_train_data, path=outputs_path, file_name=f'graphs_train_{dataset_name}')
    graphs_test_data = utils.t2g_transform(test_text_docs, t2g_instance, cut_off=cut_off)
    utils.save_data(graphs_test_data, path=outputs_path, file_name=f'graphs_test_{dataset_name}')


    for g in graphs_train_data[:5]:
        print('graph: ', str(g))
        #print("nodes: ", g['graph'].nodes(data=True))
        #print("edges: ", g['graph'].edges(data=True))
        #print('\n')

     




    


if __name__ == '__main__':
    main()

   

# ********* CMDs
# python main.py
# nohup bash main.sh >> logs/xxx.log &
# nohup python main.py >> logs/text_to_graph_transform_small.log &
# ps -ef | grep python | grep avaldez
# tail -f logs/experiments_cooccurrence_20240502062849.log 

















