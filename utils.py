import logging
import os
import sys
import glob
import pandas as pd
from statistics import mean 
import joblib
from sklearn.datasets import fetch_20newsgroups
import json
from sklearn.utils import shuffle
from datetime import datetime
from joblib import Parallel, delayed
import time
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger
polyglot_logger.setLevel("ERROR")

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = ROOT_DIR + '/datasets/'
OUTPUT_DIR_PATH = ROOT_DIR + '/outputs/'
OUTPUTS_PATH = '/home/avaldez/projects/Autextification2024/outputs/'
INPUT_DIR_PATH = ROOT_DIR + '/inputs/'
CUT_PERCENTAGE_DATASET = 100
TODAY_DATE = datetime.today().strftime('%Y-%m-%d')
CURRENT_TIME = datetime.today().strftime('%Y%m%d%H%M%S')
LANGUAGE = 'en' #es, en, fr

#************************************* UTILS

def read_csv(file_path):
  df = pd.read_csv(file_path)
  return df

def save_csv(dataframe, file_path):
    dataframe.to_csv(file_path, encoding='utf-8', index=False)
  
def read_json(file_path):
  df = pd.read_json(file_path, lines=True)
  df = df.sort_values('id', ascending=True)
  return df

def save_json(data, file_path):
    with open(file_path, "w") as outfile:
        for element in data:  
            json.dump(element, outfile)  
            outfile.write("\n")  

def save_data(data, file_name, path=OUTPUT_DIR_PATH, format_file='.pkl', compress=False):
    logger.info('Saving data: %s', file_name)
    path_file = os.path.join(path, file_name + format_file)
    joblib.dump(data, path_file, compress=compress)

def load_data(file_name, path=INPUT_DIR_PATH, format_file='.pkl', compress=False):
    logger.info('Loading data: %s', file_name)
    path_file = os.path.join(path, file_name + format_file)
    return joblib.load(path_file)

def cut_dataset(corpus_text_docs, cut_percentage_dataset):
  cut_dataset = len(corpus_text_docs) * (int(cut_percentage_dataset) / 100)
  return corpus_text_docs[:int(cut_dataset)]

def delete_dir_files(dir_path):
    if os.path.exists(dir_path):
        files = glob.glob(dir_path + '/*')
        for f in files:
            os.remove(f)

def create_dir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def text_metrics(dataframe):
    text_lens, max_text_len, min_text_len = 0, 0, 1000000000
    for index, row in dataframe.iterrows():
        text_i_len = len(row['text'].split(' '))
        text_lens += text_i_len
        if text_i_len > max_text_len:
            max_text_len = text_i_len
        if text_i_len < min_text_len:
            min_text_len = text_i_len
    print("Text AVG Tokens: ", text_lens/len(dataframe))
    print("Text Max Tokens: ", max_text_len)
    print("Text Min Tokens: ", min_text_len)
    

def read_dataset(dir_path):
    logger.info("*** Using dataset: %s", dir_path)
    return pd.read_json(path_or_buf=dir_path, lines=True)


def process_autext24_dataset(dataframe):
    text_data_lst = []
    corpus_text_docs = shuffle(dataframe)
    corpus_text_docs_dict = corpus_text_docs.to_dict('records')
    for instance in corpus_text_docs_dict:
        if len(instance['text'].split()) < 10:
            continue
        doc = {
            "id": instance['id'], 
            "doc": instance['text'][:], 
            "context": {"id": instance['id'], "target": instance['label'], "lang": instance['lang'], "lang_code": instance['lang_code'], 'lang_confidence': instance['lang_confidence']}
        }
        text_data_lst.append(doc)
    return text_data_lst


def t2g_transform(corpus_text_docs, t2g_instance, cut_off = 100):
    print("Init transform text to graph: ")
    # Apply t2g transformation
    cut_dataset = len(corpus_text_docs) * (int(cut_off) / 100)
    start_time = time.time() # time init
    graph_output = t2g_instance.transform(corpus_text_docs[:int(cut_dataset)])
    for corpus_text_doc in corpus_text_docs:
        for g in graph_output:
            if g['doc_id'] == corpus_text_doc['id']:
                g['context'] = corpus_text_doc['context']
                break
    end_time = (time.time() - start_time)
    print("\t * TOTAL TIME:  %s seconds" % end_time)
    return graph_output

def lang_identify(dataframe): # must contain 'text' column
    for index, row in dataframe[:].iterrows():
        #if len(row['text'].split(' ')) < 50:
        #    continue
        if row['lang']:
            continue
        try: 
            lang = Detector(row['text'])
            #print(len(row['text'].split(' ')), lang.language)
            dataframe.loc[index, 'lang'] = lang.language.name
            dataframe.loc[index, 'lang_code'] = lang.language.code
            dataframe.loc[index, 'lang_confidence'] = lang.language.confidence
        except Exception as err:
            print("error detecting lang: ", str(err))
    return dataframe

def joblib_delayed(funct, params):
    return delayed(funct)(params)

def joblib_parallel(delayed_funct, process_name, num_proc, backend='loky', mmap_mode='c', max_nbytes=None):
    logger.info('Parallel exec for %s, num cpus used: %s', process_name, num_proc)
    return Parallel(
        n_jobs=num_proc,
        backend=backend,
        mmap_mode=mmap_mode,
        max_nbytes=max_nbytes
    )(delayed_funct)
