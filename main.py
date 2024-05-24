

import os
import sys
import joblib
import time
import numpy as np
import pandas as pd 
import logging
import traceback 
import glob
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
import torch.utils.data as data_utils
from torch.nn.modules.module import Module

from polyglot.detect import Detector
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

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
    ...


def extract_embeddings():

    # ****************************** READ DATASET
    lang_code = "all"
    lang_confidence = 95

    # ********** TRAIN
    #autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/train_set.jsonl') 
    #autext_train_set['label'] = np.where(autext_train_set['label'] == 'human', 1, 0)
    autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/train_set_lang.jsonl') 
    autext_train_set = shuffle(autext_train_set)
    #autext_train_set = autext_train_set.loc[autext_train_set['lang_code'] == lang_code]
    #autext_train_set = autext_train_set.loc[autext_train_set['lang_confidence'] >= lang_confidence]
    print(autext_train_set.info())
    print(autext_train_set['label'].value_counts())

    # ********** VAL
    #autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/val_set.jsonl') 
    #autext_val_set['label'] = np.where(autext_val_set['label'] == 'human', 1, 0)
    autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/val_set_lang.jsonl') 
    autext_val_set = shuffle(autext_val_set)
    #autext_val_set = autext_val_set.loc[autext_val_set['lang_code'] == lang_code]
    #autext_val_set = autext_val_set.loc[autext_val_set['lang_confidence'] >= lang_confidence]
    print(autext_val_set.info())
    print(autext_val_set['label'].value_counts())


    # ****************************** identiy lang for TRAIN and TEST set
    '''
    autext_train_set_lang = utils.set_text_lang(dataset=autext_train_set)
    utils.save_json(autext_train_set_lang.to_dict('records'), file_path=utils.DATASET_DIR + 'subtask_1/train_set_lang.jsonl')
    utils.save_csv(autext_train_set_lang, file_path=utils.DATASET_DIR + 'subtask_1/train_set_lang.csv')
    print(autext_train_set_lang.info())
    print(autext_train_set_lang['label'].value_counts())
    print(autext_train_set_lang['lang'].value_counts())

    autext_val_set_lang = utils.set_text_lang(dataset=autext_val_set)
    utils.save_json(autext_val_set_lang.to_dict('records'), file_path=utils.DATASET_DIR + 'subtask_1/val_set_lang.jsonl')
    utils.save_csv(autext_val_set_lang, file_path=utils.DATASET_DIR + 'subtask_1/val_set_lang.csv')
    print(autext_val_set_lang.info())
    print(autext_val_set_lang['label'].value_counts())
    print(autext_val_set_lang['lang'].value_counts())
    '''

    # ****************************** DATA METRICS
    '''
    print(40*'*', 'Human Texts Train')
    utils.text_metrics(autext_train_set.loc[autext_train_set['label'] == 1])

    print(40*'*', 'Machine Texts Train')
    utils.text_metrics(autext_train_set.loc[autext_train_set['label'] == 0])
    '''

    # ****************************** DATASET PARTITION
    '''
    print(40*'*', 'Dataset Distro-Partition')
    train_human, test_human = train_test_split(human_autext_dataset, test_size=0.2)
    train_machine, test_machine = train_test_split(machine_autext_dataset, test_size=0.2)

    print("train_human:   ", len(train_human), '  | test_human:   ', len(test_human))
    print("train_machine: ", len(train_machine), ' | test_machine: ', len(test_machine))

    train_dataset_df = pd.concat([train_human, train_machine])
    test_dataset_df = pd.concat([test_human, test_machine])

    print("train_dataset_df: ", train_dataset_df.shape)
    print("test_dataset_df:  ", test_dataset_df.shape)
    '''

    # ****************************** BASELINES
    '''
    print(40*'*', 'Train and Test ML baseline models')
    models = ['LinearSVC','MultinomialNB','LogisticRegression','SGDClassifier','xgboost']
    for model in models:
        print(20*'*', 'model: ', model)
        baselines.main(
            train_set=autext_train_set, 
            val_set=autext_val_set, 
            algo_ml=model
        )
    return
    '''

    # ****************************** FINE TUNE LLM
    '''
    node_feat_init.llm_fine_tuning(
        model_name = 'autext24', 
        train_set_df = autext_train_set, 
        val_set_df = autext_val_set,
        device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    )
    return
    '''
    # ****************************** GRAPH NEURAL NETWORK
    train_text_docs = utils.process_autext24_dataset(autext_train_set)
    val_text_docs = utils.process_autext24_dataset(autext_val_set)

    cut_off_dataset = 100
    cut_dataset_train = len(train_text_docs) * (int(cut_off_dataset) / 100)
    train_text_docs = train_text_docs[:int(cut_dataset_train)]
    cut_dataset_val = len(val_text_docs) * (int(cut_off_dataset) / 100)
    val_text_docs = val_text_docs[:int(cut_dataset_val)]

    lang = 'en' #es, en, fr
    t2g_instance = text2graph.Text2Graph(
        graph_type = 'Graph',
            window_size = 5, 
            apply_prep = True, 
            steps_preprocessing = {
                "handle_blank_spaces": True,
                "handle_non_ascii": False,
                "handle_emoticons": True,
                "handle_html_tags": True,
                "handle_contractions": False,
                "handle_stop_words": False,
                "to_lowercase": True
            },
            language = lang, #es, en, fr
    )
    
    exp_file_name = "test"
    dataset_partition = f'autext24_{lang_code}_{cut_off_dataset}perc'
    exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
    utils.create_expriment_dirs(exp_file_path)
    
    cuda_num = 0
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    gnn.graph_neural_network( 
        exp_file_name = 'test',
        dataset_partition = dataset_partition,
        exp_file_path = exp_file_path,
        graph_trans = True, 
        nfi = 'llm', # llm, w2v
        cut_off_dataset = cut_off_dataset, 
        t2g_instance = t2g_instance,
        train_text_docs = train_text_docs, 
        val_text_docs = val_text_docs,
        device = device
    )

    #******************* GET stylo feat
    utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=train_text_docs, subset='train') # train, train_all
    utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=val_text_docs, subset='val')

    #******************* GET llm_get_embbedings
    # LLM CLS
    utils.llm_get_embbedings(text_data=train_text_docs, exp_file_path=exp_file_path+'embeddings_cls_llm/', subset='train', emb_type='llm_cls', device=device, save_emb=True)
    utils.llm_get_embbedings(text_data=val_text_docs, exp_file_path=exp_file_path+'embeddings_cls_llm/', subset='val', emb_type='llm_cls', device=device, save_emb=True)
    

def train_clf_model():

    # ----------------------------------- Setting Params

    cuda_num = 0
    train_set_mode = 'train' # train | train_all
    exp_file_path = utils.OUTPUT_DIR_PATH + f'test_autext24_all_10perc/'

    # embedding_all, embedding_gnn_llm, embedding_gnn_stylo, embedding_llm_stylo, embedding_gnn, embedding_llm, stylo_feat
    feat_type = 'embedding_all' 

    # algo_ml_clf, dense_rrnn_clf
    clf_model_type = 'dense_rrnn_clf' 

    algo_clf = 'XGBClassifier' # only for algo_ml_clf type
    ml_clf_models = {
        'LinearSVC': LinearSVC,
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'SGDClassifier': SGDClassifier,
        'XGBClassifier': XGBClassifier,
    } 

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    if clf_model_type == 'algo_ml_clf':
        device = 'cpu'

    # ----------------------------------- Embeddings GNN
    emb_train_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext24_{train_set_mode}_emb_batch_*.jsonl')
    emb_val_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext24_val_emb_batch_*.jsonl')
    emb_train_gnn_lst_df = [utils.read_json(file) for file in emb_train_gnn_files]
    emb_val_gnn_lst_df = [utils.read_json(file) for file in emb_val_gnn_files]
    emb_train_gnn_df = pd.concat(emb_train_gnn_lst_df)
    emb_val_gnn_df = pd.concat(emb_val_gnn_lst_df)
    #emb_train_gnn_df.to_csv(utils.OUTPUT_DIR_PATH+'emb_train_gnn_df.csv') 
    #print(len(emb_train_gnn_df))
    print(emb_train_gnn_df.info())

    # ----------------------------------- Embeddings LLM CLS
    emb_train_llm_files = glob.glob(f'{exp_file_path}/embeddings_cls_llm/autext24_{train_set_mode}_emb_batch_*.jsonl')
    emb_val_llm_files = glob.glob(f'{exp_file_path}/embeddings_cls_llm/autext24_val_emb_batch_*.jsonl')
    emb_train_llm_lst_df = [utils.read_json(file) for file in emb_train_llm_files]
    emb_val_llm_lst_df = [utils.read_json(file) for file in emb_val_llm_files]
    emb_train_llm_df = pd.concat(emb_train_llm_lst_df)
    emb_val_llm_lst_df = pd.concat(emb_val_llm_lst_df)
    #emb_train_llm_df.to_csv(utils.OUTPUT_DIR_PATH+'emb_train_llm_df.csv') 
    #print(len(emb_train_llm_df))
    print(emb_train_llm_df.info())

    # ----------------------------------- Features Stylometrics
    stylo_train_feat = utils.read_json(f'{exp_file_path}/stylometry_{train_set_mode}_feat.json')
    stylo_val_feat = utils.read_json(f'{exp_file_path}/stylometry_val_feat.json')
    #stylo_train_feat.to_csv(utils.OUTPUT_DIR_PATH+'stylo_train_feat.csv') 
    #print(len(stylo_train_feat))
    print(stylo_train_feat.info())
    
    # ----------------------------------- Merge/concat vectors
    emb_train_merge_df = emb_train_gnn_df.merge(emb_train_llm_df, on='doc_id', how='inner')
    emb_train_merge_df = emb_train_merge_df.merge(stylo_train_feat, on='doc_id', how='inner')
    emb_train_merge_df = emb_train_merge_df.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})
    emb_train_merge_df['embedding_gnn_llm'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm']
    emb_train_merge_df['embedding_gnn_stylo'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['stylo_feat']
    emb_train_merge_df['embedding_llm_stylo'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['stylo_feat']
    emb_train_merge_df['embedding_all'] = emb_train_merge_df['embedding_gnn']  + emb_train_merge_df['embedding_llm'] +  emb_train_merge_df['stylo_feat']
    print(emb_train_merge_df.info())
    
    emb_val_merge_df = emb_val_gnn_df.merge(emb_val_llm_lst_df, on='doc_id', how='inner')
    emb_val_merge_df = emb_val_merge_df.merge(stylo_val_feat, on='doc_id', how='inner')
    emb_val_merge_df = emb_val_merge_df.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})
    emb_val_merge_df['embedding_gnn_llm'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm']
    emb_val_merge_df['embedding_gnn_stylo'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['stylo_feat']
    emb_val_merge_df['embedding_llm_stylo'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['stylo_feat']
    emb_val_merge_df['embedding_all'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm'] + emb_val_merge_df['stylo_feat']
    print(emb_val_merge_df.info())  

    # ----------------------------------- Train CLF Model

    # TRAIN SET
    train_data = [torch.tensor(np.asarray(emb), dtype=torch.float, device=device) for emb in emb_train_merge_df[feat_type]]
    train_data = torch.vstack(train_data)
    train_labels = [torch.tensor(np.asarray(label), dtype=torch.int64, device=device) for label in emb_train_merge_df['label_gnn']]
    train_labels = torch.vstack(train_labels)
    train = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(train, batch_size=64, shuffle=True)
    
    # VAL SET
    val_data = [torch.tensor(np.asarray(emb), dtype=torch.float, device=device) for emb in emb_val_merge_df[feat_type]]
    val_data = torch.vstack(val_data)
    val_labels = [torch.tensor(np.asarray(label), dtype=torch.int64, device=device) for label in emb_val_merge_df['label_gnn']]
    val_labels = torch.vstack(val_labels)
    val = data_utils.TensorDataset(val_data, val_labels)
    val_loader = data_utils.DataLoader(val, batch_size=64, shuffle=True)

    if clf_model_type == 'algo_ml_clf':
        model_name = 'algo_ml_clf_model_' + algo_clf
        traind_model = gnn.train_ml_clf_model(ml_clf_models[algo_clf], train_data, train_labels, val_data, val_labels)
        #utils.save_data(traind_model, path=f'{exp_file_path}', file_name=f'{model_name}_{feat_type}')

    if clf_model_type == 'dense_rrnn_clf':
        model_name = 'dense_clf_model'
        dense_model = gnn.NeuralNetwork(
            in_channels = len(emb_train_merge_df[feat_type][0]),
            nhid = 128, 
            out_ch = 1, 
            layers_num = 5
        )
        print('dense_model: ', dense_model)
        traind_model = gnn.train_dense_rrnn_clf_model(dense_model, device, train_loader, val_data, val_labels)
        # save final clf model
        #torch.save(traind_model, f'{exp_file_path}/{model_name}_{feat_type}.pt')


     

if __name__ == '__main__':
    #main()
    #extract_embeddings()
    train_clf_model()

   

# ********* CMDs
# python main.py
# nohup bash main.sh >> logs/xxx.log &
# nohup python main.py >> logs/text_to_graph_transform_small.log &
# ps -ef | grep python | grep avaldez
# tail -f logs/experiments_cooccurrence_20240502062849.log 

















