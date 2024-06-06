

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
import networkx as nx
import gc
from sklearn.metrics import f1_score

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


def extract_embeddings_subtask2():
    # TASK 2

    autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_2/train_set_lang.jsonl') 
    #autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_2/train_set.jsonl') 
    #autext_train_set.loc[autext_train_set['label'] == 'A', 'label'] = 0
    #autext_train_set.loc[autext_train_set['label'] == 'B', 'label'] = 1
    #autext_train_set.loc[autext_train_set['label'] == 'C', 'label'] = 2
    #autext_train_set.loc[autext_train_set['label'] == 'D', 'label'] = 3
    #autext_train_set.loc[autext_train_set['label'] == 'E', 'label'] = 4
    #autext_train_set.loc[autext_train_set['label'] == 'F', 'label'] = 5
    print(autext_train_set.info())
    print(autext_train_set['label'].value_counts())

    autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_2/val_set_lang.jsonl') 
    #autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_2/val_set.jsonl') 
    #autext_val_set.loc[autext_val_set['label'] == 'A', 'label'] = 0
    #autext_val_set.loc[autext_val_set['label'] == 'B', 'label'] = 1
    #autext_val_set.loc[autext_val_set['label'] == 'C', 'label'] = 2
    #autext_val_set.loc[autext_val_set['label'] == 'D', 'label'] = 3
    #autext_val_set.loc[autext_val_set['label'] == 'E', 'label'] = 4
    #autext_val_set.loc[autext_val_set['label'] == 'F', 'label'] = 5
    print(autext_val_set.info())
    print(autext_val_set['label'].value_counts())

    # ****************************** BASELINES
    '''
    print(40*'*', 'Train and Test ML baseline models')
    models = ['LinearSVC', 'MultinomialNB', 'LogisticRegression','SGDClassifier', 'RandomForestClassifier', 'xgboost']
    for model in models:
        print(20*'*', 'model: ', model)
        baselines.main(
            train_set=autext_train_set[:], 
            val_set=autext_val_set[:], 
            algo_ml=model,
            target_names=['A', 'B', 'C', 'D', 'E', 'F']
        )
    return
    '''

    # ****************************** FINE TUNE LLM
    '''
    node_feat_init.llm_fine_tuning(
        model_name = 'autext24-subtask2', 
        train_set_df = autext_train_set, 
        val_set_df = autext_val_set,
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    )
    return

    '''

    # ****************************** PROCESS AUTEXT DATASET && CUTOF
    train_text_docs = utils.process_autext24_dataset(autext_train_set)
    val_text_docs = utils.process_autext24_dataset(autext_val_set)

    cut_off_dataset = 100
    cut_dataset_train = len(train_text_docs) * (int(cut_off_dataset) / 100)
    train_text_docs = train_text_docs[:int(cut_dataset_train)]
    cut_dataset_val = len(val_text_docs) * (int(cut_off_dataset) / 100)
    val_text_docs = val_text_docs[:int(cut_dataset_val)]


    # ****************************** GRAPH NEURAL NETWORK - ONE RUNNING
    lang_code = "all"

    lang = 'en' #es, en, fr
    t2g_instance = text2graph.Text2Graph(
        graph_type = 'DiGraph',
            window_size = 10,
            apply_prep = True, 
            steps_preprocessing = {
                "to_lowercase": True, 
                "handle_blank_spaces": True,
                "handle_html_tags": True,
                "handle_special_chars":False,
                "handle_stop_words": False,
            },
            language = lang, #es, en, fr
    )

    exp_file_name = "test"
    dataset_partition = f'autext24_subtask2_{lang_code}_{cut_off_dataset}perc'
    exp_file_path = f'{utils.OUTPUT_DIR_PATH}subtask2/{exp_file_name}_{dataset_partition}/'
    utils.create_expriment_dirs(exp_file_path)
    
    cuda_num = 1
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    llm_finetuned_name='andricValdez/bert-base-multilingual-cased-finetuned-autext24-subtask2'
    
    gnn.graph_neural_network( 
        exp_file_name = 'test',
        dataset_partition = dataset_partition,
        exp_file_path = exp_file_path,
        graph_trans = False, 
        nfi = 'llm', # llm, w2v
        cut_off_dataset = cut_off_dataset, 
        t2g_instance = t2g_instance,
        train_text_docs = train_text_docs[:], 
        val_text_docs = val_text_docs[:],
        device = device,
        edge_features=True,
        edge_dim=2,
        num_labels=6,
        build_dataset = False,
        llm_finetuned_name=llm_finetuned_name,
    )
    return

    #******************* GET stylo feat
    utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=train_text_docs, subset='train') # train, train_all
    utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=val_text_docs, subset='val')
    
    #******************* GET llm_get_embbedings
    torch.cuda.empty_cache()
    gc.collect()
    # LLM CLS
    utils.llm_get_embbedings(
        text_data=train_text_docs, 
        exp_file_path=exp_file_path+'embeddings_cls_llm/', subset='train', 
        emb_type='llm_cls', device=device, save_emb=True, 
        llm_finetuned_name=llm_finetuned_name, num_labels=6)

    utils.llm_get_embbedings(
        text_data=val_text_docs, 
        exp_file_path=exp_file_path+'embeddings_cls_llm/', subset='val', 
        emb_type='llm_cls', device=device, save_emb=True, 
        llm_finetuned_name=llm_finetuned_name, num_labels=6)

    #******************* GET llm_get_embbedings 2
    llm_finetuned_name_2 = 'andricValdez/multilingual-e5-large-finetuned-autext24-subtask2'
    # LLM CLS
    utils.llm_get_embbedings(
        text_data=train_text_docs, 
        exp_file_path=exp_file_path+'embeddings_cls_llm_2/', subset='train', 
        emb_type='llm_cls', device=device, save_emb=True, 
        llm_finetuned_name=llm_finetuned_name_2, num_labels=6)

    utils.llm_get_embbedings(
        text_data=val_text_docs, 
        exp_file_path=exp_file_path+'embeddings_cls_llm_2/', subset='val', 
        emb_type='llm_cls', device=device, save_emb=True, 
        llm_finetuned_name=llm_finetuned_name_2, num_labels=6)

    


def extract_embeddings_subtask1():

    # ****************************** READ DATASET
    lang_code = "all"
    lang_confidence = 95
    lang_codes = ["en", "es", "pt", "ca", "eu", "gl"]

    # ********** TRAIN
    #autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/train_set.jsonl') 
    #autext_train_set['label'] = np.where(autext_train_set['label'] == 'human', 1, 0)
    autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/train_set_lang.jsonl') 
    autext_train_set = autext_train_set[autext_train_set.lang_code.isin(lang_codes)]
    #autext_train_set = shuffle(autext_train_set)
    #autext_train_set = autext_train_set.loc[autext_train_set['lang_code'] == lang_code]
    #autext_train_set = autext_train_set.loc[autext_train_set['lang_confidence'] >= lang_confidence]
    print(autext_train_set.info())
    print(autext_train_set['label'].value_counts())
    #print(autext_train_set['lang_code'].value_counts())

    # ********** VAL
    #autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/val_set.jsonl') 
    #autext_val_set['label'] = np.where(autext_val_set['label'] == 'human', 1, 0)
    autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/val_set_lang.jsonl') 
    autext_val_set = autext_val_set[autext_val_set.lang_code.isin(lang_codes)]
    #autext_val_set = shuffle(autext_val_set)
    #autext_val_set = autext_val_set.loc[autext_val_set['lang_code'] == lang_code]
    #autext_val_set = autext_val_set.loc[autext_val_set['lang_confidence'] >= lang_confidence]
    print(autext_val_set.info())
    print(autext_val_set['label'].value_counts())
    #print(autext_val_set['lang_code'].value_counts())

    # ********** TEST
    autext_test_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/test_set_original.jsonl') 
    
    # ****************************** identiy lang for TRAIN, VAL and TEST set
    '''
    autext_train_set_lang = utils.set_text_lang(dataset=autext_train_set)
    utils.save_jsonl(autext_train_set_lang.to_dict('records'), file_path=utils.DATASET_DIR + 'subtask_1/train_set_lang.jsonl')
    utils.save_csv(autext_train_set_lang, file_path=utils.DATASET_DIR + 'subtask_1/train_set_lang.csv')
    print(autext_train_set_lang.info())
    print(autext_train_set_lang['label'].value_counts())
    print(autext_train_set_lang['lang'].value_counts())

    autext_val_set_lang = utils.set_text_lang(dataset=autext_val_set)
    utils.save_jsonl(autext_val_set_lang.to_dict('records'), file_path=utils.DATASET_DIR + 'subtask_1/val_set_lang.jsonl')
    utils.save_csv(autext_val_set_lang, file_path=utils.DATASET_DIR + 'subtask_1/val_set_lang.csv')
    print(autext_val_set_lang.info())
    print(autext_val_set_lang['label'].value_counts())
    print(autext_val_set_lang['lang'].value_counts())

    autext_test_set_lang = utils.set_text_lang(dataset=autext_test_set)
    utils.save_jsonl(autext_test_set_lang.to_dict('records'), file_path=utils.DATASET_DIR + 'subtask_1/test_set_lang.jsonl')
    utils.save_csv(autext_test_set_lang, file_path=utils.DATASET_DIR + 'subtask_1/test_set_lang.csv')
    print(autext_test_set_lang.info())
    print(autext_test_set_lang['lang'].value_counts())
    
    return
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
            algo_ml=model,
            target_names=['human', 'generated']
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

    # ****************************** PROCESS AUTEXT DATASET && CUTOF
    train_text_docs = utils.process_autext24_dataset(autext_train_set)
    val_text_docs = utils.process_autext24_dataset(autext_val_set)

    cut_off_dataset = 100
    cut_dataset_train = len(train_text_docs) * (int(cut_off_dataset) / 100)
    train_text_docs = train_text_docs[:int(cut_dataset_train)]
    cut_dataset_val = len(val_text_docs) * (int(cut_off_dataset) / 100)
    val_text_docs = val_text_docs[:int(cut_dataset_val)]


    # ****************************** GRAPH NEURAL NETWORK - RUN EXPERIMENTS IN BATCHES
    '''
    exp_file_name = 'experiments_test' 
    experiments_path_file = f'{utils.OUTPUT_DIR_PATH}batch_expriments/{exp_file_name}.csv'
    gnn.graph_neural_network_batch(train_text_docs, val_text_docs, experiments_path_file)
    return
    '''
    # ****************************** GRAPH NEURAL NETWORK - ONE RUNNING

    lang = 'en' #es, en, fr
    t2g_instance = text2graph.Text2Graph(
        graph_type = 'DiGraph',
            window_size = 10,
            apply_prep = True, 
            steps_preprocessing = {
                "to_lowercase": True, 
                "handle_blank_spaces": True,
                "handle_html_tags": True,
                "handle_special_chars":False,
                "handle_stop_words": False,
            },
            language = lang, #es, en, fr
    )

    exp_file_name = "test"
    dataset_partition = f'autext24_{lang_code}_{cut_off_dataset}perc'
    exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
    utils.create_expriment_dirs(exp_file_path)
    
    cuda_num = 0
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    llm_finetuned_name='andricValdez/bert-base-multilingual-cased-finetuned-autext24'

    gnn.graph_neural_network( 
        exp_file_name = 'test',
        dataset_partition = dataset_partition,
        exp_file_path = exp_file_path,
        graph_trans = False, 
        nfi = 'llm', # llm, w2v
        cut_off_dataset = cut_off_dataset, 
        t2g_instance = t2g_instance,
        train_text_docs = train_text_docs[:], 
        val_text_docs = val_text_docs[:],
        device = device,
        edge_features=True,
        edge_dim=2,
        build_dataset = False,
        llm_finetuned_name=llm_finetuned_name
    )
    return

    #******************* GET stylo feat

    utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=train_text_docs, subset='train') # train, train_all
    utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=val_text_docs, subset='val')

    #******************* GET llm_get_embbedings
    
    torch.cuda.empty_cache()
    gc.collect()
    # LLM CLS
    utils.llm_get_embbedings(
        text_data=train_text_docs, exp_file_path=exp_file_path+'embeddings_cls_llm/', 
        subset='train', emb_type='llm_cls', device=device, save_emb=True,
        llm_finetuned_name=llm_finetuned_name, num_labels=2)
    utils.llm_get_embbedings(
        text_data=val_text_docs, exp_file_path=exp_file_path+'embeddings_cls_llm/', 
        subset='val', emb_type='llm_cls', device=device, save_emb=True,
        llm_finetuned_name=llm_finetuned_name, num_labels=2)

    #******************* GET llm_get_embbedings 2

    llm_finetuned_name_2 = 'andricValdez/multilingual-e5-large-finetuned-autext24'
    utils.llm_get_embbedings(
        text_data=train_text_docs, exp_file_path=exp_file_path+'embeddings_cls_llm_2/', 
        subset='train', emb_type='llm_cls', device=device, save_emb=True,
        llm_finetuned_name=llm_finetuned_name_2, num_labels=2)
    utils.llm_get_embbedings(
        text_data=val_text_docs, exp_file_path=exp_file_path+'embeddings_cls_llm_2/', 
        subset='val', emb_type='llm_cls', device=device, save_emb=True,
        llm_finetuned_name=llm_finetuned_name_2, num_labels=2)


def train_clf_model_batch_subtask():
    cuda_num = 1
    train_set_mode = 'train' # train | train_all
    # test_autext24_all_100perc, subtask2/test_autext24_subtask2_all_100perc
    exp_file_path = utils.OUTPUT_DIR_PATH + f'test_autext24_all_100perc'

    feat_types = [
        'embedding_all', 'embedding_gnn_llm', 'embedding_gnn_llm_2', 'embedding_gnn_llm_llm_2', 'embedding_gnn_stylo', 
        'embedding_llm_llm_2', 'embedding_llm_2_stylo', 'embedding_llm_stylo',  'embedding_llm_llm_2_stylo',
        'embedding_gnn', 'embedding_llm', 'embedding_llm_2', 'stylo_feat'
    ]

    # delete
    feat_types = ['embedding_all', 'embedding_gnn']
    
    clf_models_dict = {
        #'LinearSVC': LinearSVC,
        #'LogisticRegression': LogisticRegression,
        #'RandomForestClassifier': RandomForestClassifier,
        #'SGDClassifier': SGDClassifier,
        #'XGBClassifier': XGBClassifier,
        'RRNN_Dense_Clf': gnn.NeuralNetwork
    } 

    for feat_type in feat_types:
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> feat_type: ', feat_type)
        for model in clf_models_dict:
            if model == 'RRNN_Dense_Clf':
                device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
            else:
                device = 'cpu'

            train_clf_model(
                exp_file_path=exp_file_path,
                feat_type=feat_type,
                clf_model=model,
                clf_models_dict=clf_models_dict,
                train_set_mode=train_set_mode,
                device = device
            )


def train_clf_model(exp_file_path, feat_type, clf_model, clf_models_dict, train_set_mode, device='cpu'):

    # ----------------------------------- Embeddings GNN
    emb_train_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext24_train_emb_batch_*.jsonl')
    emb_val_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext24_val_emb_batch_*.jsonl')
    emb_train_gnn_lst_df = [utils.read_json(file) for file in emb_train_gnn_files]
    emb_val_gnn_lst_df = [utils.read_json(file) for file in emb_val_gnn_files]
    emb_train_gnn_df = pd.concat(emb_train_gnn_lst_df)
    emb_val_gnn_df = pd.concat(emb_val_gnn_lst_df)
    #print(emb_train_gnn_df.info())

    # ----------------------------------- Embeddings LLM CLS (google-bert/bert-base-multilingual)
    emb_train_llm_files = glob.glob(f'{exp_file_path}/embeddings_cls_llm/autext24_train_emb_batch_*.jsonl')
    emb_val_llm_files = glob.glob(f'{exp_file_path}/embeddings_cls_llm/autext24_val_emb_batch_*.jsonl')
    emb_train_llm_lst_df = [utils.read_json(file) for file in emb_train_llm_files]
    emb_val_llm_lst_df = [utils.read_json(file) for file in emb_val_llm_files]
    emb_train_llm_df = pd.concat(emb_train_llm_lst_df)
    emb_val_llm_df = pd.concat(emb_val_llm_lst_df)
    #print(emb_train_llm_df.info())

    # ----------------------------------- Embeddings LLM CLS 2 (intfloat/multilingual-e5-large)
    emb_train_llm_files_2 = glob.glob(f'{exp_file_path}/embeddings_cls_llm_2/autext24_train_emb_batch_*.jsonl')
    emb_val_llm_files_2 = glob.glob(f'{exp_file_path}/embeddings_cls_llm_2/autext24_val_emb_batch_*.jsonl')
    emb_train_llm_lst_df_2 = [utils.read_json(file) for file in emb_train_llm_files_2]
    emb_val_llm_lst_df_2 = [utils.read_json(file) for file in emb_val_llm_files_2]
    emb_train_llm_df_2 = pd.concat(emb_train_llm_lst_df_2)
    emb_val_llm_df_2 = pd.concat(emb_val_llm_lst_df_2)
    #print(emb_train_llm_df_2.info())

    # ----------------------------------- Features Stylometrics
    stylo_train_feat = utils.read_json(f'{exp_file_path}/stylometry_train_feat.json')
    stylo_val_feat = utils.read_json(f'{exp_file_path}/stylometry_val_feat.json')
    #print(stylo_train_feat.info())
    
    # ----------------------------------- Merge/concat vectors
    emb_train_merge_df = emb_train_gnn_df.merge(emb_train_llm_df, on='doc_id', how='inner')
    emb_train_merge_df = emb_train_merge_df.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})
    emb_train_merge_df = emb_train_merge_df.merge(emb_train_llm_df_2, on='doc_id', how='inner')
    emb_train_merge_df = emb_train_merge_df.rename(columns={'label_y': 'label_llm_2', 'embedding': 'embedding_llm_2'})
    emb_train_merge_df = emb_train_merge_df.merge(stylo_train_feat, on='doc_id', how='inner')
    emb_train_merge_df['embedding_gnn_llm'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm']
    emb_train_merge_df['embedding_gnn_llm_2'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm_2']
    emb_train_merge_df['embedding_llm_llm_2'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_2']
    emb_train_merge_df['embedding_gnn_llm_llm_2'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_2']
    emb_train_merge_df['embedding_gnn_stylo'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['stylo_feat']
    emb_train_merge_df['embedding_llm_stylo'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['stylo_feat']
    emb_train_merge_df['embedding_llm_2_stylo'] = emb_train_merge_df['embedding_llm_2'] + emb_train_merge_df['stylo_feat']
    emb_train_merge_df['embedding_llm_llm_2_stylo'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_2'] + emb_train_merge_df['stylo_feat']
    emb_train_merge_df['embedding_all'] = emb_train_merge_df['embedding_gnn']  + emb_train_merge_df['embedding_llm']  + emb_train_merge_df['embedding_llm_2'] +  emb_train_merge_df['stylo_feat']
    #print(emb_train_merge_df.info())
    
    emb_val_merge_df = emb_val_gnn_df.merge(emb_val_llm_df, on='doc_id', how='inner')
    emb_val_merge_df = emb_val_merge_df.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})
    emb_val_merge_df = emb_val_merge_df.merge(emb_val_llm_df_2, on='doc_id', how='inner')
    emb_val_merge_df = emb_val_merge_df.rename(columns={'label_y': 'label_llm_2', 'embedding': 'embedding_llm_2'})
    emb_val_merge_df = emb_val_merge_df.merge(stylo_val_feat, on='doc_id', how='inner')
    emb_val_merge_df['embedding_gnn_llm'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm']
    emb_val_merge_df['embedding_gnn_llm_2'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm_2']
    emb_val_merge_df['embedding_llm_llm_2'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_2']
    emb_val_merge_df['embedding_gnn_llm_llm_2'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_2']
    emb_val_merge_df['embedding_gnn_stylo'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['stylo_feat']
    emb_val_merge_df['embedding_llm_stylo'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['stylo_feat']
    emb_val_merge_df['embedding_llm_2_stylo'] = emb_val_merge_df['embedding_llm_2'] + emb_val_merge_df['stylo_feat']
    emb_val_merge_df['embedding_llm_llm_2_stylo'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_2'] + emb_val_merge_df['stylo_feat']
    emb_val_merge_df['embedding_all'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_2'] + emb_val_merge_df['stylo_feat']
    #print(emb_val_merge_df.info())  

    #return 

    # ----------------------------------- Train CLF Model

    if train_set_mode == 'train_all':
        emb_train_merge_df = pd.concat([emb_train_merge_df, emb_val_merge_df])
        print(emb_train_merge_df.info())
    
    #print("feat_len: ", len(emb_train_merge_df[feat_type][0]))
    
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

    print(' ****** clf_model: ', clf_model, ' | feat_len: ', len(train_data[0]))
    if clf_model == 'RRNN_Dense_Clf':
        dense_model = gnn.NeuralNetwork(
            in_channels = len(train_data[0]),
            nhid = 64, 
            out_ch = 1, 
            layers_num = 5
        )
        traind_model = gnn.train_dense_rrnn_clf_model(dense_model, device, train_loader, val_data, val_labels)
        #torch.save(traind_model, f'{exp_file_path}clf_models/{clf_model}_{train_set_mode}_{feat_type}.pt')

    else:
        traind_model = gnn.train_ml_clf_model(clf_models_dict[clf_model], train_data, train_labels, val_data, val_labels)
        #utils.save_data(traind_model, path=f'{exp_file_path}clf_models/', file_name=f'{clf_model}_{train_set_mode}_{feat_type}')


def test_eval_subtask1():
    cut_off_dataset = 100
    exp_file_name = "test"
    dataset_partition = f'autext24_all_{cut_off_dataset}perc'
    exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'

    cuda_num = 1
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # ******************** Read TEST set
    #test_text_docs = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/test_set_original.jsonl') 
    autext_test_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/test_set_lang.jsonl') 
    print(autext_test_set.info())
    autext_test_set = autext_test_set.sort_values('id')

    test_text_docs = []
    corpus_text_docs_dict = autext_test_set.to_dict('records')
    for instance in corpus_text_docs_dict:
        doc = {
            "id": instance['id'], 
            "doc": instance['text'][:], 
            "context": {"id": instance['id'], "target": 1, "lang": instance['lang'], "lang_code": instance['lang_code'], 'lang_confidence': instance['lang_confidence']}
        }
        test_text_docs.append(doc)

    cut_dataset_test = len(test_text_docs) * (int(cut_off_dataset) / 100)
    test_text_docs = test_text_docs[:int(cut_dataset_test)]

    t2g_instance = text2graph.Text2Graph(
        graph_type = 'Graph',
            window_size = 5, 
            apply_prep = True, 
            steps_preprocessing = {
                "to_lowercase": True,
                "handle_blank_spaces": True,
                "handle_html_tags": True,
                "handle_special_chars":False,
                "handle_stop_words": False,
            },
            language = 'en', #es, en, fr
    )

    # ******************** get Features/Embeddings    
    #  get GNN Emb
    #gnn.graph_neural_network_test_eval(test_text_docs, t2g_instance, exp_file_path, dataset_partition, device) 
    #  get LLM Emb
    #utils.llm_get_embbedings(text_data=test_text_docs, exp_file_path=exp_file_path+'embeddings_cls_llm/', subset='test', emb_type='llm_cls', device=device, save_emb=True)
    # get Stylo Feat
    #utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=test_text_docs, subset='test')

    # ******************** Load feat
    # emb_test_gnn
    emb_test_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext24_test_emb_batch_*.jsonl')
    emb_test_gnn_lst_df = [utils.read_json(file) for file in emb_test_gnn_files]
    gnn_test_embeddings = pd.concat(emb_test_gnn_lst_df)
    # emb_test_llm
    emb_test_llm_files = glob.glob(f'{exp_file_path}/embeddings_cls_llm/autext24_test_emb_batch_*.jsonl')
    emb_test_llm_lst_df = [utils.read_json(file) for file in emb_test_llm_files]
    llm_test_embeddings = pd.concat(emb_test_llm_lst_df)
    # stylo_test_feat
    feat_test_stylo = utils.read_json(f'{exp_file_path}/stylometry_test_feat.json')
    
    # ******************** Concat feat
    feat_type = 'embedding_all' 

    merge_test_embeddings = gnn_test_embeddings.merge(llm_test_embeddings, on='doc_id', how='left')
    merge_test_embeddings = merge_test_embeddings.merge(feat_test_stylo, on='doc_id', how='left')
    merge_test_embeddings = merge_test_embeddings.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})
    merge_test_embeddings['embedding_gnn_llm'] = merge_test_embeddings['embedding_gnn'] + merge_test_embeddings['embedding_llm']
    merge_test_embeddings['embedding_gnn_stylo'] = merge_test_embeddings['embedding_gnn'] + merge_test_embeddings['stylo_feat']
    merge_test_embeddings['embedding_llm_stylo'] = merge_test_embeddings['embedding_llm'] + merge_test_embeddings['stylo_feat']
    merge_test_embeddings['embedding_all'] = merge_test_embeddings['embedding_gnn'] + merge_test_embeddings['embedding_llm'] + merge_test_embeddings['stylo_feat']
    print(merge_test_embeddings.info()) 

    # ******************** final clf model
    #  XGBClassifier_train_embedding_all, XGBClassifier_train_all_embedding_all
    clf_model_name_ml = 'XGBClassifier_train_embedding_all'
    clf_model_name_dense = 'RRNN_Dense_Clf_train_embedding_all' 
    device = 'cpu'

    test_feat_data_pt = [torch.tensor(np.asarray(emb), dtype=torch.float, device=device) for emb in merge_test_embeddings[feat_type]]
    test_feat_data_pt = torch.vstack(test_feat_data_pt)
   
    clf_model = utils.load_data(path=f'{exp_file_path}clf_models/', file_name=f'{clf_model_name_ml}') 
    y_pred  = clf_model.predict(test_feat_data_pt)
    merge_test_embeddings['y_pred'] = y_pred

    test_preds_df = pd.DataFrame()
    test_preds_df['doc_id'] = merge_test_embeddings['doc_id']
    test_preds_df['y_pred'] = merge_test_embeddings['y_pred']

    utils.save_csv(test_preds_df, file_path=exp_file_path+'test_preds.csv')

    #clf_model = torch.load(f"{exp_file_path}{clf_model_name}.pt", map_location = device) 
    #model_preds = test_dense_clf(model=clf_model, test_data=test_feat_data_pt, labels_data=test_label_data_pt)


if __name__ == '__main__':
    #main()
    #extract_embeddings_subtask1()
    #test_eval_subtask1()

    #train_clf_model_batch_subtask()
    
    extract_embeddings_subtask2()
   

# ********* CMDs
# python main.py
# nohup bash main.sh >> logs/xxx.log &
# nohup python main.py >> logs/text_to_graph_transform_small.log &
# ps -ef | grep python | grep avaldez
# tail -f logs/experiments_cooccurrence_20240502062849.log 

















