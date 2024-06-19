

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

    #autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_2/train_set_lang.jsonl') 
    autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_2/train_set.jsonl') 
    autext_train_set.loc[autext_train_set['label'] == 'A', 'label'] = 0
    autext_train_set.loc[autext_train_set['label'] == 'B', 'label'] = 1
    autext_train_set.loc[autext_train_set['label'] == 'C', 'label'] = 2
    autext_train_set.loc[autext_train_set['label'] == 'D', 'label'] = 3
    autext_train_set.loc[autext_train_set['label'] == 'E', 'label'] = 4
    autext_train_set.loc[autext_train_set['label'] == 'F', 'label'] = 5
    print(autext_train_set.info())
    print(autext_train_set['label'].value_counts())

    #autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_2/val_set_lang.jsonl') 
    autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_2/val_set.jsonl') 
    autext_val_set.loc[autext_val_set['label'] == 'A', 'label'] = 0
    autext_val_set.loc[autext_val_set['label'] == 'B', 'label'] = 1
    autext_val_set.loc[autext_val_set['label'] == 'C', 'label'] = 2
    autext_val_set.loc[autext_val_set['label'] == 'D', 'label'] = 3
    autext_val_set.loc[autext_val_set['label'] == 'E', 'label'] = 4
    autext_val_set.loc[autext_val_set['label'] == 'F', 'label'] = 5
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
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"),
        llm_to_finetune = "FacebookAI/xlm-roberta-base",
        num_labels = 6
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
    
    cuda_num = 0
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
        build_dataset = True,
        llm_finetuned_name=llm_finetuned_name,
    )
    
    torch.cuda.empty_cache()
    gc.collect()

    #******************* GET stylo feat
    utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=train_text_docs, subset='train') # train, train_all
    utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=val_text_docs, subset='val')
    
    #******************* GET llm_get_embbedings
    utils.llm_get_embbedings(
        text_data=train_text_docs, 
        exp_file_path=exp_file_path+'embeddings_cls_llm_1/', subset='train', 
        emb_type='llm_cls', device=device, save_emb=True, 
        llm_finetuned_name=llm_finetuned_name, num_labels=6)
    utils.llm_get_embbedings(
        text_data=val_text_docs, 
        exp_file_path=exp_file_path+'embeddings_cls_llm_1/', subset='val', 
        emb_type='llm_cls', device=device, save_emb=True, 
        llm_finetuned_name=llm_finetuned_name, num_labels=6)

    #******************* GET llm_get_embbedings 2
    llm_finetuned_name_2 = 'andricValdez/multilingual-e5-large-finetuned-autext24-subtask2'
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
    
    #******************* GET llm_get_embbedings 3
    llm_finetuned_name_3 = 'andricValdez/xlm-roberta-base-finetuned-autext24-subtask2'
    utils.llm_get_embbedings(
        text_data=train_text_docs, 
        exp_file_path=exp_file_path+'embeddings_cls_llm_3/', subset='train', 
        emb_type='llm_cls', device=device, save_emb=True, 
        llm_finetuned_name=llm_finetuned_name_3, num_labels=6)
    utils.llm_get_embbedings(
        text_data=val_text_docs, 
        exp_file_path=exp_file_path+'embeddings_cls_llm_3/', subset='val', 
        emb_type='llm_cls', device=device, save_emb=True, 
        llm_finetuned_name=llm_finetuned_name_3, num_labels=6)

    

def extract_embeddings_subtask1():

    # ****************************** READ DATASET
    lang_code = "all"
    lang_confidence = 95
    lang_codes = ["en", "es", "pt", "ca", "eu", "gl"]

    # ********** TRAIN
    autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/train_set.jsonl') 
    autext_train_set['label'] = np.where(autext_train_set['label'] == 'human', 1, 0)
    #autext_train_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/train_set_lang.jsonl') 
    #autext_train_set = autext_train_set[autext_train_set.lang_code.isin(lang_codes)]
    #autext_train_set = shuffle(autext_train_set)
    #autext_train_set = autext_train_set.loc[autext_train_set['lang_code'] == lang_code]
    #autext_train_set = autext_train_set.loc[autext_train_set['lang_confidence'] >= lang_confidence]
    print(autext_train_set.info())
    print(autext_train_set['label'].value_counts())
    #print(autext_train_set['lang_code'].value_counts())

    # ********** VAL
    autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/val_set.jsonl') 
    autext_val_set['label'] = np.where(autext_val_set['label'] == 'human', 1, 0)
    #autext_val_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/val_set_lang.jsonl') 
    #autext_val_set = autext_val_set[autext_val_set.lang_code.isin(lang_codes)]
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
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"),
        llm_to_finetune = "FacebookAI/xlm-roberta-base",
        num_labels = 2
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
    dataset_partition = f'autext24_{lang_code}_{cut_off_dataset}perc' # perc | perc_go_cls | perc_go_e5
    exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
    utils.create_expriment_dirs(exp_file_path)
    
    cuda_num = 1
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    
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
        llm_finetuned_name='andricValdez/bert-base-multilingual-cased-finetuned-autext24'
    )

    torch.cuda.empty_cache()
    gc.collect()
    #******************* GET stylo feat
    
    utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=train_text_docs, subset='train') # train, train_all
    utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=val_text_docs, subset='val')
    
    #******************* GET llm_get_embbedings
    llm_finetuned_name_1='andricValdez/bert-base-multilingual-cased-finetuned-autext24'
    utils.llm_get_embbedings(
        text_data=train_text_docs[:], exp_file_path=exp_file_path+'embeddings_cls_llm_1/', 
        subset='train', emb_type='llm_cls', device=device, save_emb=True,
        llm_finetuned_name=llm_finetuned_name_1, num_labels=2)
    utils.llm_get_embbedings(
        text_data=val_text_docs[:], exp_file_path=exp_file_path+'embeddings_cls_llm_1/', 
        subset='val', emb_type='llm_cls', device=device, save_emb=True,
        llm_finetuned_name=llm_finetuned_name_1, num_labels=2)

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
    
    #******************* GET llm_get_embbedings 3
    llm_finetuned_name_3 = 'andricValdez/xlm-roberta-base-finetuned-autext24'
    utils.llm_get_embbedings(
        text_data=train_text_docs, 
        exp_file_path=exp_file_path+'embeddings_cls_llm_3/', subset='train', 
        emb_type='llm_cls', device=device, save_emb=True, 
        llm_finetuned_name=llm_finetuned_name_3, num_labels=2)
    utils.llm_get_embbedings(
        text_data=val_text_docs, 
        exp_file_path=exp_file_path+'embeddings_cls_llm_3/', subset='val', 
        emb_type='llm_cls', device=device, save_emb=True, 
        llm_finetuned_name=llm_finetuned_name_3, num_labels=2)
    


def train_clf_model_batch_subtask():
    cuda_num = 0
    train_set_mode = 'train' # train | train_all
    # test_autext24_all_100perc, subtask2/test_autext24_subtask2_all_100perc
    exp_file_path = utils.OUTPUT_DIR_PATH + f'test_autext24_all_100perc'

    feat_types = [
        'embedding_all', 'embedding_gnn_llm', 'embedding_gnn_llm_2', 'embedding_gnn_llm_3', 'embedding_gnn_llm_llm_2', 'embedding_llm_llm_3', 'embedding_gnn_stylo', 
        'embedding_llm_llm_2', 'embedding_llm_llm_2_llm_3', 'embedding_gnn_llm_llm_3', 'embedding_gnn_llm_llm_2_llm_3', 'embedding_llm_2_stylo',  'embedding_llm_stylo', 'embedding_llm_3_stylo',  
        'embedding_llm_llm_2_stylo', 'embedding_llm_llm_3_stylo', 'embedding_llm_llm_2_llm_3_stylo', 'embedding_gnn', 'embedding_llm', 'embedding_llm_2', 'stylo_feat'
    ]

    # delete
    feat_types = [
        #'embedding_gnn_stylo',
        #'embedding_llm_llm_2_llm_3',
        #'embedding_llm_llm_2_llm_3_stylo',
        #'embedding_gnn',
        #'embedding_gnn_stylo',
        'embedding_all',
        #'embedding_gnn_llm_stylo',
        #'embedding_llm',
        #'embedding_llm_2',
    ]
    
    clf_models_dict = {
        #'LinearSVC': LinearSVC,
        #'LogisticRegression': LogisticRegression,
        #'RandomForestClassifier': RandomForestClassifier,
        #'SGDClassifier': SGDClassifier,
        'XGBClassifier': XGBClassifier,
        #'RRNN_Dense_Clf': gnn.NeuralNetwork
    } 

    emb_train_merge_df, emb_val_merge_df = get_features(exp_file_path)
    #emb_train_merge_df = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_1/autext24_train_embeddings.jsonl')
    #emb_val_merge_df = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_1/autext24_val_embeddings.jsonl')
    
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
                emb_train_merge_df=emb_train_merge_df, 
                emb_val_merge_df=emb_val_merge_df,
                device = device
            )


def train_clf_model(exp_file_path, feat_type, clf_model, clf_models_dict, train_set_mode, emb_train_merge_df, emb_val_merge_df, device='cpu'):

    # ----------------------------------- Train CLF Model

    if train_set_mode == 'train_all':
        emb_train_merge_df = pd.concat([emb_train_merge_df, emb_val_merge_df])
        print(emb_train_merge_df.info())
    
    #print("feat_len: ", len(emb_train_merge_df[feat_type][0]))
    
    # TRAIN SET
    #train_data = [torch.tensor(np.asarray(emb), dtype=torch.float, device=device) for emb in emb_train_merge_df[feat_type]]
    #train_data = torch.vstack(train_data)
    #train_labels = [torch.tensor(np.asarray(label), dtype=torch.int64, device=device) for label in emb_train_merge_df['label_llm']]
    #train_labels = torch.vstack(train_labels)
    #train = data_utils.TensorDataset(train_data, train_labels)
    #train_loader = data_utils.DataLoader(train, batch_size=64, shuffle=True)
    train_data = emb_train_merge_df[feat_type].values.tolist()
    train_labels = emb_train_merge_df['label'].values.tolist()

    # VAL SET
    #val_data = [torch.tensor(np.asarray(emb), dtype=torch.float, device=device) for emb in emb_val_merge_df[feat_type]]
    #val_data = torch.vstack(val_data)
    #val_labels = [torch.tensor(np.asarray(label), dtype=torch.int64, device=device) for label in emb_val_merge_df['label_llm']]
    #val_labels = torch.vstack(val_labels)
    #val = data_utils.TensorDataset(val_data, val_labels)
    #val_loader = data_utils.DataLoader(val, batch_size=64, shuffle=True)
    val_data = emb_val_merge_df[feat_type].values.tolist()
    val_labels = emb_val_merge_df['label'].values.tolist()


    print(' ****** clf_model: ', clf_model, ' | feat_train_len: ', len(train_data[0]), ' | feat_val_len: ', len(train_labels))
    if clf_model == 'RRNN_Dense_Clf':
        dense_model = gnn.NeuralNetwork(
            in_channels = len(train_data[0]),
            nhid = 256, 
            out_ch = 6, 
            layers_num = 3
        )
        traind_model = gnn.train_dense_rrnn_clf_model(dense_model, device, train_loader, val_data, val_labels)
        #torch.save(traind_model, f'{exp_file_path}clf_models/{clf_model}_{train_set_mode}_{feat_type}.pt')

    else:
        traind_model = gnn.train_ml_clf_model(clf_models_dict[clf_model], train_data, train_labels, val_data, val_labels)
        utils.save_data(traind_model, path=f'{exp_file_path}/clf_models/', file_name=f'{clf_model}_{train_set_mode}_{feat_type}')


def get_features(exp_file_path):
     # ----------------------------------- Embeddings GNN
    emb_train_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext24_train_emb_batch_*.jsonl')
    emb_val_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext24_val_emb_batch_*.jsonl')
    emb_train_gnn_lst_df = [utils.read_json(file) for file in emb_train_gnn_files]
    emb_val_gnn_lst_df = [utils.read_json(file) for file in emb_val_gnn_files]
    emb_train_gnn_df = pd.concat(emb_train_gnn_lst_df)
    emb_val_gnn_df = pd.concat(emb_val_gnn_lst_df)
    print(emb_train_gnn_df.info())
    print(emb_val_gnn_df.info())

    # ----------------------------------- Embeddings LLM CLS (google-bert/bert-base-multilingual)
    #emb_train_llm_files = glob.glob(f'{exp_file_path}/embeddings_cls_llm_1/autext24_train_emb_batch_*.jsonl')
    #emb_val_llm_files = glob.glob(f'{exp_file_path}/embeddings_cls_llm_1/autext24_val_emb_batch_*.jsonl')
    #emb_train_llm_lst_df = [utils.read_json(file) for file in emb_train_llm_files]
    #emb_val_llm_lst_df = [utils.read_json(file) for file in emb_val_llm_files]
    #emb_train_llm_df = pd.concat(emb_train_llm_lst_df)
    #emb_val_llm_df = pd.concat(emb_val_llm_lst_df)
    emb_train_llm_df = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_1/autext24_train_embeddings.jsonl')
    emb_val_llm_df = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_1/autext24_val_embeddings.jsonl')
    print(emb_train_llm_df.info())
    print(emb_val_llm_df.info())

    # ----------------------------------- Embeddings LLM CLS 2 (intfloat/multilingual-e5-large)
    #emb_train_llm_files_2 = glob.glob(f'{exp_file_path}/embeddings_cls_llm_2/autext24_train_emb_batch_*.jsonl')
    #emb_val_llm_files_2 = glob.glob(f'{exp_file_path}/embeddings_cls_llm_2/autext24_val_emb_batch_*.jsonl')
    #emb_train_llm_lst_df_2 = [utils.read_json(file) for file in emb_train_llm_files_2]
    #emb_val_llm_lst_df_2 = [utils.read_json(file) for file in emb_val_llm_files_2]
    #emb_train_llm_df_2 = pd.concat(emb_train_llm_lst_df_2)
    #emb_val_llm_df_2 = pd.concat(emb_val_llm_lst_df_2)
    emb_train_llm_df_2 = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_2/autext24_train_embeddings.jsonl')
    emb_val_llm_df_2 = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_2/autext24_val_embeddings.jsonl')
    print(emb_train_llm_df_2.info())
    print(emb_val_llm_df_2.info())
    

    # ----------------------------------- Embeddings LLM CLS 3 (FacebookAI/xlm-roberta-base)
    #emb_train_llm_files_3 = glob.glob(f'{exp_file_path}/embeddings_cls_llm_3/autext24_train_emb_batch_*.jsonl')
    #emb_val_llm_files_3 = glob.glob(f'{exp_file_path}/embeddings_cls_llm_3/autext24_val_emb_batch_*.jsonl')
    #emb_train_llm_lst_df_3 = [utils.read_json(file) for file in emb_train_llm_files_3]
    #emb_val_llm_lst_df_3 = [utils.read_json(file) for file in emb_val_llm_files_3]
    #emb_train_llm_df_3 = pd.concat(emb_train_llm_lst_df_3)
    #emb_val_llm_df_3 = pd.concat(emb_val_llm_lst_df_3)
    '''
    emb_train_llm_df_3 = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_3/autext24_train_embeddings.jsonl')
    emb_val_llm_df_3 = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_3/autext24_val_embeddings.jsonl')
    print(emb_train_llm_df_3.info())
    print(emb_val_llm_df_3.info())
    '''

    # ----------------------------------- Features Stylometrics
    stylo_train_feat = utils.read_json(f'{exp_file_path}/stylometry_train_feat.json')
    stylo_val_feat = utils.read_json(f'{exp_file_path}/stylometry_val_feat.json')
    print(stylo_train_feat.info())
    print(stylo_val_feat.info())
    
    # ----------------------------------- Merge/concat vectors
    emb_train_merge_df = emb_train_gnn_df.merge(emb_train_llm_df, on='doc_id', how='inner')
    emb_train_merge_df = emb_train_merge_df.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})
    emb_train_merge_df = emb_train_merge_df.merge(emb_train_llm_df_2, on='doc_id', how='inner')
    emb_train_merge_df = emb_train_merge_df.rename(columns={'label': 'label_llm_2', 'embedding': 'embedding_llm_2'})
    #emb_train_merge_df = emb_train_merge_df.rename(columns={'label_x': 'label_llm', 'label_y': 'label_llm_2', 'embedding_x': 'embedding_llm', 'embedding_y': 'embedding_llm_2'})
    #emb_train_merge_df = emb_train_merge_df.merge(emb_train_llm_df_3, on='doc_id', how='inner')
    #emb_train_merge_df = emb_train_merge_df.rename(columns={'label': 'label_llm_3', 'embedding': 'embedding_llm_3'})
    emb_train_merge_df = emb_train_merge_df.merge(stylo_train_feat, on='doc_id', how='inner')
    #emb_train_merge_df['embedding_gnn_llm'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm']
    #emb_train_merge_df['embedding_gnn_llm_2'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm_2']
    #emb_train_merge_df['embedding_gnn_llm_3'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm_3']
    #emb_train_merge_df['embedding_llm_llm_2'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_2']
    #emb_train_merge_df['embedding_llm_llm_3'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_3']
    #emb_train_merge_df['embedding_llm_llm_2_llm_3'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_2'] + emb_train_merge_df['embedding_llm_3']
    #emb_train_merge_df['embedding_gnn_llm_llm_2'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_2']
    #emb_train_merge_df['embedding_gnn_llm_llm_3'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_3']
    #emb_train_merge_df['embedding_gnn_llm_llm_2_llm_3'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_2'] + emb_train_merge_df['embedding_llm_3']
    emb_train_merge_df['embedding_gnn_llm_stylo'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['embedding_llm']  + emb_train_merge_df['stylo_feat']
    emb_train_merge_df['embedding_gnn_stylo'] = emb_train_merge_df['embedding_gnn'] + emb_train_merge_df['stylo_feat']
    emb_train_merge_df['embedding_llm_stylo'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['stylo_feat']
    #emb_train_merge_df['embedding_llm_2_stylo'] = emb_train_merge_df['embedding_llm_2'] + emb_train_merge_df['stylo_feat']
    #emb_train_merge_df['embedding_llm_3_stylo'] = emb_train_merge_df['embedding_llm_3'] + emb_train_merge_df['stylo_feat']
    #emb_train_merge_df['embedding_llm_llm_2_stylo'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_2'] + emb_train_merge_df['stylo_feat']
    #emb_train_merge_df['embedding_llm_llm_3_stylo'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_3'] + emb_train_merge_df['stylo_feat']
    #emb_train_merge_df['embedding_llm_llm_2_llm_3_stylo'] = emb_train_merge_df['embedding_llm'] + emb_train_merge_df['embedding_llm_2'] + emb_train_merge_df['embedding_llm_3'] + emb_train_merge_df['stylo_feat']
    emb_train_merge_df['embedding_all'] = emb_train_merge_df['embedding_gnn'] +  emb_train_merge_df['embedding_llm']  + emb_train_merge_df['embedding_llm_2'] + emb_train_merge_df['stylo_feat']
    #emb_train_merge_df['embedding_all'] =  emb_train_merge_df['embedding_llm']  + emb_train_merge_df['embedding_llm_2'] + emb_train_merge_df['embedding_llm_3'] +  emb_train_merge_df['stylo_feat']
    print(emb_train_merge_df.info())

    emb_val_merge_df = emb_val_gnn_df.merge(emb_val_llm_df, on='doc_id', how='inner')
    emb_val_merge_df = emb_val_merge_df.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})
    emb_val_merge_df = emb_val_merge_df.merge(emb_val_llm_df_2, on='doc_id', how='inner')
    emb_val_merge_df = emb_val_merge_df.rename(columns={'label': 'label_llm_2', 'embedding': 'embedding_llm_2'})     
    #emb_val_merge_df = emb_val_merge_df.rename(columns={'label_x': 'label_llm', 'label_y': 'label_llm_2', 'embedding_x': 'embedding_llm', 'embedding_y': 'embedding_llm_2'})
    #emb_val_merge_df = emb_val_merge_df.merge(emb_val_llm_df_3, on='doc_id', how='inner')
    #emb_val_merge_df = emb_val_merge_df.rename(columns={'label': 'label_llm_3', 'embedding': 'embedding_llm_3'})     
    emb_val_merge_df = emb_val_merge_df.merge(stylo_val_feat, on='doc_id', how='inner')
    #emb_val_merge_df['embedding_gnn_llm'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm']
    #emb_val_merge_df['embedding_gnn_llm_2'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm_2']
    #emb_val_merge_df['embedding_gnn_llm_3'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm_3']
    #emb_val_merge_df['embedding_llm_llm_2'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_2']
    #emb_val_merge_df['embedding_llm_llm_3'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_3']
    #emb_val_merge_df['embedding_llm_llm_2_llm_3'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_2'] + emb_val_merge_df['embedding_llm_3']
    #emb_val_merge_df['embedding_gnn_llm_llm_2'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_2']
    #emb_val_merge_df['embedding_gnn_llm_llm_3'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_3']
    #emb_val_merge_df['embedding_gnn_llm_llm_2_llm_3'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_2'] + emb_val_merge_df['embedding_llm_3']
    emb_val_merge_df['embedding_gnn_llm_stylo'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm']  + emb_val_merge_df['stylo_feat']
    emb_val_merge_df['embedding_gnn_stylo'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['stylo_feat']
    emb_val_merge_df['embedding_llm_stylo'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['stylo_feat']
    #emb_val_merge_df['embedding_llm_2_stylo'] = emb_val_merge_df['embedding_llm_2'] + emb_val_merge_df['stylo_feat']
    #emb_val_merge_df['embedding_llm_3_stylo'] = emb_val_merge_df['embedding_llm_3'] + emb_val_merge_df['stylo_feat']
    #emb_val_merge_df['embedding_llm_llm_2_stylo'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_2'] + emb_val_merge_df['stylo_feat']
    #emb_val_merge_df['embedding_llm_llm_3_stylo'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_3'] + emb_val_merge_df['stylo_feat']
    #emb_val_merge_df['embedding_llm_llm_2_llm_3_stylo'] = emb_val_merge_df['embedding_llm'] + emb_val_merge_df['embedding_llm_2'] + emb_val_merge_df['embedding_llm_3'] + emb_val_merge_df['stylo_feat']
    emb_val_merge_df['embedding_all'] = emb_val_merge_df['embedding_gnn'] + emb_val_merge_df['embedding_llm']  + emb_val_merge_df['embedding_llm_2'] + emb_val_merge_df['stylo_feat']
    #emb_val_merge_df['embedding_all'] = emb_val_merge_df['embedding_llm']  + emb_val_merge_df['embedding_llm_2'] + emb_val_merge_df['embedding_llm_3'] +  emb_val_merge_df['stylo_feat']
    print(emb_val_merge_df.info()) 

    return emb_train_merge_df, emb_val_merge_df


def test_eval_subtask():
    subtask = 'subtask_1' # subtask_1, subtask_2
    extract_embeddings = False
    feat_type = 'embedding_all' # embedding_all |embedding_gnn_stylo | embedding_gnn_llm_stylo
    cuda_num = 1
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    cut_off_dataset = 100
    exp_file_name = "test"

    if subtask == 'subtask_1':
        clf_model_name_ml = 'XGBClassifier_train_embedding_all' # XGBClassifier_train_embedding_all | XGBClassifier_train_embedding_gnn_stylo | XGBClassifier_train_embedding_gnn_llm_stylo
        dataset_partition = f'autext24_all_{cut_off_dataset}perc'
        exp_file_path = f'{utils.OUTPUT_DIR_PATH}{exp_file_name}_{dataset_partition}/'
        autext_test_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_1/test_set_original.jsonl') 

        num_labels=2
        llm_finetuned_name = 'andricValdez/bert-base-multilingual-cased-finetuned-autext24'
        llm_finetuned_name_2 = 'andricValdez/multilingual-e5-large-finetuned-autext24'
        llm_finetuned_name_3 = 'andricValdez/xlm-roberta-base-finetuned-autext24'

    if subtask == 'subtask_2':
        clf_model_name_ml = 'LinearSVC_train_embedding_all' # LinearSVC_train_embedding_all | LinearSVC_train_embedding_gnn_stylo | LinearSVC_train_embedding_gnn_llm_stylo
        dataset_partition = f'autext24_subtask2_all_{cut_off_dataset}perc'
        exp_file_path = f'{utils.OUTPUT_DIR_PATH}subtask2/{exp_file_name}_{dataset_partition}/'
        autext_test_set = utils.read_json(dir_path=utils.DATASET_DIR + 'subtask_2/test_set_original.jsonl') 

        num_labels=6
        llm_finetuned_name = 'andricValdez/bert-base-multilingual-cased-finetuned-autext24-subtask2'
        llm_finetuned_name_2 = 'andricValdez/multilingual-e5-large-finetuned-autext24-subtask2'
        llm_finetuned_name_3 = 'andricValdez/xlm-roberta-base-finetuned-autext24-subtask2'


    # ******************** Read TEST set
    print(autext_test_set.info())
    autext_test_set = autext_test_set.sort_values('id')

    test_text_docs = []
    corpus_text_docs_dict = autext_test_set.to_dict('records')
    for instance in corpus_text_docs_dict:
        doc = {
            "id": instance['id'], 
            "doc": instance['text'][:], 
            "context": {"id": instance['id'], "target": 1}
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
    if extract_embeddings:
        #  get GNN Emb
        gnn.graph_neural_network_test_eval(test_text_docs, t2g_instance, exp_file_path, dataset_partition, device) 
  
        #  get LLM Emb 1
        utils.llm_get_embbedings(
            text_data=test_text_docs, 
            exp_file_path=exp_file_path+'embeddings_cls_llm_1/', subset='test', 
            emb_type='llm_cls', device=device, save_emb=True, 
            llm_finetuned_name=llm_finetuned_name, 
            num_labels=num_labels)
        #  get LLM Emb 2
        utils.llm_get_embbedings(
            text_data=test_text_docs, 
            exp_file_path=exp_file_path+'embeddings_cls_llm_2/', subset='test', 
            emb_type='llm_cls', device=device, save_emb=True, 
            llm_finetuned_name=llm_finetuned_name_2, 
            num_labels=num_labels)
        #  get LLM Emb 3
        utils.llm_get_embbedings(
            text_data=test_text_docs, 
            exp_file_path=exp_file_path+'embeddings_cls_llm_3/', subset='test', 
            emb_type='llm_cls', device=device, save_emb=True, 
            llm_finetuned_name=llm_finetuned_name_3, 
            num_labels=num_labels)

        # get Stylo Feat
        utils.get_stylo_feat(exp_file_path=exp_file_path, text_data=test_text_docs, subset='test')
        
    # ******************** Load feat
    # emb_test_gnn
    emb_test_gnn_files = glob.glob(f'{exp_file_path}/embeddings_gnn/autext24_test_emb_batch_*.jsonl')
    emb_test_gnn_lst_df = [utils.read_json(file) for file in emb_test_gnn_files]
    gnn_test_embeddings = pd.concat(emb_test_gnn_lst_df)
    # emb_test_llm
    llm_1_test_embeddings = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_1/autext24_test_embeddings.jsonl')
    llm_2_test_embeddings = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_2/autext24_test_embeddings.jsonl')
    #llm_3_test_embeddings = utils.read_json(dir_path=f'{exp_file_path}/embeddings_cls_llm_3/autext24_test_embeddings.jsonl')

    # stylo_test_feat
    feat_test_stylo = utils.read_json(f'{exp_file_path}/stylometry_test_feat.json')
    
    # ******************** Concat feat
    merge_test_embeddings = gnn_test_embeddings.merge(llm_1_test_embeddings, on='doc_id', how='left')
    merge_test_embeddings = merge_test_embeddings.rename(columns={'label_x': 'label_gnn', 'label_y': 'label_llm', 'embedding_x': 'embedding_gnn', 'embedding_y': 'embedding_llm'})
    merge_test_embeddings = merge_test_embeddings.merge(llm_2_test_embeddings, on='doc_id', how='left')
    merge_test_embeddings = merge_test_embeddings.rename(columns={'label': 'label_llm_2', 'embedding': 'embedding_llm_2'})     
    #merge_test_embeddings = merge_test_embeddings.merge(llm_3_test_embeddings, on='doc_id', how='left')
    #merge_test_embeddings = merge_test_embeddings.rename(columns={'label': 'label_llm_3', 'embedding': 'embedding_llm_3'})     
    merge_test_embeddings = merge_test_embeddings.merge(feat_test_stylo, on='doc_id', how='left')

    merge_test_embeddings['embedding_gnn_stylo'] = merge_test_embeddings['embedding_gnn'] + merge_test_embeddings['stylo_feat']
    merge_test_embeddings['embedding_gnn_llm_stylo'] = merge_test_embeddings['embedding_gnn'] + merge_test_embeddings['embedding_llm']  + merge_test_embeddings['stylo_feat']
    merge_test_embeddings['embedding_all'] = merge_test_embeddings['embedding_gnn'] + merge_test_embeddings['embedding_llm']  + merge_test_embeddings['embedding_llm_2'] + merge_test_embeddings['stylo_feat'] 
    print(merge_test_embeddings.info()) 

    
    # ******************** final clf model
    device = 'cpu'
    print("device: ", device)
    #test_feat_data_pt = [torch.tensor(np.asarray(emb), dtype=torch.float, device=device) for emb in merge_test_embeddings[feat_type]]
    #test_feat_data_pt = torch.vstack(test_feat_data_pt)
    test_feat_data_pt = merge_test_embeddings[feat_type].values.tolist()
    
    clf_model = utils.load_data(path=f'{exp_file_path}clf_models/', file_name=f'{clf_model_name_ml}') 
    y_pred = clf_model.predict(test_feat_data_pt)

    test_preds_df = pd.DataFrame()
    test_preds_df['id'] = merge_test_embeddings['doc_id']
    test_preds_df['label'] = y_pred

    if subtask == 'subtask_1':
        test_preds_df['label'] = np.where(test_preds_df['label'] == 1, 'human', 'generated')
    if subtask == 'subtask_2':
        test_preds_df.loc[test_preds_df['label'] == 0, 'label'] = 'A'
        test_preds_df.loc[test_preds_df['label'] == 1, 'label'] = 'B'
        test_preds_df.loc[test_preds_df['label'] == 2, 'label'] = 'C'
        test_preds_df.loc[test_preds_df['label'] == 3, 'label'] = 'D'
        test_preds_df.loc[test_preds_df['label'] == 4, 'label'] = 'E'
        test_preds_df.loc[test_preds_df['label'] == 5, 'label'] = 'F'

    
    print(test_preds_df.info())
    utils.save_jsonl(list(test_preds_df.to_dict('records')), file_path=f'{exp_file_path}/test_preds_{clf_model_name_ml}.jsonl')
    utils.save_csv(test_preds_df, file_path=f'{exp_file_path}/test_preds_{clf_model_name_ml}.csv')

    #clf_model = torch.load(f"{exp_file_path}{clf_model_name}.pt", map_location = device) 
    #model_preds = test_dense_clf(model=clf_model, test_data=test_feat_data_pt, labels_data=test_label_data_pt)


def eval_subtask_ricardo():
    # experiments_ricardo_s1  |  experiments_ricardo_s2
    subtask = 'experiments_ricardo_s2'
    # subtask_1: SGDClassifier_all | SGDClassifier_all_full_train
    # subtask_2: SGDClassifier_bert_multi_e5_multi_xlm_multi | SGDClassifier_bert_multi_e5_multi_xlm_multi_full_train
    model_name = 'SGDClassifier_bert_multi_e5_multi_xlm_multi_full_train' 
    path_test = utils.OUTPUT_DIR_PATH + subtask + '/test/'

    test_set = utils.read_json(dir_path=path_test + 'test_set_original.jsonl')
    test_stylo = pd.read_csv(path_test + 'stylometry_test.csv').values.tolist()
    test_stylo_2 = [[x, y] for x, y in zip(test_set['text'].str.count('\d').values.tolist(), test_set['text'].str.count('\s').values.tolist())]
    test_bert_multi = pd.read_csv(path_test + 'test_bert-base-multilingual-cased-finetuned-autext24.csv', header=None).values.tolist()
    test_e5_multi = pd.read_csv(path_test + 'test_multilingual-e5-large-finetuned-autext24.csv', header=None).values.tolist()
    test_xlm_multi = pd.read_csv(path_test + 'test_xlm-roberta-base-finetuned-autext24.csv', header=None).values.tolist()
    
    print(f"test_set:", test_set.info())
    print(f"test_stylo -> num_instances: {len(test_stylo)} | len_vect: {len(test_stylo[0])}")
    print(f"test_stylo_2 -> num_instances: {len(test_stylo_2)} | len_vect: {len(test_stylo_2[0])}")
    print(f"test_bert_multi -> num_instances: {len(test_bert_multi)} | len_vect: {len(test_bert_multi[0])}")
    print(f"test_e5_multi -> num_instances: {len(test_e5_multi)} | len_vect: {len(test_e5_multi[0])}")
    print(f"test_xlm_multi -> num_instances: {len(test_xlm_multi)} | len_vect: {len(test_xlm_multi[0])}")
    
    test_emb = []
    for index in range(len(test_bert_multi)):
        test_emb.append(
            #test_stylo[index] + 
            #test_stylo_2[index] + 
            test_bert_multi[index] + 
            test_e5_multi[index] + 
            test_xlm_multi[index]
        )

    clf_model = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}{subtask}', file_name=f'{model_name}') 
    y_pred  = clf_model.predict(test_emb)

    test_preds_df = pd.DataFrame()
    test_preds_df['id'] = test_set['id']
    test_preds_df['label'] = y_pred

    if subtask == 'experiments_ricardo_s1':
        test_preds_df['label'] = np.where(test_preds_df['label'] == 1, 'human', 'generated')
    if subtask == 'experiments_ricardo_s2':
        test_preds_df.loc[test_preds_df['label'] == 0, 'label'] = 'A'
        test_preds_df.loc[test_preds_df['label'] == 1, 'label'] = 'B'
        test_preds_df.loc[test_preds_df['label'] == 2, 'label'] = 'C'
        test_preds_df.loc[test_preds_df['label'] == 3, 'label'] = 'D'
        test_preds_df.loc[test_preds_df['label'] == 4, 'label'] = 'E'
        test_preds_df.loc[test_preds_df['label'] == 5, 'label'] = 'F'

    print(test_preds_df.info())
    utils.save_jsonl(list(test_preds_df.to_dict('records')), file_path=f'{utils.OUTPUT_DIR_PATH}{subtask}/test_preds_full_train.jsonl')
    utils.save_csv(test_preds_df, file_path=f'{utils.OUTPUT_DIR_PATH}{subtask}/test_preds_full_train.csv')
    #utils.save_jsonl(list(test_preds_df.to_dict('records')), file_path=f'{utils.OUTPUT_DIR_PATH}{subtask}/test_preds.jsonl')
    #utils.save_csv(test_preds_df, file_path=f'{utils.OUTPUT_DIR_PATH}{subtask}/test_preds.csv')


def feat_subtask1_ricardo():
    path_train = utils.OUTPUT_DIR_PATH + '/experiments_ricardo_s1/train/'
    path_val = utils.OUTPUT_DIR_PATH + '/experiments_ricardo_s1/val/'

    # TRAIN
    train_S1 = pd.read_csv(path_train + 'train_S1.csv')
    train_S1['label'] = np.where(train_S1['label'] == 'human', 1, 0)
    #train_S1 = train_S1.iloc[1:]
    train_stylo = pd.read_csv(path_train + 'stylometry_train.csv').values.tolist()
    train_stylo_2 = [[x, y] for x, y in zip(train_S1['text'].str.count('\d').values.tolist(), train_S1['text'].str.count('\s').values.tolist())]
    train_bert_multi = pd.read_csv(path_train + 'train_bert-base-multilingual-cased-finetuned-autext24.csv', header=None).values.tolist()
    train_e5_multi = pd.read_csv(path_train + 'train_multilingual-e5-large-finetuned-autext24.csv', header=None).values.tolist()
    train_xlm_multi = pd.read_csv(path_train + 'train_xlm-roberta-base-finetuned-autext24.csv', header=None).values.tolist()
    
    print(f"train_S1:", train_S1.info())
    print(f"train_stylo -> num_instances: {len(train_stylo)} | len_vect: {len(train_stylo[0])}")
    print(f"train_stylo_2 -> num_instances: {len(train_stylo_2)} | len_vect: {len(train_stylo_2[0])}")
    print(f"train_bert_multi -> num_instances: {len(train_bert_multi)} | len_vect: {len(train_bert_multi[0])}")
    print(f"train_e5_multi -> num_instances: {len(train_e5_multi)} | len_vect: {len(train_e5_multi[0])}")
    print(f"train_xlm_multi -> num_instances: {len(train_xlm_multi)} | len_vect: {len(train_xlm_multi[0])}")
    
    # VAL-TEST
    val_S1 = pd.read_csv(path_val + 'test_S1.csv')
    val_S1['label'] = np.where(val_S1['label'] == 'human', 1, 0)
    #val_S1 = val_S1.iloc[1:]
    val_stylo = pd.read_csv(path_val + 'stylometry_test.csv').values.tolist()
    val_stylo_2 = [[x, y] for x, y in zip(val_S1['text'].str.count('\d').values.tolist(), val_S1['text'].str.count('\s').values.tolist())]
    val_bert_multi = pd.read_csv(path_val + 'test_bert-base-multilingual-cased-finetuned-autext24.csv', header=None).values.tolist()
    val_e5_multi = pd.read_csv(path_val + 'test_multilingual-e5-large-finetuned-autext24.csv', header=None).values.tolist()
    val_xlm_multi = pd.read_csv(path_val + 'test_xlm-roberta-base-finetuned-autext24.csv', header=None).values.tolist()

    print(f"\nval_S1:", val_S1.info())
    print(f"val_stylo -> num_instances: {len(val_stylo)} | len_vect: {len(val_stylo[0])}")
    print(f"val_stylo_2 -> num_instances: {len(val_stylo_2)} | len_vect: {len(val_stylo_2[0])}")
    print(f"val_bert_multi -> num_instances: {len(val_bert_multi)} | len_vect: {len(val_bert_multi[0])}")
    print(f"val_e5_multi -> num_instances: {len(val_e5_multi)} | len_vect: {len(val_e5_multi[0])}")
    print(f"val_xlm_multi -> num_instances: {len(val_xlm_multi)} | len_vect: {len(val_xlm_multi[0])}")
    
    train_emb = {
        "all": [], "bert_multi": [], "e5_multi": [], "xlm_multi": [], 
        "bert_multi_e5_multi": [], "bert_multi_xlm_multi": [], "e5_multi_xlm_multi": [],
        "bert_multi_e5_multi_xlm_multi": [] 
    }
    val_emb = {
        "all": [], "bert_multi": [], "e5_multi": [], "xlm_multi": [], 
        "bert_multi_e5_multi": [], "bert_multi_xlm_multi": [], "e5_multi_xlm_multi": [],
        "bert_multi_e5_multi_xlm_multi": [] 
    }
    
    for index in range(len(train_bert_multi)):
        train_emb["bert_multi"].append(train_bert_multi[index])
        train_emb["e5_multi"].append(train_e5_multi[index])
        train_emb["xlm_multi"].append(train_xlm_multi[index])
        train_emb["bert_multi_e5_multi"].append(train_bert_multi[index] + train_e5_multi[index])
        train_emb["bert_multi_xlm_multi"].append(train_bert_multi[index] + train_xlm_multi[index])
        train_emb["e5_multi_xlm_multi"].append(train_e5_multi[index] + train_xlm_multi[index])
        train_emb["bert_multi_e5_multi_xlm_multi"].append(train_bert_multi[index] + train_e5_multi[index] + train_xlm_multi[index])
        train_emb["all"].append(train_stylo[index] + train_stylo_2[index] + train_bert_multi[index] + train_e5_multi[index] + train_xlm_multi[index])
    
    for index in range(len(val_bert_multi)):
        val_emb["bert_multi"].append(val_bert_multi[index])
        val_emb["e5_multi"].append(val_e5_multi[index])
        val_emb["xlm_multi"].append(val_xlm_multi[index])
        val_emb["bert_multi_e5_multi"].append(val_bert_multi[index] + val_e5_multi[index])
        val_emb["bert_multi_xlm_multi"].append(val_bert_multi[index] + val_xlm_multi[index])
        val_emb["e5_multi_xlm_multi"].append(val_e5_multi[index] + val_xlm_multi[index])
        val_emb["bert_multi_e5_multi_xlm_multi"].append(val_bert_multi[index] + val_e5_multi[index] + val_xlm_multi[index])
        val_emb["all"].append(val_stylo[index] + val_stylo_2[index] + val_bert_multi[index] + val_e5_multi[index] + val_xlm_multi[index])

    #train_all = train_emb['all'] + val_emb['all']
    #label_all = train_S1['label'].tolist()  + val_S1['label'].tolist() 

    #print(f"\ntrain_emb_all -> num_instances: {len(train_all)} | len_vect: {len(train_all[0])} | len_label {len(label_all)}")
    print(f"\ntrain_emb_all -> num_instances: {len(train_emb['all'])} | len_vect: {len(train_emb['all'][0])} | len_label {len(train_S1['label'])}")
    print(f"val_emb_all -> num_instances: {len(val_emb['all'])} | len_vect: {len(val_emb['all'][0])}")

    clf_models_dict = {
        #'LinearSVC': LinearSVC,
        #'LogisticRegression': LogisticRegression,
        'SGDClassifier': SGDClassifier,
        #'XGBClassifier': XGBClassifier,
    } 
    emb_type = 'all'

    #for emb_type in train_emb.keys():
    #    print("***** emb_type: ", emb_type)
    for model_name in clf_models_dict.keys():
        print("\t model_name: ", model_name)
        #trained_model = gnn.train_ml_clf_model(clf_models_dict[model_name], train_all, label_all, val_emb[emb_type], val_S1['label'])
        #utils.save_data(trained_model, path=f'{utils.OUTPUT_DIR_PATH}experiments_ricardo_s1/', file_name=f'{model_name}_{emb_type}_full_train')
        trained_model = gnn.train_ml_clf_model(clf_models_dict[model_name], train_emb['all'], train_S1['label'], val_emb[emb_type], val_S1['label'])
        utils.save_data(trained_model, path=f'{utils.OUTPUT_DIR_PATH}experiments_ricardo_s1/', file_name=f'{model_name}_{emb_type}')


def feat_subtask2_ricardo():
    path_train = utils.OUTPUT_DIR_PATH + '/experiments_ricardo_s2/train/'
    path_val = utils.OUTPUT_DIR_PATH + '/experiments_ricardo_s2/val/'

    # TRAIN
    train_S2 = pd.read_csv(path_train + 'train_S2.csv')
    #train_S2.loc[train_S2['label'] == 'A', 'label'] = 0
    #train_S2.loc[train_S2['label'] == 'B', 'label'] = 1
    #train_S2.loc[train_S2['label'] == 'C', 'label'] = 2
    #train_S2.loc[train_S2['label'] == 'D', 'label'] = 3
    #train_S2.loc[train_S2['label'] == 'E', 'label'] = 4
    #train_S2.loc[train_S2['label'] == 'F', 'label'] = 5
    #train_S2 = train_S2.iloc[1:]
    train_stylo = pd.read_csv(path_train + 'stylometry_train_S2.csv').values.tolist()
    train_stylo_2 = [[x, y] for x, y in zip(train_S2['text'].str.count('\d').values.tolist(), train_S2['text'].str.count('\s').values.tolist())]
    train_bert_multi = pd.read_csv(path_train + 'train_bert-base-multilingual-cased-finetuned-autext24-subtask2.csv', header=None).values.tolist()
    train_e5_multi = pd.read_csv(path_train + 'train_multilingual-e5-large-finetuned-autext24-subtask2.csv', header=None).values.tolist()
    train_xlm_multi = pd.read_csv(path_train + 'train_xlm-roberta-base-finetuned-autext24-subtask2.csv', header=None).values.tolist()
    
    print(f"train_S2:", train_S2.info())
    print(f"train_stylo -> num_instances: {len(train_stylo)} | len_vect: {len(train_stylo[0])}")
    print(f"train_stylo_2 -> num_instances: {len(train_stylo_2)} | len_vect: {len(train_stylo_2[0])}")
    print(f"train_bert_multi -> num_instances: {len(train_bert_multi)} | len_vect: {len(train_bert_multi[0])}")
    print(f"train_e5_multi -> num_instances: {len(train_e5_multi)} | len_vect: {len(train_e5_multi[0])}")
    print(f"train_xlm_multi -> num_instances: {len(train_xlm_multi)} | len_vect: {len(train_xlm_multi[0])}")
    
    # VAL-TEST
    val_S2 = pd.read_csv(path_val + 'test_S2.csv')
    #val_S2.loc[val_S2['label'] == 'A', 'label'] = 0
    #val_S2.loc[val_S2['label'] == 'B', 'label'] = 1
    #val_S2.loc[val_S2['label'] == 'C', 'label'] = 2
    #val_S2.loc[val_S2['label'] == 'D', 'label'] = 3
    #val_S2.loc[val_S2['label'] == 'E', 'label'] = 4
    #val_S2.loc[val_S2['label'] == 'F', 'label'] = 5
    #val_S2 = val_S2.iloc[1:]
    val_stylo = pd.read_csv(path_val + 'stylometry_test_S2.csv').values.tolist()
    val_stylo_2 = [[x, y] for x, y in zip(val_S2['text'].str.count('\d').values.tolist(), val_S2['text'].str.count('\s').values.tolist())]
    val_bert_multi = pd.read_csv(path_val + 'test_bert-base-multilingual-cased-finetuned-autext24-subtask2.csv', header=None).values.tolist()
    val_e5_multi = pd.read_csv(path_val + 'test_multilingual-e5-large-finetuned-autext24-subtask2.csv', header=None).values.tolist()
    val_xlm_multi = pd.read_csv(path_val + 'test_xlm-roberta-base-finetuned-autext24-subtask2.csv', header=None).values.tolist()

    print(f"\nval_S2:", val_S2.info())
    print(f"val_stylo -> num_instances: {len(val_stylo)} | len_vect: {len(val_stylo[0])}")
    print(f"val_stylo_2 -> num_instances: {len(val_stylo_2)} | len_vect: {len(val_stylo_2[0])}")
    print(f"val_bert_multi -> num_instances: {len(val_bert_multi)} | len_vect: {len(val_bert_multi[0])}")
    print(f"val_e5_multi -> num_instances: {len(val_e5_multi)} | len_vect: {len(val_e5_multi[0])}")
    print(f"val_xlm_multi -> num_instances: {len(val_xlm_multi)} | len_vect: {len(val_xlm_multi[0])}")
    
    train_emb = {
        "all": [], "bert_multi": [], "e5_multi": [], "xlm_multi": [], 
        "bert_multi_e5_multi": [], "bert_multi_xlm_multi": [], "e5_multi_xlm_multi": [],
        "bert_multi_e5_multi_stylo": [], "bert_multi_e5_multi_xlm_multi": [] 
    }
    val_emb = {
        "all": [], "bert_multi": [], "e5_multi": [], "xlm_multi": [], 
        "bert_multi_e5_multi": [], "bert_multi_xlm_multi": [], "e5_multi_xlm_multi": [],
        "bert_multi_e5_multi_stylo": [], "bert_multi_e5_multi_xlm_multi": [] 
    }
    
    for index in range(len(train_bert_multi)):
        train_emb["bert_multi"].append(train_bert_multi[index])
        train_emb["e5_multi"].append(train_e5_multi[index])
        train_emb["xlm_multi"].append(train_xlm_multi[index])
        train_emb["bert_multi_e5_multi"].append(train_bert_multi[index] + train_e5_multi[index])
        train_emb["bert_multi_e5_multi_stylo"].append(train_bert_multi[index] + train_e5_multi[index] + train_stylo[index])
        train_emb["bert_multi_xlm_multi"].append(train_bert_multi[index] + train_xlm_multi[index])
        train_emb["e5_multi_xlm_multi"].append(train_e5_multi[index] + train_xlm_multi[index])
        train_emb["bert_multi_e5_multi_xlm_multi"].append(train_bert_multi[index] + train_e5_multi[index] + train_xlm_multi[index])
        train_emb["all"].append(train_stylo[index] + train_stylo_2[index] + train_bert_multi[index] + train_e5_multi[index] + train_xlm_multi[index])
    
    for index in range(len(val_bert_multi)):
        val_emb["bert_multi"].append(val_bert_multi[index])
        val_emb["e5_multi"].append(val_e5_multi[index])
        val_emb["xlm_multi"].append(val_xlm_multi[index])
        val_emb["bert_multi_e5_multi"].append(val_bert_multi[index] + val_e5_multi[index])
        val_emb["bert_multi_e5_multi_stylo"].append(val_bert_multi[index] + val_e5_multi[index] + val_stylo[index] )
        val_emb["bert_multi_xlm_multi"].append(val_bert_multi[index] + val_xlm_multi[index])
        val_emb["e5_multi_xlm_multi"].append(val_e5_multi[index] + val_xlm_multi[index])
        val_emb["bert_multi_e5_multi_xlm_multi"].append(val_bert_multi[index] + val_e5_multi[index] + val_xlm_multi[index])
        val_emb["all"].append(val_stylo[index] + val_stylo_2[index] + val_bert_multi[index] + val_e5_multi[index] + val_xlm_multi[index])

    emb_type = 'bert_multi_e5_multi_xlm_multi'

    #train_all = train_emb[emb_type] + val_emb[emb_type]
    #label_all = train_S2['label'].tolist()  + val_S2['label'].tolist() 

    #print(f"\ntrain_emb_all -> num_instances: {len(train_all)} | len_vect: {len(train_all[0])} | len_label {len(label_all)}")        
    print(f"\ntrain_emb_all -> num_instances: {len(train_emb['all'])} | len_vect: {len(train_emb['all'][0])}")
    print(f"val_emb_all -> num_instances: {len(val_emb['all'])} | len_vect: {len(val_emb['all'][0])}")

    clf_models_dict = {
        #'LinearSVC': LinearSVC,
        #'LogisticRegression': LogisticRegression,
        'SGDClassifier': SGDClassifier,
        #'XGBClassifier': XGBClassifier,
    } 

    #for emb_type in train_emb.keys():
    #    print("***** emb_type: ", emb_type)
    for model_name in clf_models_dict.keys():
        print("\t model_name: ", model_name)
        #trained_model = gnn.train_ml_clf_model(clf_models_dict[model_name], train_all, label_all, val_emb[emb_type], val_S2['label'])
        #utils.save_data(trained_model, path=f'{utils.OUTPUT_DIR_PATH}experiments_ricardo_s2/', file_name=f'{model_name}_{emb_type}_full_train')
        trained_model = gnn.train_ml_clf_model(clf_models_dict[model_name], train_emb[emb_type], train_S2['label'], val_emb[emb_type], val_S2['label'])
        utils.save_data(trained_model, path=f'{utils.OUTPUT_DIR_PATH}experiments_ricardo_s2/', file_name=f'{model_name}_{emb_type}')


if __name__ == '__main__':
    #main()
    #extract_embeddings_subtask1()
    #feat_subtask1_ricardo()
    
    test_eval_subtask()
    #eval_subtask_ricardo()
    #train_clf_model_batch_subtask()
    
    #extract_embeddings_subtask2()
    #feat_subtask2_ricardo()



# ********* CMDs
# python main.py
# nohup bash main.sh >> logs/xxx.log &
# nohup python main.py >> logs/text_to_graph_transform_small.log &
# ps -ef | grep python | grep avaldez
# tail -f logs/experiments_cooccurrence_20240502062849.log 

















