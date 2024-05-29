import os
import sys
import joblib
import time
import numpy as np
import pandas as pd 
import logging
import traceback 
import math
from tqdm import tqdm
import torch
import networkx as nx 
import scipy as sp
import gensim
from scipy.sparse import coo_array 
from sklearn.datasets import fetch_20newsgroups
import gc
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch.nn import Linear, BatchNorm1d, ModuleList, LayerNorm
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, TopKPooling, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn.modules.module import Module
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import utils
import node_feat_init
from stylometric import StyloCorpus
import text2graph

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class BuildDataset():
    def __init__(self, graphs_data, subset, device, model_w2v=None, edge_features=False, nfi='llm', llm_finetuned_name=node_feat_init.LLM_HF_FINETUNED_NAME):
        self.graphs_data = graphs_data
        self.model_w2v = model_w2v
        self.llm_model = None
        self.device = device
        self.subset = subset
        self.nfi = nfi
        self.edge_features = edge_features
        self.llm_finetuned_name = llm_finetuned_name

    def process_dataset(self):
        #if self.nfi == 'llm' and True:
        #    dataset = [{'id': d['context']['id'], 'label': d['context']['target'], 'text': " ".join(list(d['graph'].nodes))} for d in self.graphs_data]
        #    self.llm_model = node_feat_init.llm_get_embbedings(dataset, subset=self.subset, emb_type='llm_word', device=self.device, save_emb=False)
        
        block = 1
        batch_size = utils.LLM_GET_EMB_BATCH_SIZE_DATALOADER
        num_batches = math.ceil(len(self.graphs_data) / batch_size)
        data_list = []
        for index, _ in enumerate(tqdm(range(num_batches))):
            graphs_data_batch = self.graphs_data[batch_size*(block-1) : batch_size*block]
            if self.nfi == 'llm':
                # Data storage/saved
                #self.llm_model = utils.read_json(file_path=path_file + f'pan24_{self.subset}_emb_batch_{block-1}.json')
                #self.llm_model = self.llm_model.to_dict('records')
                # Data In memory
                dataset = [{'id': d['context']['id'], 'label': d['context']['target'], 'text': " ".join(list(d['graph'].nodes))} for d in graphs_data_batch]
                self.llm_model = node_feat_init.llm_get_embbedings(dataset, subset=self.subset, emb_type='llm_word', device=self.device, save_emb=False, llm_finetuned_name=self.llm_finetuned_name)

            for index, g in enumerate(graphs_data_batch):
                #print(g['graph'])
                try:
                    # Get node features
                    node_feats = self.get_node_features(g, type=self.nfi)
                    # Get edge features
                    edge_attr = self.get_edge_features(g['graph'])
                    # Get adjacency info
                    edge_index = self.get_adjacency_info(g['graph'])
                    # Get labels info
                    label = self.get_labels(g["context"]["target"])
                    
                    #print(node_feats.shape, edge_index.shape, label.shape)
                    data = Data(
                        x = node_feats,
                        edge_index = edge_index,
                        edge_attr = edge_attr,
                        y = label,
                        pred = '',
                        context = g["context"]
                    )
                    data_list.append(data)
                except Exception as e:
                    logger.error('Error: %s', str(e))

            block += 1
        del self.llm_model
        return data_list
        

    def get_node_features(self, g, type='w2v'):
        if type == 'ohe':
            graph_node_feat = []
            for node in list(g['graph'].nodes):
                vector = np.zeros(len(self.vocab))
                vector[self.node_to_index[node]] = 1
                graph_node_feat.append(vector)
        
        if type == 'w2v':
            graph_node_feat = []
            for node in list(g['graph'].nodes):
                graph_node_feat.append(self.model_w2v.wv[node])

        if type == 'llm':
            graph_node_feat = []    
            for j in range(0,len(self.llm_model)):
                if g['context']['id'] == self.llm_model[j]['doc_id']:
                    n_f = self.llm_model[j]
                    for n in list(g['graph'].nodes):
                        if n in n_f['embedding']:
                            graph_node_feat.append(n_f['embedding'][n])
                        else:
                           g['graph'].remove_node(n)
    
        graph_node_feat = np.asarray(graph_node_feat)
        return torch.tensor(graph_node_feat, dtype=torch.float)

    def get_adjacency_info(self, g):
        adj = nx.to_scipy_sparse_array(g,  weight='weight', dtype=np.cfloat)
        adj_coo = sp.sparse.coo_array(adj)
        edge_indices = []
        for index in range(len(g.edges)):
            edge_indices += [[adj_coo.row[index], adj_coo.col[index]]]

        edge_indices = torch.tensor(edge_indices) 
        t = edge_indices.t().to(torch.long).view(2, -1)  
        #print("edge_index:", t.shape)
        return edge_indices.t().to(torch.long).view(2, -1)

    def get_edge_features(self, g):
        if self.edge_features:
            all_edge_feats = []
            for edge in g.edges(data=True):
                feats = edge[2]
                edge_feats = []
                # Feature 1: freq
                edge_feats.append(feats['freq'])
                # Feature 2: pmi
                edge_feats.append(feats['pmi'])
                # Append node features to matrix (twice, per direction)
                edge_feats = np.asarray(edge_feats)
                edge_feats = edge_feats/np.linalg.norm(edge_feats)
                all_edge_feats += [edge_feats]

            all_edge_feats = np.asarray(all_edge_feats)
            all_edge_feats = torch.tensor(all_edge_feats, dtype=torch.float)
            #print("edge_feat :", all_edge_feats.shape)
            return all_edge_feats
        else:
            return None


    def get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)


class GNN(torch.nn.Module):
    def __init__(self, 
        gnn_type='TransformerConv',
        num_features=768,
        hidden_channels=64,
        out_emb_size=256,
        num_classes=2,
        heads=1,
        dropout=0.5,
        pooling='gmeanp',
        batch_norm='BatchNorm1d',
        layers_convs=3,
        dense_nhid=64,
        edge_dim=None
    ):
        super(GNN, self).__init__()
        torch.manual_seed(1234567)

        # setting vars
        self.n_layers = layers_convs
        self.dropout_rate = dropout
        self.dense_neurons = dense_nhid
        self.batch_norm = batch_norm
        self.pooling = pooling
        self.out_emb_size = out_emb_size
        self.top_k_every_n = 2
        self.top_k_ratio = 0.5
        self.edge_dim = edge_dim

        # setting ModuleList
        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # select convolution layer
        GNN_LAYER_BY_NAME = {
            "GCNConv": GCNConv,
            "GATConv": GATConv,
            "GraphConv": GraphConv,
            "TransformerConv": TransformerConv,
        }
        conv_layer = GNN_LAYER_BY_NAME[gnn_type]
        if gnn_type in ['GATConv', 'TransformerConv']:
            self.support_edge_attr = True
        else:    
            self.support_edge_attr = False

        # Transformation layer
        if self.support_edge_attr:
            self.conv1 = conv_layer(num_features, hidden_channels, heads, edge_dim=self.edge_dim) 
        else:
            self.conv1 = conv_layer(num_features, hidden_channels, heads) 
        
        self.transf1 = Linear(hidden_channels*heads, hidden_channels)
        
        if batch_norm != None:
            #self.bn1 = BatchNorm1d(hidden_channels*heads)
            self.bn1 = BatchNorm1d(hidden_channels)

        # Other layers
        for i in range(self.n_layers):
            if self.support_edge_attr:
                self.conv_layers.append(conv_layer(hidden_channels, hidden_channels, heads, edge_dim=self.edge_dim))
            else:
                self.conv_layers.append(conv_layer(hidden_channels, hidden_channels, heads))
            
            self.transf_layers.append(Linear(hidden_channels*heads, hidden_channels))
            
            if batch_norm != None:
                #self.bn_layers.append(BatchNorm1d(hidden_channels*heads))
                self.bn_layers.append(BatchNorm1d(hidden_channels))
            if pooling == 'topkp':
                if i % self.top_k_every_n == 0:
                    #self.pooling_layers.append(TopKPooling(hidden_channels*heads, ratio=self.top_k_ratio))
                    self.pooling_layers.append(TopKPooling(hidden_channels, ratio=self.top_k_ratio))
            
        # Linear layers opt 1
        self.linear1 = Linear(hidden_channels, self.dense_neurons) 
        self.linear2 = Linear(self.dense_neurons, int(self.dense_neurons)//2)
        self.linear3 = Linear(int(self.dense_neurons)//2, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # Initial transformation
        if self.support_edge_attr:
            x = self.conv1(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
        x = torch.relu(self.transf1(x))
        #x = x.relu()
        if self.batch_norm != None:
            x = self.bn1(x)

        # Holds the intermediate graph representations only for TopKPooling
        global_representation = []

        # iter trought n_layers, apply convs
        for i in range(self.n_layers):
            if self.support_edge_attr:
                x = self.conv_layers[i](x, edge_index, edge_attr)
            else:
                x = self.conv_layers[i](x, edge_index)
            #x = x.relu()
            x = torch.relu(self.transf_layers[i](x))

            if self.batch_norm != None:
                x = self.bn_layers[i](x)
            
            #x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if self.pooling == 'topkp':
                if i % self.top_k_every_n == 0 or i == self.n_layers:
                    if self.support_edge_attr:
                        x, edge_index, edge_attr, batch, _, _  = self.pooling_layers[int(i/self.top_k_every_n)](x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                    else:
                        x, edge_index, _, batch, _, _  = self.pooling_layers[int(i/self.top_k_every_n)](x=x, edge_index=edge_index, batch=batch)
                    global_representation.append(global_mean_pool(x, batch))

        # Aplpy graph pooling
        if self.pooling == 'topkp':
            x = sum(global_representation)
        elif self.pooling == 'gmeanp':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'gmaxp':
            x = global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        
        out = torch.relu(self.linear1(x))
        out = F.dropout(out, p=self.dropout_rate, training=self.training) 
        out = torch.relu(self.linear2(out))
        out = F.dropout(out, p=self.dropout_rate, training=self.training) 
        out = self.linear3(out)
        
        return out, x 
        #return x


class NeuralNetwork(torch.nn.Module):
    def __init__(self, in_channels, nhid, out_ch, layers_num):
        super(NeuralNetwork,self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_channels, nhid),
            torch.nn.ReLU(),
            torch.nn.Linear(nhid, nhid),
            torch.nn.ReLU(),
            torch.nn.Linear(nhid, nhid),
            torch.nn.ReLU(),
            torch.nn.Linear(nhid, out_ch),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_ml_clf_model(algo_clf, train_data, train_labels, val_data, val_labels):
    model = CalibratedClassifierCV(algo_clf()) #n_jobs=-3
    model.fit(train_data, train_labels)
    predicted = model.predict(val_data)
    print('\t Accuracy:', np.mean(predicted == val_labels.view(1,-1)[0].numpy()))
    return model


def train_dense_rrnn_clf_model(dense_model, device, train_loader, val_data, val_labels):
    learning_rate = 0.0001
    early_stopper = EarlyStopper(patience=20, min_delta=0)
    optimizer = torch.optim.Adam(dense_model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss() # BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
    sigmoid = torch.nn.Sigmoid()
    dense_model = dense_model.to(device) 
    print_preds_test = False
    train_loss = 0.0
    epochs = 100
    avg_acc = 0.0
    best_acc_epoch = 0
    best_acc = 0.0
    steps = 0
    for epoch in range(1, epochs):
        dense_model.train()
        for features, labels, in train_loader:
            features.to(device)           
            labels.to(device)           
            #print(batch.x.shape, batch.y)
            logits = dense_model(features)
            #logits = logits.argmax(dim=1)
            #print(logits)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            steps += 1
            
        acc, val_loss, preds_test = test_train_concat_emb(dense_model, criterion, val_data, val_labels, epoch)
        avg_acc += acc
        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch
        
        #print(f"Epoch: {epoch} | train_loss: {loss.item():4f} | val_loss: {val_loss:4f} | acc: {acc:4f} | avg_acc: {avg_acc/(epoch+1):4f} | best_acc (epoch {best_acc_epoch}): {best_acc:4f})")
        if early_stopper.early_stop(val_loss): 
            print('Early stopping fue to not improvement!') 
            print(f"Epoch: {epoch} | train_loss: {loss.item():4f} | val_loss: {val_loss:4f} | acc: {acc:4f} | avg_acc: {avg_acc/(epoch+1):4f} | best_acc (epoch {best_acc_epoch}): {best_acc:4f})")
            break

        if epoch == 10 and print_preds_test:
            print(preds_test['outs'])

    return dense_model


def test_train_concat_emb(dense_model, criterion, val_data, val_labels, epoch):
    sigmoid = torch.nn.Sigmoid()
    dense_model.eval()
    targ = val_labels.float()
    preds_test = {}
    with torch.no_grad():
        logits = dense_model.forward(val_data)      
        loss = criterion(logits, targ)    

        acc = (logits.round() == targ).float().mean()
        acc = float(acc)

        if epoch == 10:
            #pred_probab = sigmoid(logits)
            #y_pred = pred_probab.argmax(1)      
            #print(logits)
            #print(logits.round())
            #print(targ)
            preds_test = {'preds': logits.round().cpu().numpy().tolist(),'outs': logits.cpu().numpy().tolist()}
        
        return acc, loss, preds_test


def word2vect(graph_data, num_features):
    sent_w2v = []
    for g in graph_data:
        sent_w2v.append(list(g['graph'].nodes))
    model_w2v = gensim.models.Word2Vec(sent_w2v, min_count=1,vector_size=num_features, window=3)
    return model_w2v


def gnn_model(
                train_loader, val_loader, metrics, device,
                epoch_num, gnn_type, num_features, 
                hidden_channels, learning_rate, 
                gnn_dropout, gnn_pooling, gnn_batch_norm,
                gnn_layers_convs, gnn_heads, gnn_dense_nhid,
                num_classes, edge_dim
            ):

    model = GNN(
        gnn_type=gnn_type,
        num_features=num_features, 
        hidden_channels=hidden_channels, 
        num_classes=num_classes,
        heads=gnn_heads,
        dropout=gnn_dropout, 
        pooling=gnn_pooling, 
        batch_norm=gnn_batch_norm,
        layers_convs=gnn_layers_convs, 
        dense_nhid=gnn_dense_nhid,
        edge_dim=edge_dim
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    logger.info("model: %s", str(model))
    logger.info("device: %s", str(device))
    model = model.to(device)
    early_stopper = EarlyStopper(patience=30, min_delta=0)
    best_train_embeddings = None
    best_val_embeddings = None
    best_val_score =  0
    best_epoch_score =  0
    avg_val_score =  0
    epochs_cnt =  0
    
    torch.cuda.empty_cache()
    gc.collect()

    for epoch in range(1, epoch_num):
        epochs_cnt += 1
        train_loss, train_embeddings, _ = train(model, criterion, optimizer, train_loader, device=device)
        _, train_acc, _, _ = test(model, criterion, train_loader, device=device)
        val_loss, val_acc, val_embeddings, _ = test(model, criterion, val_loader , device=device)
        print(f'Epoch: {epoch:03d} | Train Loss {train_loss} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        avg_val_score += val_acc
        if val_acc > best_val_score:
            best_val_score = val_acc
            best_epoch_score = epoch
            best_train_embeddings = train_embeddings
            best_val_embeddings = val_embeddings
        
        metrics['_epoch_stop'] = epoch
        metrics['_train_loss'] = train_loss
        metrics['_val_loss'] = val_loss
        metrics['_train_acc'] = train_acc
        metrics['_val_last_acc'] = val_acc
        if early_stopper.early_stop(val_loss): 
            print('Early stopping fue to not improvement!')            
            break

    #metrics['_best_metrics'] = {'best_val_score': best_val_score,'best_epoch_score': best_epoch_score, 'avg_val_score': avg_val_score/epochs_cnt}
    metrics['_val_best_acc'] = best_val_score
    metrics['_val_best_epoch_acc'] = best_epoch_score
    metrics['_val_avg_acc'] = avg_val_score/epochs_cnt
    print(f'-----> Best Val Score: {best_val_score:.4f} in epoch: {best_epoch_score} | Avg Val Score: {avg_val_score/epochs_cnt:.4f}')
    return model, metrics, best_train_embeddings, best_val_embeddings


def test(model, criterion, loader, device='cpu'):
    model.eval()
    correct = 0
    test_loss = 0.0
    steps = 0
    pred_loader = []
    embeddings_data = []

    with torch.no_grad():
        for step, data in enumerate(loader):  # Iterate in batches over the training/test dataset.
            data.to(device)
            #print('testing batch...', step)
            out, embeddings = model(data.x, data.edge_index, data.edge_attr, data.batch)  
            embeddings_data.append({'batch': step, 'doc_id': data.context['id'], 'labels': data.y, 'embedding': embeddings})
            loss = criterion(out, data.y)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            data.pred = pred
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            test_loss += loss.item()
            steps += 1
            pred_loader.append(data)
        return test_loss / steps, correct / len(loader.dataset), embeddings_data, pred_loader  # Derive ratio of correct predictions.


def train(model, criterion, optimizer, loader, device='cpu'):
    model.train()
    train_loss = 0.0
    steps = 0
    embeddings_data = []
    for step, data in enumerate(loader):  # Iterate in batches over the training dataset.
        data.to(device) 
        #print('training batch...', step)
        out, embeddings = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
        embeddings_data.append({'batch': step, 'doc_id': data.context['id'], 'labels': data.y, 'embedding': embeddings})
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        train_loss += loss.item()
        steps += 1
    return train_loss / steps, embeddings_data, loader


def graph_neural_network(
        exp_file_name,
        dataset_partition, 
        exp_file_path, 
        graph_trans, 
        nfi, 
        cut_off_dataset, 
        t2g_instance, 
        train_text_docs, 
        val_text_docs,
        device,
        edge_features,
        llm_finetuned_name,
        edge_dim=2,
    ):

    num_classes = 2
    num_features = 768 # llm: 768 | w2v: 768
    if llm_finetuned_name == 'andricValdez/multilingual-e5-large-finetuned-autext24':
        num_features = 1024 # llm: 768 | w2v: 768

    if not edge_features:
        edge_dim = None

    if graph_trans: 
        # Text 2 Graph train data
        graphs_train_data = utils.t2g_transform(train_text_docs, t2g_instance)
        utils.save_data(graphs_train_data, path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_train_{dataset_partition}')

        # Text 2 Graph test data
        graphs_val_data = utils.t2g_transform(val_text_docs, t2g_instance)
        utils.save_data(graphs_val_data, path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_val_{dataset_partition}')
        
        # Feat Init - Word2vect Model
        model_w2v = word2vect(graph_data=graphs_train_data + graphs_val_data, num_features=num_features)
        utils.save_data(model_w2v, path=f'{utils.OUTPUT_DIR_PATH}w2v_models/', file_name=f'model_w2v_{dataset_partition}')

    else:
        ...
        graphs_train_data = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_train_{dataset_partition}')  
        graphs_val_data = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}graphs/', file_name=f'graphs_val_{dataset_partition}')  
        model_w2v = utils.load_data(path=f'{utils.OUTPUT_DIR_PATH}w2v_models/', file_name=f'model_w2v_{dataset_partition}')

    print("graphs_train_data: ", len(graphs_train_data))
    print("graphs_val_data: ", len(graphs_val_data))
    print('device: ', device)

    #******************* TRAIN and GET GNN Embeddings
    train_build_dataset = BuildDataset(graphs_train_data[:], subset='train', device=device, model_w2v=model_w2v, edge_features=edge_features, nfi=nfi, llm_finetuned_name=llm_finetuned_name)
    train_dataset = train_build_dataset.process_dataset()
    val_build_dataset = BuildDataset(graphs_val_data[:], subset='val', device=device, model_w2v=model_w2v, edge_features=edge_features, nfi=nfi, llm_finetuned_name=llm_finetuned_name)
    val_dataset = val_build_dataset.process_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
    init_metrics = {'_epoch_stop': 0,'_train_loss': 0,'_val_loss': 0,'_train_acc': 0,'_val_last_acc': 0,'_test_acc': 0,'_exec_time': 0,}

    torch.cuda.empty_cache()
    gc.collect()

    metrics = init_metrics.copy()
    train_model_args = {
        'train_loader': train_loader, 
        'val_loader': val_loader,  
        'metrics': metrics, 
        'device': device,
        'epoch_num': 100,
        'gnn_type': 'TransformerConv', # GCNConv, GATConv, TransformerConv
        'num_features': num_features, 
        'hidden_channels': 256, 
        'learning_rate': 0.00001, # W2V: 0.0001 | LLM: 0.00001
        'gnn_dropout': 0.5,
        'gnn_pooling': 'gmeanp', # gmeanp, gmaxp, topkp
        'gnn_batch_norm': 'BatchNorm1d', # None, BatchNorm1d
        'gnn_layers_convs': 4,
        'gnn_heads': 3, 
        'gnn_dense_nhid': 128,
        'num_classes': 2,
        'edge_dim': edge_dim, # None, 2 
    }
    model, metrics, embeddings_train_gnn, embeddings_val_gnn = gnn_model(**train_model_args)
    
    configs = {"train_model_args": str(train_model_args), "metrics": str(metrics), "model": str(model)}
    torch.save(model, f'{exp_file_path}/model_GNN_{exp_file_name}_{dataset_partition}.pt')
    utils.save_llm_embedings(embeddings_data=embeddings_train_gnn, emb_type='gnn', file_path=f"{exp_file_path}/embeddings_gnn/autext24_train_emb_batch_")
    utils.save_llm_embedings(embeddings_data=embeddings_val_gnn, emb_type='gnn', file_path=f"{exp_file_path}/embeddings_gnn/autext24_val_emb_batch_")
    utils.save_json(configs, file_path=f'{exp_file_path}configs.json')



# *************************************** EXPERIMENTS IN BATCHES
def graph_neural_network_batch(train_text_docs, val_text_docs, experiments_path_file):
    
    print('*** INIT EXPERIMENTS')
    experiments_data = utils.read_csv(f'{experiments_path_file}')
    print(experiments_data.info())

    init_metrics = {'_epoch_stop': 0,'_train_loss': 0,'_val_loss': 0,'_train_acc': 0,'_val_last_acc': 0,'_test_acc': 0,'_exec_time': 0,}
    num_classes = 2
    nfi = 'llm'
    cuda_num = 0
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    for index, row in experiments_data.iterrows():
        print("******************************************* Running experiment with ID: ", row['id'])
        start = time.time()

        if row['_done'] == True or row['_done'] == 'True':
            print('Experiment already DONE')
            continue

        t2g_instance = text2graph.Text2Graph(
            graph_type = row['graph_edge_type'],
            window_size = row['window_size'], 
            apply_prep = True, 
            steps_preprocessing = {
                "to_lowercase": True,
                "handle_blank_spaces": True,
                "handle_html_tags": True,
                "handle_special_chars": row['prep_espcial_chars'],
                "handle_stop_words": row['prep_stop_words'],
            },
            language = 'en',
        )

        graphs_train_data = utils.t2g_transform(train_text_docs, t2g_instance)
        graphs_val_data = utils.t2g_transform(val_text_docs, t2g_instance)
        num_features = 768 
        if row['graph_node_feat_init'] == 'andricValdez/multilingual-e5-large-finetuned-autext24':
            num_features = 1024 
       
        edge_dim = 2
        if not row['gnn_edge_attr']:
            edge_dim = None

        try:
            train_build_dataset = BuildDataset(graphs_train_data[:], subset='train', device=device, model_w2v=None, edge_features=row['gnn_edge_attr'], nfi=nfi, llm_finetuned_name=row['graph_node_feat_init'])
            train_dataset = train_build_dataset.process_dataset()
            val_build_dataset = BuildDataset(graphs_val_data[:], subset='val', device=device, model_w2v=None, edge_features=row['gnn_edge_attr'], nfi=nfi, llm_finetuned_name=row['graph_node_feat_init'])
            val_dataset = val_build_dataset.process_dataset()
            
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

            metrics = init_metrics.copy()
            train_model_args = {
                'train_loader': train_loader, 
                'val_loader': val_loader,  
                'metrics': metrics, 
                'device': device,
                'epoch_num': row['epoch_num'],
                'gnn_type': row['gnn_type'], # GCN, GAT, TransformerConv
                'num_features': num_features, 
                'hidden_channels': row['gnn_nhid'], 
                'learning_rate': row['gnn_learning_rate'], # W2V: 0.0001 | LLM: 0.00001
                'gnn_dropout': row['gnn_dropout'],
                'gnn_pooling': row['gnn_pooling'], # gmeanp, gmaxp, topkp
                'gnn_batch_norm': row['gnn_batch_norm'], # None, BatchNorm1d
                'gnn_layers_convs': row['gnn_layers_convs'],
                'gnn_heads': row['gnn_heads'], 
                'gnn_dense_nhid': row['gnn_dense_nhid'],
                'num_classes': num_classes,
                'edge_dim': edge_dim, # None, 2 
            }

            model, metrics, embeddings_train_gnn, embeddings_val_gnn = gnn_model(**train_model_args)
            metrics['_done'] = True
        except Exception as err:
            metrics['_done'] = 'Error'
            metrics['_desc'] = str(err)
            print("traceback: ", str(traceback.format_exc()))
            print(f"An error ocurred running experiment {row['id']}: ", err)
        finally:
            metrics['_exec_time'] = time.time() - start
            for key, value in metrics.items():
                experiments_data.loc[index, key] = value
            utils.save_csv(experiments_data, file_path=f'{experiments_path_file}')

    print('*** DONE EXPERIMENTS') 

