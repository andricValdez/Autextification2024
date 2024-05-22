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

from scipy.sparse import coo_array 
from sklearn.datasets import fetch_20newsgroups
from torch_geometric.data import DataLoader
from torch.nn import Linear, BatchNorm1d, ModuleList, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool

import utils
import node_feat_init

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    def __init__(self, graphs_data, model_w2v, device='cpu', nfi='llm'):
        self.graphs_data = graphs_data
        self.model_w2v = model_w2v
        self.llm_model = None
        self.nfi = nfi
        self.device = device

    def process_dataset(self):
        if self.nfi == 'llm':
            dataset = [{'id': d['context']['id'], 'label': d['context']['target'], 'text': " ".join(list(d['graph'].nodes))} for d in self.graphs_data]
            self.llm_model = node_feat_init.llm_get_embbedings(dataset, self.device)

        data_list = []
        for index, g in enumerate(tqdm(self.graphs_data)):
            # Get node features
            node_feats = self.get_node_features(g, type=self.nfi)
            # Get edge features
            edge_feats = self.get_edge_features(g['graph'])
            # Get adjacency info
            edge_index = self.get_adjacency_info(g['graph'])
            # Get labels info
            label = self.get_labels(g["context"]["target"])
            
            #print(node_feats.shape, edge_index.shape, label.shape)
            data = Data(
                x = node_feats,
                edge_index = edge_index,
                y = label,
                pred = '',
                context = g["context"],
                #edge_attr = edge_feats,
            )
            data_list.append(data)
        return data_list
        
    def get_node_features(self, g, type='ohe'):

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
                if g['context']['id'] == self.llm_model[j]['id']:
                    n_f = self.llm_model[j]
                    for n in list(g['graph'].nodes):
                        if n in n_f['embeddings']:
                            graph_node_feat.append(n_f['embeddings'][n])
                        else:
                           g['graph'].remove_node(n)
    
        graph_node_feat = np.asarray(graph_node_feat)
        return torch.tensor(graph_node_feat, dtype=torch.float)

    def get_edge_features(self, g):
        return None

    def get_adjacency_info(self, g):
        adj = nx.to_scipy_sparse_array(g,  weight='weight', dtype=np.cfloat)
        adj_coo = sp.sparse.coo_array(adj)
        edge_indices = []
        for index in range(adj_coo.shape[0]):
            edge_indices += [[adj_coo.row[index], adj_coo.col[index]]]

        edge_indices = torch.tensor(edge_indices) 
        return edge_indices.t().to(torch.long).view(2, -1)

    def get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

class GNN(torch.nn.Module):
    def __init__(self, 
        gnn_type='TransformerConv',
        num_features=768,
        hidden_channels=64,
        num_classes=2,
        heads=1,
        dropout=0.5,
        pooling='gmeanp',
        batch_norm='BatchNorm1d',
        layers_convs=3,
        dense_nhid=64
    ):
        super(GNN, self).__init__()

        # setting vars
        self.n_layers = layers_convs
        self.dropout_rate = dropout
        self.dense_neurons = dense_nhid
        self.batch_norm = batch_norm
        self.pooling = pooling
        self.top_k_every_n = 2
        self.top_k_ratio = 0.5

        # setting ModuleList
        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # select convolution layer
        if gnn_type == 'GCN':
            conv_layer = GCNConv
        elif gnn_type == 'GAT':
            conv_layer = GATConv
        elif gnn_type == 'TransformerConv':
            conv_layer = TransformerConv
        else:
            conv_layer = TransformerConv

        # Transformation layer
        self.conv1 = conv_layer(num_features, hidden_channels, heads) 
        #self.transf1 = Linear(hidden_channels*heads, hidden_channels)
        if batch_norm != None:
            self.bn1 = BatchNorm1d(hidden_channels*heads)

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(conv_layer(hidden_channels*heads, hidden_channels, heads))
            #self.transf_layers.append(Linear(hidden_channels*heads, hidden_channels))
            if batch_norm != None:
                self.bn_layers.append(BatchNorm1d(hidden_channels*heads))
            if pooling == 'topkp':
                if i % self.top_k_every_n == 0:
                    self.pooling_layers.append(TopKPooling(hidden_channels*heads, ratio=self.top_k_ratio))
            
        # Linear layers
        #self.linear2 = Linear(self.dense_neurons, num_classes)  
        self.linear1 = Linear(hidden_channels*heads, self.dense_neurons)
        self.linear2 = Linear(self.dense_neurons, int(self.dense_neurons/2))  
        self.linear3 = Linear(int(self.dense_neurons/2), num_classes)


    def forward(self, x, edge_index, batch):
        # Initial transformation
        x = self.conv1(x, edge_index)
        #x = torch.relu(self.transf1(x))
        x = x.relu()
        if self.batch_norm != None:
            x = self.bn1(x)

        # Holds the intermediate graph representations only for TopKPooling
        global_representation = []

        # iter trought n_layers, apply convs
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = x.relu()

            #x = torch.relu(self.transf_layers[i](x))
            if self.batch_norm != None:
                x = self.bn_layers[i](x)
            if self.pooling == 'topkp':
                if i % self.top_k_every_n == 0 or i == self.n_layers:
                    x, edge_index, _, batch, _, _  = self.pooling_layers[int(i/self.top_k_every_n)](x=x, edge_index=edge_index, batch=batch)
                    global_representation.append(global_mean_pool(x, batch))
    
        # Aplpy pooling
        if self.pooling == 'topkp':
            x = sum(global_representation)
        elif self.pooling == 'gmeanp':
            x = global_mean_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        # Final classification layer
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        out = self.linear3(x)
        return out, x

def word2vect(graph_data, num_features):
    sent_w2v = []
    for g in graph_data:
        sent_w2v.append(list(g['graph'].nodes))
    model_w2v = gensim.models.Word2Vec(sent_w2v, min_count=1,vector_size=num_features, window=3)
    return model_w2v

def gnn_train_model(
                train_loader, val_loader, metrics, device,
                epoch_num, gnn_type, num_features, 
                hidden_channels, learning_rate, 
                gnn_dropout, gnn_pooling, gnn_batch_norm,
                gnn_layers_convs, gnn_heads, gnn_dense_nhid,
                num_classes, early_stopper
            ):

    start = time.time()

    #model = GAT(num_features=num_features, hidden_channels=hidden_channels, num_classes=num_classes, heads=gnn_heads)
    
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
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    logger.info("model: %s", str(model))
    logger.info("device: %s", str(device))
    model = model.to(device)
    embeddings = None

    for epoch in range(1, epoch_num):
        train_loss, embeddings, _ = train(model, criterion, optimizer, train_loader, device=device)
        _, train_acc, _ = test(model, criterion, train_loader, device=device)
        val_loss, val_acc, _ = test(model, criterion, val_loader , device=device)
        print(f'Epoch: {epoch:03d} | Train Loss {train_loss} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        metrics['_epoch_stop'] = epoch
        metrics['_train_loss'] = train_loss
        metrics['_val_loss'] = val_loss
        metrics['_train_acc'] = train_acc
        metrics['_val_acc'] = val_acc
        if early_stopper.early_stop(val_loss): 
            print('Early stopping fue to not improvement!')            
            break

    end = time.time()
    metrics['_exec_time'] = end - start
    return model, embeddings, metrics


def test(model, criterion, loader, device='cpu'):
    model.eval()
    correct = 0
    test_loss = 0.0
    steps = 0
    pred_loader = []
    for step, data in enumerate(loader):  # Iterate in batches over the training/test dataset.
        data.to(device)
        #print('testing batch...', step)
        out, embedding = model(data.x, data.edge_index, data.batch)  
        loss = criterion(out, data.y)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        data.pred = pred
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        #print('pred: ', pred)
        #print('data.y: ', data.y)
        #print('correct: ', correct)
        test_loss += loss.item()
        steps += 1
        pred_loader.append(data)
    return test_loss / steps, correct / len(loader.dataset), pred_loader  # Derive ratio of correct predictions.


def train(model, criterion, optimizer, loader, device='cpu'):
    model.train()
    train_loss = 0.0
    steps = 0
    for step, data in enumerate(loader):  # Iterate in batches over the training dataset.
        data.to(device) 
        #print('training batch...', step)
        out, embeddings = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        train_loss += loss.item()
        steps += 1
    return train_loss / steps, embeddings, loader

def graph_neural_network(dataset, graph_trans, nfi, cut_off, t2g_instance, train_text_docs, test_text_docs):
    #dataset = 'autext24' # autext24
    cut_percentage_dataset = 100
    num_classes = 2
    num_features = 768 # llm: 768 | w2v: 150,300
    if nfi == 'w2v':
        num_features = 300
    if nfi == 'llm':
        num_features = 768 

    if graph_trans:
        # Text 2 Graph train data
        graphs_train_data = utils.t2g_transform(train_text_docs, t2g_instance, cut_off=cut_off)
        utils.save_data(graphs_train_data, path=utils.OUTPUTS_PATH, file_name=f'graphs_train_{dataset}')

        # Text 2 Graph test data
        graphs_val_data = utils.t2g_transform(test_text_docs, t2g_instance, cut_off=cut_off)
        utils.save_data(graphs_val_data, path=utils.OUTPUTS_PATH, file_name=f'graphs_test_{dataset}')
        
        # Feat Init - Word2vect Model
        model_w2v = word2vect(graph_data=graphs_train_data + graphs_val_data, num_features=num_features)
        utils.save_data(model_w2v, path=utils.OUTPUTS_PATH, file_name=f'model_w2v_{dataset}')

    else:
        model_w2v = utils.load_data(path=utils.OUTPUTS_PATH, file_name=f'model_w2v_{dataset}')
        graphs_train_data = utils.load_data(path=utils.OUTPUTS_PATH, file_name=f'graphs_train_{dataset}')  
        graphs_val_data = utils.load_data(path=utils.OUTPUTS_PATH, file_name=f'graphs_test_{dataset}')  
    
    cuda_num = 1
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    train_build_dataset = BuildDataset(graphs_train_data[:], model_w2v, device=device, nfi=nfi)
    train_dataset = train_build_dataset.process_dataset()
    val_build_dataset = BuildDataset(graphs_val_data[:], model_w2v, device=device, nfi=nfi)
    val_dataset = val_build_dataset.process_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    init_metrics = {'_epoch_stop': 0,'_train_loss': 0,'_val_loss': 0,'_train_acc': 0,'_val_acc': 0,'_test_acc': 0,'_exec_time': 0,}

    # *************************************** EXPERIMENTS ONE RUN
    exp_file_name = 'ID684_W2V'  # ID440_LLM, ID440_W2V, ID246_W2V, ID684_W2V
    exp_file_path = utils.OUTPUTS_PATH + f'best_models/{exp_file_name}_{dataset}/'
    utils.create_dir(dir_path=utils.OUTPUTS_PATH + 'best_models/')
    utils.create_dir(dir_path=exp_file_path)

    metrics = init_metrics.copy()
    model, embeddings, metric = gnn_train_model(
        train_loader=train_loader, 
        val_loader=val_loader, 
        metrics=metrics, 
        device=device,
        epoch_num=100,
        gnn_type='GAT', # GCN, GAT, TransformerConv
        num_features=num_features, 
        hidden_channels=128, 
        learning_rate=0.001, 
        gnn_dropout=0.5,
        gnn_pooling='gmeanp', # gmeanp, topkp
        gnn_batch_norm='BatchNorm1d', # None, BatchNorm1d
        gnn_layers_convs=3,
        gnn_heads=3, 
        gnn_dense_nhid=64,
        num_classes=2,
        early_stopper = EarlyStopper(patience=20, min_delta=0)
    )
    torch.save(model, f'{exp_file_path}/model_{exp_file_name}_{dataset}.pt')
    utils.save_data(embeddings, path=f'{exp_file_path}/', file_name=f"embeddings_{exp_file_name}_{dataset}")
