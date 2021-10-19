import torch
import pandas as pd
import numpy as np
import os.path as osp

from torch_geometric.utils import (negative_sampling, add_self_loops)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

import random
import numpy as np
import seal_algorithm as seal
import time
import pickle

class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

class HashEncoder(object):
    # The 'HashEncoder' takes the column values and hashes them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype
        self.change = True

    def __call__(self, df):
        if self.change:
            self.blah = np.asarray([hash(x) for x in df.values])
            self.change = False
        return torch.from_numpy(self.blah).view(-1, 1).to(self.dtype)

def get_initial_graph_data(node_path,edge_path):
	node_encoders = {cname:IdentityEncoder() for cname in ['a','b','c','d','e','ff']}
	edge_encoders = {cname:IdentityEncoder() for cname in ['a','b','c','d']}
	node_df = pd.read_csv(node_path,index_col="idx")
	nodes = [encoder(node_df[col]) for col,encoder in node_encoders.items()]
	nodes_tf = torch.cat(nodes,dim=-1)

	edge_df = pd.read_csv(edge_path)
	src = [i for i in edge_df['src']]
	dst = [i for i in edge_df['dst']]
	edge_index = torch.tensor([src,dst])
	edge_traits = [encoder(edge_df[col]) for col, encoder in edge_encoders.items()]
	edges_tf = torch.cat(edge_traits,dim=-1)
	data = Data(x=nodes_tf,edge_index=edge_index,edge_attr=edges_tf,num_nodes=node_df.shape[0])
	print("data has",len(nodes_tf),"nodes, and",len(edge_index[0]),"edges")
	data = ToUndirected()(data)
	return data

def sortpair(a,b):
	if a<b:
		return (a,b)
	else:
		return (b,a)

def forget_the_future(node_feats,edge_idxs,edge_feats,source,destination):
	max_node = max(source,destination)
	node_feats = node_feats[:max_node+1]
	edge_info = [(i,src,dst) for (i,(src,dst)) in enumerate(zip(edge_idxs[0],edge_idxs[1])) if src<=max_node and dst<=max_node and sortpair(source,destination)!=sortpair(src,dst) ]
	edge_idx_idxs = [i for (i,src,dst) in edge_info]
	edge_srcs = [src for (i,src,dst) in edge_info]
	edge_dsts = [dst for (i,src,dst) in edge_info]
	edge_feats = edge_feats[edge_idx_idxs]
	data = Data(x=node_feats,edge_index=edge_idxs[:,edge_idx_idxs],edge_attr=edge_feats,num_nodes=max_node+1)
	return data

def edge_tensor_to_list(edge_idx):
	return list(zip(list(edge_idx[0]),list(edge_idx[1])))

def withlabel(edges,lab):
	alllab = [lab for _ in range(len(edges))]
	return list(zip(edges,alllab))

def collect_pos_neg_examples(data):
	pos_edges = data.edge_index
	num_pos = pos_edges.size(1)
	neg_edges = negative_sampling(pos_edges,num_nodes=data.num_nodes,num_neg_samples=num_pos)
	graphs = []
	l = withlabel(edge_tensor_to_list(pos_edges),1)+withlabel(edge_tensor_to_list(neg_edges),0)
	print(len(l))
	t = time.time()
	for ((src,dst),lab) in l:
		foggedgraph = forget_the_future(data.x,data.edge_index,data.edge_attr,src,dst)
		sealed_graph = seal.seal_data(foggedgraph,src,dst,lab)
		graphs.append(sealed_graph)
	return graphs

def edge_in(edges,src,dst):
	if (src,dst) in edges or (dst,src) in edges:
		return 1
	return 0

def all_from_last_node(data):
	graphs = []
	pos_edges = data.edge_index
	pos_edges_list = edge_tensor_to_list(pos_edges)
	for a in range(data.num_nodes):
		for b in [list(range(data.num_nodes))[-3]]:
			(a,b) = (torch.tensor(a),torch.tensor(b))
			foggedgraph = forget_the_future(data.x,data.edge_index,data.edge_attr,a,b)
			sealed_graph = seal.seal_data(foggedgraph,a,b,edge_in(pos_edges_list,a,b))
			graphs.append(sealed_graph)
	print("the true edges are",[i-2 for (i,d) in enumerate(graphs) if d.y == 1])
	print(sum([data.y for data in graphs]))
	return graphs

def all_positives(data):
	graphs = []
	pos_edges = data.edge_index
	pos_edges_list = edge_tensor_to_list(pos_edges)

	print("getting positives")
	for (a,b) in pos_edges_list:
		print("("+str(a)+","+str(b)+")")
		foggedgraph = forget_the_future(data.x,data.edge_index,data.edge_attr,a,b)
		sealed_graph = seal.seal_data(foggedgraph,a,b,edge_in(pos_edges_list,a,b))
		graphs.append(sealed_graph)
	print(sum([data.y for data in graphs]))
	return graphs

def all_negatives(data):
	graphs = []
	pos_edges = data.edge_index
	pos_edges_list = edge_tensor_to_list(pos_edges)
	neg_edges_list = []
	for a in range(data.num_nodes):
		for b in range(data.num_nodes):
			(a,b) = (torch.tensor(a),torch.tensor(b))
			if (a,b) not in pos_edges_list:
				neg_edges_list.append((a,b))
	print("getting positives")
	for (a,b) in neg_edges_list:
		print("("+str(a)+","+str(b)+")")
		foggedgraph = forget_the_future(data.x,data.edge_index,data.edge_attr,a,b)
		sealed_graph = seal.seal_data(foggedgraph,a,b,1)
		graphs.append(sealed_graph)
	print(len(graphs))
	return graphs

def get_dataset(folder):
	pickle_path = folder+"sealed.pkl"
	if osp.exists(pickle_path):
		f = open(pickle_path,'rb')
		dataset_raw = pickle.load(f)
		dataset = [Data.from_dict(d) for d in dataset_raw]
	else:
		f = open(pickle_path,'wb')
		table = get_initial_graph_data(folder+"node.csv",folder+"edge.csv")
		dataset = collect_pos_neg_examples(table)
		pickle.dump([data.to_dict() for data in dataset],f)
	f.close()
	return dataset

def get_predict_data(folder):
	table = get_initial_graph_data(folder+"node.csv",folder+"edge.csv")
	return all_from_last_node(table)

def get_positives(folder):
	table = get_initial_graph_data(folder+"node.csv",folder+"edge.csv")
	return all_positives(table)

def get_negatives(folder):
	table = get_initial_graph_data(folder+"node.csv",folder+"edge.csv")
	return all_negatives(table)
# def get_input(folder):
# 	table = get_initial_graph_data(folder+"node.csv",folder+"edge.csv")


def create_dataset():
	table_datasets = []
	for x in range(50):
		folder = "raw_data/e"+str(x)+"/"
		print(folder)
		dataset = get_dataset(folder)
		table_datasets.append(dataset)
	return table_datasets
