import math
import numpy as np
import torch
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU, Sequential, BatchNorm1d as BN)
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, SAGEConv, GINConv, 
                                global_sort_pool, global_add_pool, global_mean_pool)

config = {"max_z":1000,"hidden_channels":128,"k":30,'lr':0.0001,'num_layers':4}

class DGCNN(torch.nn.Module):
	def __init__(self):
		super(DGCNN,self).__init__()
		hidden_channels = config['hidden_channels']
		num_layers = config['num_layers']
		max_z = config['max_z']
		k = config['k']
		self.k=k
		self.lr = config['lr']

		self.z_embedding = Embedding(max_z,hidden_channels)
		self.convs = ModuleList()
		#TODO: get it to include features
		initial_channels = hidden_channels
		# initial_channels += train_dataset.num_features
		self.convs.append(GCNConv(initial_channels, hidden_channels))
		for i in range(0, num_layers-1):
			self.convs.append(GCNConv(hidden_channels,hidden_channels))
		self.convs.append(GCNConv(hidden_channels,1))

		#TODO: i don't understand the convolution layers
		conv1d_channels = [16,32]
		total_latent_dim = hidden_channels * num_layers + 1
		conv1d_kws = [total_latent_dim,5]
		self.conv1 = Conv1d(1,conv1d_channels[0],conv1d_kws[0],conv1d_kws[0])
		self.maxpool1d = MaxPool1d(2,2)
		self.conv2 = Conv1d(conv1d_channels[0],conv1d_channels[1],conv1d_kws[1],1)
		dense_dim = int(k/2)
		dense_dim = (dense_dim - conv1d_kws[1]+1) * conv1d_channels[1]
		self.lin1 = Linear(dense_dim,128)
		self.lin2 = Linear(128,1)

	def forward(self,z,edge_index,batch):
		z_emb = self.z_embedding(z)
		if z_emb.ndim >=3:
			z_emb = z_emb.sum(dim=1)
		#TODO: add in features
		xs = [z_emb]

		for conv in self.convs:
			xs += [torch.tanh(conv(xs[-1],edge_index,None))]
		x = torch.cat(xs[1:],dim=-1)
		x = global_sort_pool(x,batch,self.k)
		x = x.unsqueeze(1)
		x = F.relu(self.conv1(x))
		x = self.maxpool1d(x)
		x = F.relu(self.conv2(x))
		x = x.view(x.size(0),-1)

		x = F.relu(self.lin1(x))
		x = F.dropout(x,p=0.5,training=self.training)
		x = self.lin2(x)
		return x
