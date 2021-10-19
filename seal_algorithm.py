import scipy.sparse as ssp
import torch
import numpy as np
from torch_geometric.data import Data
from scipy.sparse.csgraph import shortest_path
import time

seal_config = {'num_hops': 2, 'trick': 'drnl'}

def neighbors(fringe, A):
    fringe = list(fringe)
    res = set(A[fringe].indices)
    return res

def k_hop_subgraph(src, dst, A, node_feats):

    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [int(src), int(dst)] #shouldn't be needed
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    num_hops = seal_config['num_hops']
    t1 = time.time()
    for dist in range(1, num_hops+1):
        if len(fringe) == 0:
            break
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    t2 = time.time()-t1
    if t2 > 0.006:
        print('1',t2)
    t1 = time.time()
    subgraph = A[nodes, :][:, nodes]
    t2 = time.time()-t1
    if t2 > 0.006:
        print('2',t2)

    # Remove target link between the subgraph.
    t1 = time.time()
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0
    t2 = time.time()-t1
    if t2 > 0.006:
        print('3',t2)
    return nodes, subgraph, dists, node_feats[nodes]

def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)

def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More 
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()

def make_sealed_data(node_ids,adj_sparse,dists,node_feats,label):
    rowidxs,colidxs,_ = ssp.find(adj_sparse)
    num_nodes = adj_sparse.shape[0]
    node_ids = torch.LongTensor(node_ids)
    rowidxs,colidxs = torch.LongTensor(rowidxs), torch.LongTensor(colidxs)
    edge_index = torch.stack([rowidxs,colidxs],0)
    label = torch.tensor([label])
    if 'trick' not in seal_config or seal_config['trick'] == 'drnl':
        sealed = drnl_node_labeling(adj_sparse,0,1)
    elif seal_config['trick'] == 'hop':
        sealed = torch.tensor(distances)
    elif seal_config['trick'] == 'zero-one':
        sealed = (torch.tensor(distances)==0).to(torch.long)
    elif seal_config['trick'] == 'distance encoding':
        sealed = de_node_labeling(adj_sparse,0,1)
    data = Data(node_feats,edge_index,y=label,z=sealed,node_ids=node_ids,num_nodes=num_nodes)
    return data

def seal_data(data,src,dst,label):
    all_ones = torch.ones(data.edge_index.size(1))
    adj_sparse = ssp.csr_matrix((all_ones,(data.edge_index[0],data.edge_index[1])),shape=(data.num_nodes,data.num_nodes))
    node_idxs,subgraph,dists,node_feats = k_hop_subgraph(src,dst,adj_sparse,data['x'])
    data = make_sealed_data(node_idxs,subgraph,dists,node_feats,label)
    return data

