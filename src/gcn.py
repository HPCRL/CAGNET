import os.path as osp
import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, PPI, Reddit
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv # noqa
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, PPI
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops, to_dense_adj, dense_to_sparse, to_scipy_sparse_matrix

import time

def symmetric(adj_matrix):
    # print(adj_matrix)
    # not sure whether the following is needed
    adj_matrix = adj_matrix.to(torch.device("cpu"))

    adj_matrix, _ = remove_self_loops(adj_matrix)

    # Make adj_matrix symmetrical
    idx = torch.LongTensor([1,0])
    adj_matrix_transpose = adj_matrix.index_select(0,idx)
    # print(adj_matrix_transpose)

    adj_matrix = torch.cat([adj_matrix,adj_matrix_transpose],1)

    adj_matrix, _ = add_remaining_self_loops(adj_matrix)

    adj_matrix.to(device)
    return adj_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--graphname', type=str, default='ogbn-arxiv')
parser.add_argument('--midlayer', type=int, default=128)
args = parser.parse_args()

graphname = args.graphname
midlayer = args.midlayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if graphname == "Cora":
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', graphname)
    dataset = Planetoid(path, graphname, transform=T.NormalizeFeatures())
    data = dataset[0]
    data = data.to(device)
    data.x.requires_grad = True
    inputs = data.x.to(device)
    inputs.requires_grad = True
    data.y = data.y.to(device)
    edge_index = data.edge_index
    num_features = dataset.num_features
    num_classes = dataset.num_classes
elif graphname == "Reddit":
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', graphname)
    path = '/scratch/general/nfs1/u1320844/dataset/reddit_pyg/'
    dataset = Reddit(path)
    data = dataset[0]
    data = data.to(device)
    data.x.requires_grad = True
    inputs = data.x.to(device)
    inputs.requires_grad = True
    data.y = data.y.to(device)
    edge_index = data.edge_index
    num_features = dataset.num_features
    num_classes = dataset.num_classes
elif graphname == 'Amazon':
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', graphname)
    # edge_index = torch.load(path + "/processed/amazon_graph.pt")
    # edge_index = torch.load("/gpfs/alpine/bif115/scratch/alokt/Amazon/processed/amazon_graph_jsongz.pt")
    # edge_index = edge_index.t_()
    print(f"Loading coo...", flush=True)
    edge_index = torch.load("../data/Amazon/processed/data.pt")
    print(f"Done loading coo", flush=True)
    # n = 9430088
    n = 14249639
    num_features = 300
    num_classes = 24
    # mid_layer = 24
    inputs = torch.rand(n, num_features)
    data = Data()
    data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
    data.train_mask = torch.ones(n).long()
    # edge_index = edge_index.to(device)
    print(f"edge_index.size: {edge_index.size()}", flush=True)
    print(f"edge_index: {edge_index}", flush=True)
    data = data.to(device)
    # inputs = inputs.to(device)
    inputs.requires_grad = True
    data.y = data.y.to(device)
elif graphname == 'subgraph3':
    # path = "/gpfs/alpine/bif115/scratch/alokt/HipMCL/"
    # print(f"Loading coo...", flush=True)
    # edge_index = torch.load(path + "/processed/subgraph3_graph.pt")
    # print(f"Done loading coo", flush=True)
    print(f"Loading coo...", flush=True)
    edge_index = torch.load("../data/subgraph3/processed/data.pt")
    print(f"Done loading coo", flush=True)
    n = 8745542
    num_features = 128
    # mid_layer = 512
    # mid_layer = 64
    num_classes = 256
    inputs = torch.rand(n, num_features)
    data = Data()
    data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
    data.train_mask = torch.ones(n).long()
    print(f"edge_index.size: {edge_index.size()}", flush=True)
    data = data.to(device)
    inputs.requires_grad = True
    data.y = data.y.to(device)
elif 'ogb' in graphname:
    path = '/scratch/general/nfs1/u1320844/dataset'
    # path = '../data/'
    dataset = PygNodePropPredDataset(graphname, path,transform=T.NormalizeFeatures())
    #evaluator = Evaluator(name=graphname)
    if 'mag' in graphname:
        rel_data = dataset[0]
        # only train with paper <-> paper relations.
        data = Data(
            x=rel_data.x_dict['paper'],
            edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
            y=rel_data.y_dict['paper'])
        data = T.NormalizeFeatures()(data)
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']['paper']
        val_idx = split_idx['valid']['paper']
        test_idx = split_idx['test']['paper']
    else:
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        data = data.to(device)
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']

    data.x.requires_grad = True
    inputs = data.x.to(device)
    #inputs.requires_grad = True
    data.y = data.y.squeeze().to(device)
    edge_index = data.edge_index
    if graphname == 'ogbn-arxiv':
        edge_index = symmetric(edge_index)
    num_features = dataset.num_features if not 'mag' in graphname else 128
    num_classes = dataset.num_classes
    num_nodes = len(data.x)
    train_mask = torch.zeros(num_nodes)
    train_mask[train_idx] = 1
    val_mask = torch.zeros(num_nodes)
    val_mask[val_idx] = 1
    test_mask = torch.zeros(num_nodes)
    test_mask[test_idx] = 1
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

elif graphname == 'com-orkut':
    edge_index = torch.load('/scratch/general/nfs1/u1320844/dataset/com_orkut/com-orkut.pt')
    print(f"Done loading coo", flush=True)
    n = 3072441
    n = 3072627
    num_features = 128
    num_classes = 100
    inputs = torch.rand(n, num_features)
    data = Data()
    data.y = torch.rand(n).uniform_(0, num_classes-1).long()
    data.train_mask = torch.ones(n).long()
    inputs.requires_grad = True
    data.y = data.y.to(device)
elif graphname == 'web-google':
    edge_index = torch.load('/scratch/general/nfs1/u1320844/dataset/web_google/web-Google.pt')
    print(f"Done loading coo", flush=True)
    n = 916428
    num_features = 256
    num_classes = 100
    inputs = torch.rand(n, num_features)
    data = Data()
    data.y = torch.rand(n).uniform_(0, num_classes-1).long()
    data.train_mask = torch.ones(n).long()
    inputs.requires_grad = True
    data.y = data.y.to(device)



# dataset = Planetoid(path, graphname, transform=T.NormalizeFeatures())
# dataset = PPI(path, 'train', T.NormalizeFeatures())
# dataset = Reddit(path, T.NormalizeFeatures())
# dataset = Yelp(path, T.NormalizeFeatures())
# data = dataset[0]

seed = 0

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, midlayer, cached=True, normalize=False, bias=False)
        self.conv2 = GCNConv(midlayer, dataset.num_classes, cached=True, normalize=False, bias=False)

        self.conv1.node_dim = 0
        self.conv2.node_dim = 0

        with torch.no_grad():
            self.conv1.weight = Parameter(weight1)
            self.conv2.weight = Parameter(weight2)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        # x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return x



torch.manual_seed(seed)
weight1 = torch.rand(dataset.num_features, midlayer)
weight1 = weight1.to(device)

weight2 = torch.rand(midlayer, dataset.num_classes)
weight2 = weight2.to(device)

data.y = data.y.type(torch.LongTensor)
model, data = Net().to(device), data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    outputs = model()
    
    # Note: bool type removes warnings, unsure of perf penalty
    F.nll_loss(outputs[data.train_mask.bool()], data.y[data.train_mask.bool()]).backward()
    # F.nll_loss(outputs, torch.max(data.y, 1)[1]).backward()

    for W in model.parameters():
        if W.grad is not None:
            print(W.grad)

    optimizer.step()
    return outputs

def test(outputs):
    model.eval()
    logits, accs = outputs, []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        mask = mask.bool()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def main(): 
    best_val_acc = test_acc = 0
    outputs = None

    tstart = time.time()

    # for epoch in range(1, 101):
    for epoch in range(1):
        outputs = train()
        train_acc, val_acc, tmp_test_acc = test(outputs)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    tstop = time.time()
    print("Time: " + str(tstop - tstart))

    return outputs

if __name__=='__main__':
    print(main())
