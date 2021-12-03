import os
import argparse
from GAT import GAT


parser = argparse.ArgumentParser(description = 'GAT')
parser.add_argument('--dataset', type = str, default = 'citeseer',
                    help = 'dataset name') #'citeseer', 'cora', 'pubmed'
parser.add_argument('--h_dim', type = int, default = 8,
                    help = 'hidden dim')
parser.add_argument('--n_head_1', type = int, default = 8,
                    help = 'number of head of layer1')
parser.add_argument('--n_head_2', type = int, default = 1,
                    help = 'number of head of layer2')
parser.add_argument('--n_hop', type = int, default = 1,
                    help = 'number of hop')
parser.add_argument('--dropout', type = float, default = 0.6, 
                    help = 'dropout rate')
parser.add_argument('--l2', type = float, default = 5e-4,
                    help = 'l2 penalty coefficient')
parser.add_argument('--l_r', type = float, default = 5e-3, 
                    help = 'learning rate')
parser.add_argument('--epoches', type = int, default = 200,
                    help = 'training epoches')
parser.add_argument('--earlystop', type = int, default = 3,
                    help = 'earlystop steps')

args = parser.parse_args()
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

model = GAT(args)
model.run(10)