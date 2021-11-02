import os
import argparse
from GAT import GAT


parser = argparse.ArgumentParser(description = 'GAT')
parser.add_argument('--dataset', type = str, default = 'cora',
                    help = 'dataset name') #'cora', 'citeseer', 'pubmed'
parser.add_argument('--h_dim', type = int, default = 8,
                    help = 'hidden dim')
parser.add_argument('--n_head', type = list, default = [8, 1],
                    help = 'number of head')
parser.add_argument('--dropout', type = float, default = 0.6, 
                    help = 'dropout rate')
parser.add_argument('--l2', type = float, default = 5e-4,
                    help = 'hidden dim')
parser.add_argument('--l_r', type = float, default = 5e-3, 
                    help = 'learning rate')
parser.add_argument('--epoches', type = int, default = 400,
                    help = 'training epoches')
parser.add_argument('--earlystop', type = int, default = 3,
                    help = 'earlystop steps')

args = parser.parse_args()
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

model = GAT(args)
model.run(10)
        