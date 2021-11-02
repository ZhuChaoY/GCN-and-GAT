import os
import argparse
from GCN import GCN


parser = argparse.ArgumentParser(description = 'GCN')
parser.add_argument('--dataset', type = str, default = 'cora',
                    help = 'dataset name') #'cora', 'citeseer', 'pubmed'
parser.add_argument('--h_dim', type = int, default = 16,
                    help = 'hidden dim')
parser.add_argument('--dropout', type = float, default = 0.5, 
                    help = 'dropout rate')
parser.add_argument('--l2', type = float, default = 5e-4,
                    help = 'hidden dim')
parser.add_argument('--l_r', type = float, default = 1e-2, 
                    help = 'learning rate')
parser.add_argument('--epoches', type = int, default = 200,
                    help = 'training epoches')
parser.add_argument('--earlystop', type = int, default = 3,
                    help = 'earlystop steps')

args = parser.parse_args()
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

model = GCN(args)
model.run(10)
    
    
    