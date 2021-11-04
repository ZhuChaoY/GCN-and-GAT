import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from GNN import GNN


class GCN(GNN): #2 layers
    """@ SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS"""

    def __init__(self, args):
        super().__init__(args)
        print('\n' + '==' * 4 + ' < GCN > && < {} > '.format(self.dataset) + \
              '==' * 4)  
        self.load_data()
    
        
    def get_X(self, test_reorder, test_range):
        """
        Get feature matrix, row-normalize and convert to tuple representation.
        """
        
        X = sp.vstack((self.allx, self.tx)).tolil()
        X[test_reorder, :] = X[test_range, :]
        rowsum = np.array(X.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        X = r_mat_inv.dot(X).tocoo()
        return np.vstack((X.row, X.col)).transpose(), X.data, X.shape
    
    
    def get_A(self):
        """
        Get adjacency matrix, preprocessing for simple GCN model and convert
        to tuple representation.
        """
        
        A = nx.adjacency_matrix(nx.from_dict_of_lists(self.graph))
        A = sp.coo_matrix(A + sp.eye(self.n_node))
        rowsum = np.array(A.sum(1))
        inv_sqrt = np.power(rowsum, -0.5).flatten()
        inv_sqrt[np.isinf(inv_sqrt)] = 0.0
        mat_inv_sqrt = sp.diags(inv_sqrt) 
        A = A.dot(mat_inv_sqrt).transpose().dot(mat_inv_sqrt).tocoo()
        return np.vstack((A.row, A.col)).transpose(), A.data, A.shape
    
            
    def gnn_layer(self):
        """A layer of GCN structure."""
        
        self.support = tf.sparse_placeholder(tf.float32)
        self.input = tf.sparse_placeholder(tf.float32, shape = \
                                           [self.n_node, self.in_dim])
        self.feed_dict = {self.input: self.X, self.support: self.A}
            
        with tf.variable_scope('GCN'):
            with tf.variable_scope('layer1'):
                h_out = self.gcn_layer(self.input, self.in_dim, self.h_dim,
                                       tf.nn.relu, True)
            with tf.variable_scope('layer2'):
                output = self.gcn_layer(h_out, self.h_dim, self.out_dim,
                                        None, False)
                            
        loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            
        return output, loss

    
    def gcn_layer(self, _input, in_dim, out_dim, act, is_sparse):
        """A layer of GCN."""
        
        K = np.sqrt(6.0 / (in_dim + out_dim))
        w = tf.get_variable('weight', initializer = \
            tf.random_uniform([in_dim, out_dim], -K, K))
        
        if is_sparse:
            tmp = tf.sparse_tensor_dense_matmul(_input, w)
        else:
            tmp = tf.matmul(_input, w)
        out = tf.sparse_tensor_dense_matmul(self.support, tmp)
        
        if act:
            out = act(out)
            
        return tf.nn.dropout(out, self.keep)
            