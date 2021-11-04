import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from GNN import GNN


class GAT(GNN): #2 layers
    """@ GRAPH ATTENTION NETWORKS"""

    def __init__(self, args):
        super().__init__(args)
        print('\n' + '==' * 4 + ' < GAT > && < {} > '.format(self.dataset) + \
              '==' * 4)  
        self.load_data()
        self.B = self.get_B(nhood = 1)

    
    def get_X(self, test_reorder, test_range):
        """Get feature matrix, row-normalize."""
        
        X = sp.vstack((self.allx, self.tx)).tolil()
        X[test_reorder, :] = X[test_range, :]
        rowsum = np.array(X.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        X = r_mat_inv.dot(X)
        return X.todense()


    def get_A(self):
        """Get dense adjacency matrix."""
        
        A = nx.adjacency_matrix(nx.from_dict_of_lists(self.graph)).todense()
        return A
    
    
    def get_B(self, nhood):
        """Get adjacency matrix bias with nhood neiborhood."""
        
        B = np.eye(self.n_node)
        for _ in range(nhood):
            B = np.matmul(B, (self.A + np.eye(self.n_node)))
        for i in range(self.n_node):
            for j in range(self.n_node):
                if B[i, j] > 0.0:
                    B[i, j] = 0.0
                else:
                    B[i, j] = -1e9
        return B


    def gnn_layer(self):
        """A layer of GAT structure."""
        
        self.bias = tf.placeholder(tf.float32, [self.n_node, self.n_node])
        self.input = tf.placeholder(tf.float32, [self.n_node, self.in_dim])
        self.feed_dict = {self.input: self.X, self.bias: self.B}
        
        with tf.variable_scope('GAT'):
            with tf.variable_scope('layer1'):
                h_out = tf.concat([self.gat_layer(self.input, self.h_dim,
                        tf.nn.elu) for _ in range(self.n_head[0])], axis = -1)
            with tf.variable_scope('layer2'):            
                output = tf.add_n([self.gat_layer(h_out, self.out_dim, None) \
                         for _ in range(self.n_head[1])]) / self.n_head[1]
                
        loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if
                         'kernel' in v.name])
        
        return output, loss
    

    def gat_layer(self, _input, out_dim, act):
        """A layer of GAT"""
        
        _input = tf.expand_dims(_input, 0)
        fts = tf.layers.conv1d(_input, out_dim, 1, use_bias = False)
        f_1 = tf.layers.conv1d(fts, 1, 1)
        f_2 = tf.layers.conv1d(fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.dropout(tf.nn.softmax(tf.nn.leaky_relu(logits) + \
                                            self.bias), self.keep)
        out = tf.matmul(coefs, tf.nn.dropout(fts, self.keep))
        if act:
            out = act(out)
        return tf.squeeze(out)