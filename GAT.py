import numpy as np
import networkx as nx
import tensorflow as tf
from GNN import GNN


class GAT(GNN): #2 layers
    """@ GRAPH ATTENTION NETWORKS"""

    def __init__(self, args):
        super().__init__(args)
        print('\n' + '==' * 4 + ' < GAT > && < {} > '.format(self.dataset) + \
              '==' * 4)  
        self.load_data()


    def get_A(self):
        """Get dense adjacency matrix bias with n_hop neiborhood."""
        
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(self.graph)).todense()
        adj.astype(np.float16)        
        A = np.eye(self.n_node, dtype = np.float16)
        for _ in range(self.n_hop):
            A = np.matmul(A, adj + np.eye(self.n_node, dtype = np.float16))
        A[A == 0.0] = -1.0
        A[A > 0.0] = 0.0
        return A
    

    def gnn_layer(self):
        """A layer of GAT structure."""
        
        self.bias = tf.placeholder(tf.float32, [self.n_node, self.n_node])
        self.feed_dict = {self.input: self.X, self.bias: self.A}
        
        with tf.variable_scope('GAT'):
            with tf.variable_scope('layer1'):
                h_out = []
                for i in range(self.n_head_1):
                    with tf.variable_scope('head_{}'.format(i + 1)):
                        h_out.append(self.gat_layer(self.input, self.in_dim,
                                     self.h_dim, tf.nn.elu, True))   
                h_out = tf.concat(h_out, -1)
            with tf.variable_scope('layer2'):    
                output = []
                for i in range(self.n_head_2):
                    with tf.variable_scope('head_{}'.format(i + 1)):
                        output.append(self.gat_layer(h_out, self.h_dim * \
                                     self.n_head_1, self.out_dim, None, False))
                output = tf.add_n(output) / self.n_head_2
        
        return output
    

    def gat_layer(self, X, in_dim, out_dim, act, is_sparse):
        """A layer of GAT."""
        
        K1 = np.sqrt(6.0 / (in_dim + out_dim))
        w0 = tf.get_variable('weight_0', initializer = \
             tf.random_uniform([in_dim, out_dim], -K1, K1))
        
        #(N, in_dim) * (in_dim, out_dim) ==> (N, out_dim)
        if is_sparse:
            e = tf.sparse_tensor_dense_matmul(X, w0)
        else:
            e = tf.matmul(X, w0)

        K2 = np.sqrt(6.0 / (out_dim + 1))
        w1 = tf.get_variable('weight_1', initializer = \
             tf.random_uniform([out_dim, 1], -K2, K2))
        w2 = tf.get_variable('weight_2', initializer = \
             tf.random_uniform([out_dim, 1], -K2, K2))
            
        #(N, out_dim) * (out_dim, 1) ==> (N, 1) (==> (1, N))
        e1 = tf.matmul(e, w1)
        e2 = tf.transpose(tf.matmul(e, w2), [1, 0])

        if not self.att_batch_size:
            out = self.att_layer(e, e1, e2, self.bias)
        else:
            bs = self.att_batch_size
            n_batch = self.n_node // bs
            out = []
            for i in range(n_batch):
                out.append(self.att_layer(e, e1[i * bs: (i + 1) * bs], e2,
                           self.bias[i * bs: (i + 1) * bs]))
            if self.n_node % bs != 0:
                out.append(self.att_layer(e, e1[n_batch * bs: ], e2, 
                           self.bias[n_batch * bs: ]))
            #(B, out_dim) * n_batch ==> (N, out_dim)
            out = tf.concat(out, 0)
        
        if act:
            out = act(out)
        return out
    

    def att_layer(self, e, e1, e2, bias):
        """Attention layer."""
        
        #(B, 1) + (1, N) ==> (B, N) * (N, out_dim) ==> (B, out_dim)
        a = tf.nn.dropout(tf.nn.softmax(tf.nn.leaky_relu(e1 + e2) + \
                                        1e6 * bias), self.keep)
        return tf.matmul(a, tf.nn.dropout(e, self.keep))
    