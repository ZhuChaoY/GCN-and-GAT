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
        self.B = self.get_B()


    def get_A(self):
        """Get dense adjacency matrix."""
        
        A = nx.adjacency_matrix(nx.from_dict_of_lists(self.graph)).todense()
        return A
    

    def get_B(self):
        """Get adjacency matrix bias with n_hop neiborhood."""
        
        B = np.eye(self.n_node)
        for _ in range(self.n_hop):
            B = np.matmul(B, self.A + np.eye(self.n_node))
        for i in range(self.n_node):
            for j in range(self.n_node):
                if B[i, j] > 0.0:
                    B[i, j] = 0.0
                else:
                    B[i, j] = -1e9
        return B
    

    def gnn_layer(self):
        """A layer of GAT structure."""
        
        self.input = tf.sparse_placeholder(tf.float32)
        self.bias = tf.placeholder(tf.float32, [self.n_node, self.n_node])
        self.feed_dict = {self.input: self.X, self.bias: self.B}
        
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
                
        loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        
        return output, loss
    

    def gat_layer(self, _input, in_dim, out_dim, act, is_sparse):
        """A layer of GAT."""
        
        K1 = np.sqrt(6.0 / (in_dim + out_dim))
        w0 = tf.get_variable('weight_0', initializer = \
             tf.random_uniform([in_dim, out_dim], -K1, K1))
        
        #(N, in_dim) * (in_dim, out_dim) ==> (N, out_dim)
        if is_sparse:
            e = tf.sparse_tensor_dense_matmul(_input, w0)
        else:
            e = tf.matmul(_input, w0)
        
        K2 = np.sqrt(6.0 / (out_dim + 1))
        w1 = tf.get_variable('weight_1', initializer = \
             tf.random_uniform([out_dim, 1], -K2, K2))
        w2 = tf.get_variable('weight_2', initializer = \
             tf.random_uniform([out_dim, 1], -K2, K2))
            
        #(N, out_dim) * (out_dim, 1) ==> (N, 1) + (1, N) ==> (N, N)
        e1 = tf.matmul(e, w1)
        e2 = tf.matmul(e, w2)
        logits = e1 + tf.transpose(e2, [1, 0])
        
        #(N, N) * (N, out_dim) ==> (N, out_dim)
        a = tf.nn.dropout(tf.nn.softmax(tf.nn.leaky_relu(logits) + \
                                        self.bias), self.keep)
        out = tf.matmul(a, tf.nn.dropout(e, self.keep))
        
        if act:
            out = act(out)
        return tf.squeeze(out)