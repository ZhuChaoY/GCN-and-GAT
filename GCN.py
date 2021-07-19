import os
import time
import pickle
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow.compat.v1 as tf


class GCN(): #2 LAYER
    def __init__(self, args):
        self.args = args
        for key, value in self.args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
        
        self.load_data()
        self.construct_model()
        
    
    def load_data(self): 
        """
        Loads input data from gcn/data directory
        x => the feature vectors of the training instances;
        y => the one-hot labels of the labeled training instances;
        tx => the feature vectors of the test instances;
        ty => the one-hot labels of the test instances;
        allx => the feature vectors of both labeled and unlabeled training 
                instances (superset of x);
        ally => the labels for instances in allx;
        graph => format of {index: [index_of_neighbor_nodes]};
        test.index => the indices of test instances in graph for the inductive
                      setting as list object.
        All objects above must be saved using python pickle module.
        """

        for key in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']:
            with open('data/ind.{}.{}'.format(self.dataset, key), 'rb') as f:
                _ = pickle.load(f, encoding = 'latin1')
                exec('self.{} = _'.format(key))
        self.n_train = self.x.shape[0]
        self.n_val = 500
        self.in_dim = self.x.shape[1]
        self.out_dim = self.y.shape[1]
        self.n_node = len(self.graph)
        
        test_reorder = []
        for line in open('data/ind.{}.test.index'.format(self.dataset)):
            test_reorder.append(int(line.strip()))
        train_range = range(self.n_train)
        val_range = range(self.n_train, self.n_train + self.n_val)
        test_range = np.sort(test_reorder)
        
        if self.dataset == 'citeseer':
            n_test = max(test_reorder) + 1 - min(test_reorder)
            tx_extended = sp.lil_matrix((n_test, self.in_dim))
            tx_extended[test_range - min(test_range), :] = self.tx
            self.tx = tx_extended
            ty_extended = np.zeros((n_test, self.out_dim))
            ty_extended[test_range - min(test_range), :] = self.ty
            self.ty = ty_extended
    
        self.X = self.get_X(test_reorder, test_range)
        self.A = self.get_A()
        self.m_train = self.get_m(train_range)
        self.m_val = self.get_m(val_range)
        self.m_test = self.get_m(test_range)
        self.Y = np.vstack((self.ally, self.ty))
        self.Y[test_reorder, :] = self.Y[test_range, :]
        self.y_train = self.get_y(train_range)
        self.y_val = self.get_y(val_range)
        self.y_test = self.get_y(test_range)
        
        
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
        
    
    def get_m(self, key_range):
        """Get mask indexes, normalized to sum 1."""
        m = np.array([x in key_range for x in range(self.n_node)])
        return m / np.mean(m)
    
    
    def get_y(self, key_range):
        """Get masked label."""
        y = self.Y.copy()
        y[list(set(range(self.n_node)) - set(key_range)), :] = 0.0
        return y
    
    
    def construct_model(self):
        """Construct 2 layer GCN."""
        
        tf.reset_default_graph()
        self.support = tf.sparse_placeholder(tf.float32)
        self.input = tf.sparse_placeholder(tf.float32, shape = \
                     tf.constant([self.n_node, self.in_dim], tf.int64))
        self.label = tf.placeholder(tf.float32, [self.n_node, self.out_dim])
        self.mask = tf.placeholder(tf.float32, [self.n_node])
        self.keep = tf.placeholder(tf.float32)
        self.n_nonzero = tf.placeholder(tf.int32) 
        
        with tf.variable_scope('GCN'):
            with tf.variable_scope('layer1'):
                K1 = np.sqrt(6.0 / (self.in_dim + self.hidden))
                self.w1 = tf.get_variable('weight1', initializer = \
                        tf.random_uniform([self.in_dim, self.hidden], -K1, K1))
                hidden_out = self.gcn_layer(self.input, self.w1,
                                            tf.nn.relu, True)
            with tf.variable_scope('layer2'):
                K2 = np.sqrt(6.0 / (self.hidden + self.out_dim))
                self.w2 = tf.get_variable('weight2', initializer = \
                       tf.random_uniform([self.hidden, self.out_dim], -K2, K2))
                self.output = self.gcn_layer(hidden_out, self.w2,
                                             lambda x: x, False)
                            
        self.predict = tf.nn.softmax(self.output)
        self.loss = self.l2 * (tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2))
        self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
                      logits = self.output, labels = self.label) * self.mask)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output,
                        1), tf.argmax(self.label, 1)), tf.float32) * self.mask)
        self.train_op = tf.train.AdamOptimizer(self.l_r).minimize(self.loss)

    
    def gcn_layer(self, _input, w, act, is_sparse):
        """A layer of GCN."""
        if is_sparse:
            dropout_mask = tf.cast(tf.floor(self.keep + \
                           tf.random_uniform(self.n_nonzero)), dtype = tf.bool)
            _input = tf.sparse_retain(_input, dropout_mask) * (1.0 / self.keep)
        else:
            _input = tf.nn.dropout(_input, self.keep)
        if is_sparse:
            midden = tf.sparse_tensor_dense_matmul(_input, w)
        else:
            midden = tf.matmul(_input, w)
        return act(tf.sparse_tensor_dense_matmul(self.support, midden))


    def _train(self, sess):
        t0 = t1 = time.time()
        print('              Train          Val')
        print('    EPOCH  LOSS   ACC    LOSS   ACC   time   TIME')
        Loss_val = []
        for epoch in range(self.epochs):
            print('    {:^5}'.format(epoch + 1), end = '')

            feed_dict = {self.input: self.X, self.support: self.A,
                         self.label: self.y_train, self.mask: self.m_train,
                         self.n_nonzero: self.X[1].shape,
                         self.keep: 1.0 - self.dropout}
            _, loss_train, acc_train = \
                sess.run([self.train_op, self.loss, self.accuracy], feed_dict)
            loss_val, acc_val = self._evaluate(sess)
            Loss_val.append(loss_val)
            _t = time.time()
            print(' {:^6.4f} {:^5.3f}  {:^6.4f} {:^5.3f} {:^6.2f} {:^6.2f}'. \
                  format(loss_train, acc_train, loss_val, acc_val,
                         _t - t1, _t - t0))
            t1 = _t
            
            if epoch > self.early_stop and \
               loss_val > np.mean(Loss_val[-(self.early_stop + 1): -1]):
                   print('\n>>  Early stopping...')
                   break
        
        loss_test, acc_test = self._test(sess)
        print('\n>>  Test Result.')
        print('    Loss: {:.4f}\n    Acc : {:.3f}'.format(loss_test, acc_test))
               

    def _evaluate(self, sess):
        feed_dict = {self.input: self.X, self.support: self.A,
                     self.label: self.y_val, self.mask: self.m_val, 
                     self.n_nonzero: self.X[1].shape, self.keep: 1.0}
        return sess.run([self.loss, self.accuracy], feed_dict)
        

    def _test(self, sess):
        feed_dict = {self.input: self.X, self.support: self.A,
                     self.label: self.y_test, self.mask: self.m_test, 
                     self.n_nonzero: self.X[1].shape, self.keep: 1.0}
        return sess.run([self.loss, self.accuracy], feed_dict)
        
    

args = {'dataset' : ['cora', 'citeseer', 'pubmed'][0],
        'l_r' : 0.01,
        'epochs' : 200,
        'hidden' : 16,
        'dropout' : 0.5,
        'l2' : 5e-4,
        'early_stop' : 10}

     
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True

model = GCN(args)
with tf.Session(config = config) as sess:
    tf.global_variables_initializer().run()   
    model._train(sess)
    
