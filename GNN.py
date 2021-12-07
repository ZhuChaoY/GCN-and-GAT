import re
import time
import pickle
import numpy as np
import tensorflow as tf
import scipy.sparse as sp


class GNN(): #2 layers
    """A class of graph neural network."""

    def __init__(self, args):
        self.args = dict(args._get_kwargs())
        for key, value in self.args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
        
    
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
            with open('dataset/ind.{}.{}'.format(self.dataset, key), 'rb') as f:
                _ = pickle.load(f, encoding = 'latin1')
                exec('self.{} = _'.format(key))
        self.n_train = self.x.shape[0]
        self.n_dev = 500
        self.in_dim = self.x.shape[1]
        self.out_dim = self.y.shape[1]
        self.n_node = len(self.graph)
        self.n_edge = sum([len(x) for x in self.graph.values()])
        print('    #Node  : {}'.format(self.n_node))
        print('    #Edge  : {}'.format(self.n_edge))

        test_reorder = []
        for line in open('dataset/ind.{}.test.index'.format(self.dataset)):
            test_reorder.append(int(line.strip()))
        train_range = range(self.n_train)
        dev_range = range(self.n_train, self.n_train + self.n_dev)
        test_range = np.sort(test_reorder)
        print('    #Train : {}'.format(self.n_train))
        print('    #Dev   : {}'.format(self.n_dev))
        print('    #Test  : {}'.format(len(test_range)))
        
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
        self.train_m = self.get_m(train_range)
        self.dev_m = self.get_m(dev_range)
        self.test_m = self.get_m(test_range)
        self.Y = np.vstack((self.ally, self.ty))
        self.Y[test_reorder, :] = self.Y[test_range, :]
        self.train_y = self.get_y(train_range)
        self.dev_y = self.get_y(dev_range)
        self.test_y = self.get_y(test_range)
    
    
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
    
    
    def get_m(self, key_range):
        """Get mask indexes, normalized to mean 1."""
        
        m = np.array([x in key_range for x in range(self.n_node)])
        return m / np.mean(m)
    
    
    def get_y(self, key_range):
        """Get masked label."""
        
        y = self.Y.copy()
        y[list(set(range(self.n_node)) - set(key_range)), :] = 0.0
        return y
        
    
    def common_structure(self):        
        """Common structure of GCN and GAT."""
        
        tf.reset_default_graph()
        self.input = tf.sparse_placeholder(tf.float32)
        self.label = tf.placeholder(tf.float32, [self.n_node, self.out_dim])
        self.mask = tf.placeholder(tf.float32, [self.n_node])
        self.keep = tf.placeholder(tf.float32)
        
        output = self.gnn_layer()                            

        loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
                    logits = output, labels = self.label) * self.mask) + \
                    self.l2 * loss
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,
                        1), tf.argmax(self.label, 1)), tf.float32) * self.mask)
        self.train_op = tf.train.AdamOptimizer(self.l_r).minimize(self.loss)
        
                
    def _train(self, sess):
        """Training Process."""
        
        eps = self.epoches
        print('              Train          Dev')
        print('    EPOCH  LOSS   ACC    LOSS   ACC   time   TIME')
        
        temp_kpi, KPI = [], []
        t0 = t1 = time.time()
        for ep in range(eps):
            feed_dict = {self.label: self.train_y, self.mask: self.train_m,
                         self.keep: 1.0 - self.dropout}
            feed_dict.update(self.feed_dict)
            _, train_loss, train_acc = \
                sess.run([self.train_op, self.loss, self.accuracy], feed_dict)
            dev_loss, dev_acc = self._evaluate(sess)
            if (ep + 1) % 20 == 0:
                _t = time.time()
                print('    {:^5} {:^6.4f} {:^5.3f}  {:^6.4f} {:^5.3f}' \
                      ' {:^6.2f} {:^6.2f}'.format(ep + 1, train_loss,
                      train_acc, dev_loss, dev_acc, _t - t1, _t - t0))
                t1 = _t
            
            kpi = round(dev_loss, 5)
            if ep == 0 or kpi < KPI[-1]:
                if len(temp_kpi) > 0:
                    KPI.extend(temp_kpi)
                    temp_kpi = []
                KPI.append(kpi)
            else:
                if len(temp_kpi) == self.earlystop:
                    break
                else:
                    temp_kpi.append(kpi)
                    
        best_ep = len(KPI)
        if best_ep != eps:
            print('\n    Early stop at epoch of {} !'.format(best_ep))
            
    
    def _evaluate(self, sess):
        """Validation Process."""
        
        feed_dict = {self.label: self.dev_y, self.mask: self.dev_m, 
                     self.keep: 1.0}
        feed_dict.update(self.feed_dict)
        return sess.run([self.loss, self.accuracy], feed_dict)
        

    def _test(self, sess):
        """Test Process."""
        
        feed_dict = {self.label: self.test_y, self.mask: self.test_m,
                     self.keep: 1.0}
        feed_dict.update(self.feed_dict)
        test_loss, test_acc = sess.run([self.loss, self.accuracy], feed_dict)
        print('\n    Test : [ Loss: {:.4f} ; Acc : {:.3f} ]\n'. \
              format(test_loss, test_acc))
        return test_acc


    def run(self, N): 
        """Repeat N times run and calculate mean accuracy."""
        
        Acc = []
        for i in range(N):            
            config = tf.ConfigProto() 
            config.gpu_options.allow_growth = True
            self.common_structure()
            if i == 0:       
                print('\n    *Hidden Dim     : {}'.format(self.h_dim))
                if 'n_head_1' in self.args:
                    print('    *Head of layer1 : {}'.format(self.n_head_1))
                    print('    *Head of layer2 : {}'.format(self.n_head_2))
                    print('    *Number of Hop  : {}'.format(self.n_hop))
                print('    *Drop Out Rate  : {}'.format(self.dropout))
                print('    *L2 Rate        : {}'.format(self.l2))
                print('    *Learning Rate  : {}'.format(self.l_r))
                print('    *Epoches        : {}'.format(self.epoches))
                print('    *Earlystop Step : {}\n'.format(self.earlystop))
                    
                shape = {re.match('^(.*):\\d+$', v.name).group(1):
                         v.shape.as_list() for v in tf.trainable_variables()}
                tvs = [re.match('^(.*):\\d+$', v.name).group(1)
                        for v in tf.trainable_variables()]                      
                for v in tvs:
                    print('    -{} : {}'.format(v, shape[v]))
                
            with tf.Session(config = config) as sess:
                tf.global_variables_initializer().run()  
                print('\n>>  {} | {} Training Process.'.format(i + 1, N))
                self._train(sess)
                Acc.append(self._test(sess))
        
        print('\n>>  Result of {} Runs: {:.3f} ({:.3f})'.format(N,
              np.mean(Acc), np.std(Acc)))
