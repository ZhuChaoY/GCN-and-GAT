# GCN-GAT
A version of GCN and GAT.

(1) GCN: SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS (https://github.com/tkipf/gcn)   
```
$ python Run_GCN.py --dataset cora --h_dim 16 --dropout 0.5 --l2 5e-4 --l_r 1e-2 --epoches 200 --earlystop 3
```

(2) GAT: GRAPH ATTENTION NETWORKS (https://github.com/PetarV-/GAT)
```
$ python Run_GAT.py --dataset cora --h_dim 8 --n_head [8, 1] --dropout 0.6 --l2 5e-4 --l_r 5e-3 --epoches 400 --earlystop 3
```

