# GCN and GAT
An implementation of GCN and GAT by tensorflow.

## Reference
(1) GCN: [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf) (https://github.com/tkipf/gcn)   
(2) GAT: [GRAPH ATTENTION NETWORKS](https://arxiv.org/pdf/1710.10903v1.pdf) (https://github.com/PetarV-/GAT)   

## Results (10 runs)                    
|        | **citeseer**  |   **cora**    |   **pubmed**  |  
|   --   |      --       |      --       |      --       |  
|**GCN** | 0.712 (0.005) | 0.811 (0.008) | 0.787 (0.002) |   
|**GAT** | 0.714 (0.003) | 0.815 (0.006) | out of memory |   

```
python Run_GCN.py --dataset citeseer --h_dim 16 --dropout 0.5 --l2 5e-4 --l_r 1e-2
```
```
python Run_GCN.py --dataset cora --h_dim 16 --dropout 0.5 --l2 5e-4 --l_r 1e-2
```
```
python Run_GCN.py --dataset pubmed --h_dim 16 --dropout 0.5 --l2 5e-4 --l_r 1e-2
```
```
python Run_GAT.py --dataset citeseer --h_dim 8 --n_head_1 8 --n_head_2 1 --dropout 0.6 --l2 5e-4 --l_r 5e-3
```
```
python Run_GAT.py --dataset cora --h_dim 8 --n_head_1 8 --n_head_2 1 --dropout 0.6 --l2 5e-4 --l_r 5e-3
```
```
python Run_GAT.py --dataset pubmed --h_dim 8 --n_head_1 8 --n_head_2 1 --dropout 0.6 --l2 5e-4 --l_r 1e-2
```
