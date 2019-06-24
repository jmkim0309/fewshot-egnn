# fewshot-egnn

### Introduction

The current project page provides pytorch code that implements the following CVPR2019 paper:   
**Title:**      "Edge-labeling Graph Neural Network for Few-shot Learning"    
**Authors:**     Jongmin Kim, Taesup Kim, Sungwoong Kim, Chang D.Yoo

**Institution:** KAIST, KaKaoBrain     
**Code:**        https://github.com/khy0809/fewshot-egnn  
**Arxiv:**       https://arxiv.org/abs/1905.01436

**Abstract:**
In this paper, we propose a novel edge-labeling graph
neural network (EGNN), which adapts a deep neural network
on the edge-labeling graph, for few-shot learning.
The previous graph neural network (GNN) approaches in
few-shot learning have been based on the node-labeling
framework, which implicitly models the intra-cluster similarity
and the inter-cluster dissimilarity. In contrast, the
proposed EGNN learns to predict the edge-labels rather
than the node-labels on the graph that enables the evolution
of an explicit clustering by iteratively updating the edgelabels
with direct exploitation of both intra-cluster similarity
and the inter-cluster dissimilarity. It is also well suited
for performing on various numbers of classes without retraining,
and can be easily extended to perform a transductive
inference. The parameters of the EGNN are learned
by episodic training with an edge-labeling loss to obtain a
well-generalizable model for unseen low-data problem. On
both of the supervised and semi-supervised few-shot image
classification tasks with two benchmark datasets, the proposed
EGNN significantly improves the performances over
the existing GNNs.

### Citation
If you find this code useful you can cite us using the following bibTex:
```
@article{kim2019egnn,
  title={Edge-labeling Graph Neural Network for Few-shot Learning},
  author={Jongmin Kim, Taesup Kim, Sungwoong Kim, Chang D. Yoo},
  journal={arXiv preprint arXiv:1905.01436},
  year={2019}
}
```


### Platform
This code was developed and tested with pytorch version 1.0.1

### Setting

You can download miniImagenet dataset from [here](https://drive.google.com/open?id=15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w).

Download 'mini_imagenet_train/val/test.pickle', and put them in the path 
'tt.arg.dataset_root/mini-imagenet/compacted_dataset/'

In ```train.py```, replace the dataset root directory with your own:
tt.arg.dataset_root = '/data/private/dataset'



### Training

```
# ************************** miniImagenet, 5way 1shot *****************************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 1 --transductive False
$ python3 train.py --dataset mini --num_ways 5 --num_shots 1 --transductive True

# ************************** miniImagenet, 5way 5shot *****************************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 5 --transductive False
$ python3 train.py --dataset mini --num_ways 5 --num_shots 5 --transductive True

# ************************** miniImagenet, 10way 5shot *****************************
$ python3 train.py --dataset mini --num_ways 10 --num_shots 5 --meta_batch_size 20 --transductive True

# ************************** tieredImagenet, 5way 5shot *****************************
$ python3 train.py --dataset tiered --num_ways 5 --num_shots 5 --transductive False
$ python3 train.py --dataset tiered --num_ways 5 --num_shots 5 --transductive True

# **************** miniImagenet, 5way 5shot, 20% labeled (semi) *********************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 5 --num_unlabeled 4 --transductive False
$ python3 train.py --dataset mini --num_ways 5 --num_shots 5 --num_unlabeled 4 --transductive True

```

### Evaluation
The trained models are saved in the path './asset/checkpoints/', with the name of 'D-{dataset}-N-{ways}-K-{shots}-U-{num_unlabeld}-L-{num_layers}-B-{batch size}-T-{transductive}'.
So, for example, if you want to test the trained model of 'miniImagenet, 5way 1shot, transductive' setting, you can give --test_model argument as follow:
```
$ python3 eval.py --test_model D-mini_N-5_K-1_U-0_L-3_B-40_T-True
```


## Result
Here are some experimental results presented in the paper. You should be able to reproduce all the results by using the trained models which can be downloaded from [here](https://drive.google.com/open?id=15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w).
#### miniImageNet, non-transductive

| Model                    |  5-way 5-shot acc (%)| 
|--------------------------|  ------------------: | 
| Matching Networks [1]    |         55.30        | 
| Reptile [2]              |         62.74        | 
| Prototypical Net [3]     |         65.77        | 
| GNN [4]                  |         66.41        | 
| **(ours)** EGNN          |         **66.85**        | 

#### miniImageNet, transductive

| Model                    |  5-way 5-shot acc (%)| 
|--------------------------|  ------------------: | 
| MAML [5]                 |         63.11        | 
| Reptile + BN [2]         |         65.99        | 
| Relation Net [6]         |         67.07        | 
| MAML + Transduction [5]  |         66.19        | 
| TPN [7]                  |         69.43        | 
| TPN (Higher K) [7]       |         69.86        | 
| **(ours)** EGNN          |         **76.37**        | 

#### tieredImageNet, non-transductive

| Model                    |  5-way 5-shot acc (%)| 
|--------------------------|  ------------------: | 
| Reptile [2]              |         66.47        | 
| Prototypical Net [3]     |         69.57        | 
| **(ours)** EGNN          |         **70.98**        | 

#### tieredImageNet, transductive

| Model                    |  5-way 5-shot acc (%)| 
|--------------------------|  ------------------: | 
| MAML [5]                 |         70.30        | 
| Reptile + BN [2]         |         71.03        | 
| Relation Net [6]         |         71.31        | 
| MAML + Transduction [5]  |         70.83        | 
| TPN [7]                  |         72.58        | 
| **(ours)** EGNN          |         **80.15**        | 


#### miniImageNet, semi-supervised, 5-way 5-shot

| Model                    |  20%                 | 40%                 | 60%                 | 100%                 | 
|--------------------------|  ------------------: | ------------------: | ------------------: | ------------------:  | 
| GNN-LabeledOnly [4]       |      50.33                |      56.91               |        -             |        66.41              |
| GNN-Semi [4]             |      52.45                |      58.76               |        -             |        66.41              |
| EGNN-LabeledOnly         |      52.86                |        -             |            -         |            66.85          |
| EGNN-Semi                |      61.88                |        62.52             |        63.53             |    66.85                  |
| EGNN-LabeledOnly (Transductive) |      59.18         |         -            |           -          |           76.37           |
| EGNN-Semi (Transductive)        |      63.62         |        64.32             |        66.37             |   76.37                   |


#### miniImageNet, cross-way experiment
| Model                    |  train way                 | test way                 |  Accuracy |
|--------------------------|  ------------------: | ------------------: | ------------------: |
| GNN       |      5                |      5               |      66.41     |
| GNN       |      5                |      10               |     N/A      |
| GNN       |      10                |     10            |       51.75    |
| GNN       |      10             |      5              |       N/A    |
| EGNN       |      5             |      5              |       76.37    |
| EGNN       |      5             |      10              |       56.35    |
| EGNN       |      10             |      10              |       57.61   |
| EGNN       |      10             |      5              |       76.27   |



### References
```
[1] O. Vinyals et al. Matching networks for one shot learning.
[2] A Nichol, J Achiam, J Schulman, On first-order meta-learning algorithms.
[3] J. Snell, K. Swersky, and R. S. Zemel. Prototypical networks for few-shot learning.
[4] V Garcia, J Bruna, Few-shot learning with graph neural network.
[5] C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast adaptation of deep networks.
[6] F. Sung et al, Learning to Compare: Relation Network for Few-Shot Learning.
[7] Y Liu, J Lee, M Park, S Kim, Y Yang, Transductive propagation network for few-shot learning.
