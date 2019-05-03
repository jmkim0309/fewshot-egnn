# fewshot-egnn

### Introduction

The current project page provides pytorch code that implements the following CVPR2019 paper:   
**Title:**      "Edge-labeling Graph Neural Network for Few-shot Learning"    
**Authors:**     Jongmin Kim, Taesup Kim, Sungwoong Kim, Chang D.Yoo

**Institution:** KAIST, KaKaoBrain     
**Code:**        https://github.com/khy0809/fewshot-egnn  
**Arxiv:**       -

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
  journal={arXiv preprint arXiv:-},
  year={2019}
}
```


### Platform
- This code was developed and tested with pytorch version 0.4.1

### Setting
- In ```train.py```, replace the dataset root directory with your own:

  tt.arg.dataset_root = '/data/private/dataset'



### Training & evaluation

```
# ************************** miniImagenet, 5way 1shot *****************************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 1 --transductive False
$ python3 train.py --dataset mini --num_ways 5 --num_shots 1 --transductive True

# ************************** miniImagenet, 5way 5shot *****************************
$ python3 trainer.py --dataset mini --num_ways 5 --num_shots 5 --trainsductive False
$ python3 trainer.py --dataset mini --num_ways 5 --num_shots 5 --trainsductive True

# ************************** miniImagenet, 10way 5shot *****************************
$ python3 trainer.py --dataset mini --num_ways 10 --num_shots 5 --meta_batch_size 20 --trainsductive True

# ************************** tieredImagenet, 5way 5shot *****************************
$ python3 trainer.py --dataset tiered --num_ways 5 --num_shots 5 --trainsductive False
$ python3 trainer.py --dataset tiered --num_ways 5 --num_shots 5 --trainsductive True

# **************** miniImagenet, 5way 5shot, 20% labeled (semi) *********************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 5 --num_unlabeled 4 --transductive False
$ python3 train.py --dataset mini --num_ways 5 --num_shots 5 --num_unlabeled 4 --transductive True

```


## Result
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
| MAML [1]                 |         63.11        | 
| Reptile + BN [2]         |         65.99        | 
| Relation Net [3]         |         67.07        | 
| MAML + Transduction [4]  |         66.19        | 
| TPN [4]                  |         69.43        | 
| TPN (Higher K) [4]       |         69.86        | 
| **(ours)** EGNN          |         **76.37**        | 

#### tieredImageNet, non-transductive

| Model                    |  5-way 5-shot acc (%)| 
|--------------------------|  ------------------: | 
| Reptile [1]              |         66.47        | 
| Prototypical Net [2]     |         69.57        | 
| **(ours)** EGNN          |         **70.98**        | 

#### tieredImageNet, transductive

| Model                    |  5-way 5-shot acc (%)| 
|--------------------------|  ------------------: | 
| MAML [1]                 |         70.30        | 
| Reptile + BN [2]         |         71.03        | 
| Relation Net [3]         |         71.31        | 
| MAML + Transduction [4]  |         70.83        | 
| TPN [4]                  |         72.58        | 
| **(ours)** EGNN          |         **80.15**        | 


#### miniImageNet, semi-supervised, 5-way 5-shot

| Model                    |  20%                 | 40%                 | 60%                 | 100%                 | 
|--------------------------|  ------------------: | ------------------: | ------------------: | ------------------:  | 
| GNN-LabeledOnly[1]       |      50.33                |      56.91               |        -             |        66.41              |
| GNN-Semi [2]             |      52.45                |      58.76               |        -             |        66.41              |
| EGNN-LabeledOnly         |      52.86                |        -             |            -         |            66.85          |
| EGNN-Semi                |      61.88                |        62.52             |        63.53             |    66.85                  |
| EGNN-LabeledOnly (Transductive) |      59.18         |         -            |           -          |           76.37           |
| EGNN-Semi (Transductive)        |      63.62         |        64.32             |        66.37             |   76.37                   |


### References
```
[1] B. Hariharan and R. Girshick. Low-shot visual recognition by shrinking and hallucinating features.
[2] Y.-X. Wang and R. Girshick, M. Hebert, B. Hariharan. Low-shot learning from imaginary data.
[3] O. Vinyals et al. Matching networks for one shot learning.
[4] J. Snell, K. Swersky, and R. S. Zemel. Prototypical networks for few-shot learning.
[5] S. Ravi and H. Larochelle. Optimization as a model for few-shot learning.
[6] C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast adaptation of deep networks.
```