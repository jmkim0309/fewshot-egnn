# fewshot-egnn

PyTorch implementation of the following paper:

  "Edge-labeling Graph Neural Network for Few-shot Learning", CVPR 2019 [arXiv link]

# Platform
- pytorch 0.4.1, python 3

## Setting
- In ```data.py```, replace the dataset root directory with your own:

  root_dir = '/mnt/hdd/jmkim/maml_pytorch/asset/data/miniImagenet/'

- For resnet experiment, download the pre-trained 64-way cls models from the following link:
  https://drive.google.com/open?id=1pic_LWnRUP1IaGJLvujF-0k9WtSHPW_Y

  and place it under ./asset/pre-trained/

## Supervised few-shot classification 
```
# miniImagenet, 5way 1shot, non-transductive
$ python3 train.py --dataset mini --num_ways 5 --num_shots 1 --transductive False
# miniImagenet, 5way 1shot, transductive
$ python3 train.py --dataset mini --num_ways 5 --num_shots 1 --transductive True

# miniImagenet, 5way 5shot, non-transductive
$ python3 trainer.py --dataset mini --num_ways 5 --num_shots 5 --trainsductive False
# miniImagenet, 5way 5shot, transductive
$ python3 trainer.py --dataset mini --num_ways 5 --num_shots 5 --trainsductive True
# miniImagenet, 10way 5shot, transductive
$ python3 trainer.py --dataset mini --num_ways 10 --num_shots 5 --meta_batch_size 20 --trainsductive True

# tieredImagenet, 5way 1shot, non-transductive
$ python3 train.py --dataset tiered --num_ways 5 --num_shots 1 --meta_batch_size 100 --transductive False
# miniImagenet, 5way 1shot, transductive
$ python3 train.py --dataset tiered --num_ways 5 --num_shots 1 --meta_batch_size 100 --transductive True

# tieredImagenet, 5way 5shot, non-transductive
$ python3 trainer.py --dataset tiered --num_ways 5 --num_shots 5 --trainsductive False
# miniImagenet, 5way 5shot, transductive
$ python3 trainer.py --dataset tiered --num_ways 5 --num_shots 5 --trainsductive True

```

## Semi-supervsied
```
# miniImagenet, 5way 5shot, 20% labeled, transductive
$ python3 train.py --dataset mini --num_ways 5 --num_shots 5 --num_unlabeled 4 --transductive True

```

### Adapt Metric_NN, while Enc_NN is updated only in outer-loop
```
# 5-way 5-shot, initilized with 5-way 5-shot pre-trained model (enc_nn + metric_nn)
$ python3 trainer.py --config asset/config/mini-gnn-maml-N5S5-N5S5init-joint.ini --reinit 1
```

## Training (resnet-18, resnet-50) 
### Adapt Metric_NN, while Enc_NN is fixed
```
# 5-way 5-shot, initilized with 64-way cls pre-trained enc_nn model (metric_nn is trained from scratch!)
$ python3 trainer.py --config asset/config/mini-resnet18-gnn-maml-N5S5-64wayinit-scratch.ini --reinit 1

# TODO: 5-way 5-shot, initilized with 5-way 5-shot pre-trained model (enc_nn + metric_nn)

```
### Adapt Metric_NN, while Enc_NN is updated only in outer-loop
```
# 5-way 5-shot, initilized with 64-way cls pre-trained enc_nn model (metric_nn is trained from scratch!)
$ python3 trainer.py --config asset/config/mini-resnet18-gnn-maml-N5S5-64wayinit-scratch-joint.ini --reinit 1

# TODO: 5-way 5-shot, initilized with 5-way 5-shot pre-trained model (enc_nn + metric_nn)
```

## Result
- MiniImagenet, 5-way, 4convblock

| Model                               |   |   |      |   |5-way Acc.|  |  |     |     |
|-------------------------------------|----|---|-----|---|----------|--|--|-----|-----|
|                                     | |1-shot|     || 2-shot|| |5-shot |    |
|                                     |train|val|test|train|val|test|train|val|test|
| MAML                                | -|-|48.70  | -|-|-| -|-|63.11  | 
| GNN                                 | -|-|50.33  | -|-|-| -|-|66.41  | 
| MAML (our implementation)           | 51.29|45.24|44.58 | 63.93|52.57|52.55| 74.50|60.99|61.97 | 
| GNN (our implementation)            | 62.80|47.62|44.64 | 76.40|54.48|51.41| 82.00|60.37|60.45 |
| GNN + MAML (N5S1init)               | -|-|-      | 69.64|52.59|50.63| 73.58|57.86|56.78 |
| GNN + MAML (N5S2init)      		      | -|-|-      | 74.25|54.36|51.05| -|-|-   |
| GNN + MAML (N5S5init)               | -|-|-      | -|-|- | 81.83|60.13|59.23 |

