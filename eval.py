from torchtools import *
from data import MiniImagenetLoader, TieredImagenetLoader
from model import EmbeddingImagenet, GraphNetwork, ConvNet
import shutil
import os
import random
from train import ModelTrainer

if __name__ == '__main__':

    tt.arg.test_model = 'D-mini_N-5_K-1_U-0_L-3_B-40_T-True' if tt.arg.test_model is None else tt.arg.test_model

    list1 = tt.arg.test_model.split("_")
    param = {}
    for i in range(len(list1)):
        param[list1[i].split("-", 1)[0]] = list1[i].split("-", 1)[1]
    tt.arg.dataset = param['D']
    tt.arg.num_ways = int(param['N'])
    tt.arg.num_shots = int(param['K'])
    tt.arg.num_unlabeled = int(param['U'])
    tt.arg.num_layers = int(param['L'])
    tt.arg.meta_batch_size = int(param['B'])
    tt.arg.transductive = False if param['T'] == 'False' else True


    ####################
    tt.arg.device = 'cuda:0' if tt.arg.device is None else tt.arg.device
    # replace dataset_root with your own
    tt.arg.dataset_root = '/data/private/dataset'
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 1 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.num_unlabeled = 0 if tt.arg.num_unlabeled is None else tt.arg.num_unlabeled
    tt.arg.num_layers = 3 if tt.arg.num_layers is None else tt.arg.num_layers
    tt.arg.meta_batch_size = 40 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
    tt.arg.transductive = False if tt.arg.transductive is None else tt.arg.transductive
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1 if tt.arg.num_gpus is None else tt.arg.num_gpus

    tt.arg.num_ways_train = tt.arg.num_ways
    tt.arg.num_ways_test = tt.arg.num_ways

    tt.arg.num_shots_train = tt.arg.num_shots
    tt.arg.num_shots_test = tt.arg.num_shots

    tt.arg.train_transductive = tt.arg.transductive
    tt.arg.test_transductive = tt.arg.transductive

    # model parameter related
    tt.arg.num_edge_features = 96
    tt.arg.num_node_features = 96
    tt.arg.emb_size = 128

    # train, test parameters
    tt.arg.train_iteration = 100000 if tt.arg.dataset == 'mini' else 200000
    tt.arg.test_iteration = 10000
    tt.arg.test_interval = 5000
    tt.arg.test_batch_size = 10
    tt.arg.log_step = 1000

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 15000 if tt.arg.dataset == 'mini' else 30000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    #set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)

    # set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # to check
    exp_name = 'D-{}'.format(tt.arg.dataset)
    exp_name += '_N-{}_K-{}_U-{}'.format(tt.arg.num_ways, tt.arg.num_shots, tt.arg.num_unlabeled)
    exp_name += '_L-{}_B-{}'.format(tt.arg.num_layers, tt.arg.meta_batch_size)
    exp_name += '_T-{}'.format(tt.arg.transductive)


    if not exp_name == tt.arg.test_model:
        print(exp_name)
        print(tt.arg.test_model)
        print('Test model and input arguments are mismatched!')
        AssertionError()

    gnn_module = GraphNetwork(in_features=tt.arg.emb_size,
                              node_features=tt.arg.num_edge_features,
                              edge_features=tt.arg.num_node_features,
                              num_layers=tt.arg.num_layers,
                              dropout=tt.arg.dropout)

    if tt.arg.dataset == 'mini':
        test_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='test')
    elif tt.arg.dataset == 'tiered':
        test_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='test')
    else:
        print('Unknown dataset!')


    data_loader = {'test': test_loader}

    # create trainer
    tester = ModelTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           data_loader=data_loader)


    #checkpoint = torch.load('asset/checkpoints/{}/'.format(exp_name) + 'model_best.pth.tar')
    checkpoint = torch.load('./trained_models/{}/'.format(exp_name) + 'model_best.pth.tar')


    tester.enc_module.load_state_dict(checkpoint['enc_module_state_dict'])
    print("load pre-trained enc_nn done!")

    # initialize gnn pre-trained
    tester.gnn_module.load_state_dict(checkpoint['gnn_module_state_dict'])
    print("load pre-trained egnn done!")

    tester.val_acc = checkpoint['val_acc']
    tester.global_step = checkpoint['iteration']

    print(tester.global_step)


    tester.eval(partition='test')





