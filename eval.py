from torchtools import *
from data import MiniImagenetLoader, TieredImagenetLoader
from model import EmbeddingImagenet, GraphNetwork, ConvNet
import shutil
import os
import random
from train import ModelTrainer

if __name__ == '__main__':

    #exp_name = 'D-mini_NTr-5_NTe-5_STr-5_STe-5_U-0_ENC-4conv_L-3_B-40_NQTr-1_NQTe-1_EF-96_NF-96_SD-False_TTr-True_TTe-True_NL-0.0_LR-0.001_ELT-bce'
    #exp_name = 'D-mini_NTr-10_NTe-10_STr-5_STe-5_U-0_ENC-4conv_L-3_B-20_NQTr-1_NQTe-1_EF-96_NF-96_SD-False_TTr-True_TTe-True_NL-0.0_LR-0.001_ELT-bce_DROP-0.0'
    exp_name = 'D-tiered_NTr-5_NTe-5_STr-5_STe-5_U-0_ENC-4conv_L-3_B-40_NQTr-1_NQTe-1_EF-96_NF-96_SD-False_TTr-True_TTe-True_NL-0.0_LR-0.001_ELT-bce'

    list1 = exp_name.split("_")
    param = {}
    for i in range(len(list1)):
        param[list1[i].split("-", 1)[0]] = list1[i].split("-", 1)[1]
    tt.arg.dataset = param['D']
    tt.arg.num_ways_train = int(param['NTr'])
    tt.arg.num_ways_test = int(param['NTe'])
    tt.arg.meta_batch_size = int(param['B'])
    tt.arg.num_shots_train = int(param['STr'])
    tt.arg.num_shots_test = int(param['STe'])
    tt.arg.num_layers = int(param['L'])
    tt.arg.num_unlabeled = int(param['U'])
    tt.arg.enc_nn = param['ENC']
    tt.arg.train_transductive = False if param['TTr'] == 'False' else True
    tt.arg.test_transductive = False if param['TTe'] == 'False' else True


    # exp parameters
    tt.arg.device = 'cuda:0' if tt.arg.device is None else tt.arg.device

    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways_train = 5 if tt.arg.num_ways_train is None else tt.arg.num_ways_train
    tt.arg.num_ways_test = 5 if tt.arg.num_ways_test is None else tt.arg.num_ways_test

    tt.arg.num_shots_train = 5 if tt.arg.num_shots_train is None else tt.arg.num_shots_train
    tt.arg.num_shots_test = 5 if tt.arg.num_shots_test is None else tt.arg.num_shots_test
    tt.arg.num_unlabeled = 0 if tt.arg.num_unlabeled is None else tt.arg.num_unlabeled  # number of unlabeled support per class (for semi-supervised setting)
    tt.arg.num_queries_train = 1 if tt.arg.num_queries_train is None else tt.arg.num_queries_train # number of queries per class (train task)
    tt.arg.num_queries_test = 1 if tt.arg.num_queries_test is None else tt.arg.num_queries_test # number of queries per class (test task)
    tt.arg.meta_batch_size = 40 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size

    tt.arg.num_edge_features = 96 if tt.arg.num_edge_features is None else tt.arg.num_edge_features
    tt.arg.num_node_features = 96 if tt.arg.num_node_features is None else tt.arg.num_node_features
    tt.arg.num_layers = 3 if tt.arg.num_layers is None else tt.arg.num_layers
    tt.arg.emb_size = 128 if tt.arg.emb_size is None else tt.arg.emb_size

    tt.arg.train_iteration = 100000 if tt.arg.train_iteration is None else tt.arg.train_iteration
    tt.arg.test_iteration = 10000 if tt.arg.test_iteration is None else tt.arg.test_iteration
    tt.arg.test_interval = 5000 if tt.arg.test_interval is None else tt.arg.test_interval
    tt.arg.test_batch_size = 10 if tt.arg.test_batch_size is None else tt.arg.test_batch_size
    tt.arg.log_step = 1000 if tt.arg.log_step is None else tt.arg.log_step

    tt.arg.lr = 1e-3 if tt.arg.lr is None else tt.arg.lr
    tt.arg.grad_clip = 5 if tt.arg.grad_clip is None else tt.arg.grad_clip
    tt.arg.weight_decay = 1e-6 if tt.arg.weight_decay is None else tt.arg.weight_decay
    tt.arg.dec_lr = 15000 if tt.arg.dec_lr is None else tt.arg.dec_lr

    tt.arg.separate_dissimilarity = False if tt.arg.separate_dissimilarity is None else tt.arg.separate_dissimilarity

    tt.arg.data_root = '/data/private/dataset' if tt.arg.data_root is None else tt.arg.data_root
    tt.arg.train_transductive = False if tt.arg.train_transductive is None else tt.arg.train_transductive
    tt.arg.test_transductive = False if tt.arg.test_transductive is None else tt.arg.test_transductive
    tt.arg.ll_weight = 0.5 if tt.arg.ll_weight is None else tt.arg.ll_weight
    tt.arg.node_loss_weight = 0.0 if tt.arg.node_loss_weight is None else tt.arg.node_loss_weight
    tt.arg.num_gpus = 1 if tt.arg.num_gpus is None else tt.arg.num_gpus

    tt.arg.enc_nn = '4conv' if tt.arg.enc_nn is None else tt.arg.enc_nn
    tt.arg.seed = 111 if tt.arg.seed is None else tt.arg.seed
    tt.arg.edge_loss_type = 'bce' if tt.arg.edge_loss_type is None else tt.arg.edge_loss_type # 'ce' or 'bce' or 'bceAvg'
    tt.arg.dropout = 0.0 if tt.arg.dropout is None else tt.arg.dropout
    tt.arg.cutout = False if tt.arg.cutout is None else tt.arg.cutout
    tt.arg.visualization = False if tt.arg.visualization is None else tt.arg.visualization

    #set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    if tt.arg.enc_nn == '4conv':
        enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)
    elif tt.arg.enc_nn == 'Conv64':
        net_optionsF = {'userelu': False, 'in_planes': 3, 'out_planes': [64, 64, 64, 64], 'num_stages': 4}
        enc_module = ConvNet(net_optionsF)
        tt.arg.emb_size = 64 * 5 * 5
    else:
        raise NotImplementedError

    # set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # to check
    exp_name_gen = 'D-{}'.format(tt.arg.dataset)
    exp_name_gen += '_NTr-{}_NTe-{}_STr-{}_STe-{}_U-{}'.format(tt.arg.num_ways_train, tt.arg.num_ways_test,
                                                           tt.arg.num_shots_train, tt.arg.num_shots_test,
                                                           tt.arg.num_unlabeled)
    exp_name_gen += '_ENC-{}'.format(tt.arg.enc_nn)
    exp_name_gen += '_L-{}_B-{}'.format(tt.arg.num_layers, tt.arg.meta_batch_size)
    exp_name_gen += '_NQTr-{}_NQTe-{}'.format(tt.arg.num_queries_train, tt.arg.num_queries_test)
    exp_name_gen += '_EF-{}_NF-{}'.format(tt.arg.num_edge_features, tt.arg.num_node_features)
    exp_name_gen += '_SD-{}'.format(tt.arg.separate_dissimilarity)
    exp_name_gen += '_TTr-{}_TTe-{}'.format(tt.arg.train_transductive, tt.arg.test_transductive)
    exp_name_gen += '_NL-{}'.format(tt.arg.node_loss_weight)
    exp_name_gen += '_LR-{}'.format(tt.arg.lr)
    exp_name_gen += '_ELT-{}'.format(tt.arg.edge_loss_type)
    exp_name_gen += '_DROP-{}'.format(tt.arg.dropout)
    exp_name_gen += '_LLW-{}'.format(tt.arg.ll_weight)
    exp_name_gen += '_AUG-'
    exp_name_gen += 'CUTOUT-{}'.format(tt.arg.cutout)


    if not exp_name_gen == exp_name:
        print(exp_name_gen)
        print(exp_name)
        print('Test model and input arguments are mismatched!')
        AssertionError()



    gnn_module = GraphNetwork(in_features=tt.arg.emb_size,
                              node_features=tt.arg.num_edge_features,
                              edge_features=tt.arg.num_node_features,
                              num_layers=tt.arg.num_layers,
                              dropout=tt.arg.dropout)

    if tt.arg.dataset == 'mini':
        test_loader = MiniImagenetLoader(root=tt.arg.data_root, partition='test')
    elif tt.arg.dataset == 'tiered':
        test_loader = TieredImagenetLoader(root=tt.arg.data_root, partition='test')
    else:
        print('Unknown dataset!')

    data_loader = {
        'test': test_loader}

    # create trainer
    tester = ModelTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           data_loader=data_loader)


    checkpoint = torch.load('asset/checkpoints/{}/'.format(exp_name) + 'model_best.pth.tar')


    tester.enc_module.load_state_dict(checkpoint['enc_module_state_dict'])
    print("load pre-trained enc_nn done!")

    # initialize gnn pre-trained
    tester.gnn_module.load_state_dict(checkpoint['gnn_module_state_dict'])
    print("load pre-trained egnn done!")

    tester.val_acc = checkpoint['val_acc']
    tester.global_step = checkpoint['iteration']

    print(tester.global_step)


    tt.arg.num_shots_test = 3
    tt.arg.num_queries_test = 3
    tt.arg.visualization = True

    tt.arg.test_batch_size = 1

    tester.eval(partition='test')

    tt.arg.num_ways_test = 10

    tester.eval(partition='test')




