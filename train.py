from torchtools import *
from data import MiniImagenetLoader, TieredImagenetLoader
from model import EmbeddingImagenet, GraphNetwork, ConvNet
import shutil
import os
import random
#import seaborn as sns


class ModelTrainer(object):
    def __init__(self,
                 enc_module,
                 gnn_module,
                 data_loader):
        # set encoder and gnn
        self.enc_module = enc_module.to(tt.arg.device)
        self.gnn_module = gnn_module.to(tt.arg.device)

        if tt.arg.num_gpus > 1:
            print('Construct multi-gpu model ...')
            self.enc_module = nn.DataParallel(self.enc_module, device_ids=[0, 1, 2, 3], dim=0)
            self.gnn_module = nn.DataParallel(self.gnn_module, device_ids=[0, 1, 2, 3], dim=0)

            print('done!\n')

        # get data loader
        self.data_loader = data_loader

        # set optimizer
        self.module_params = list(self.enc_module.parameters()) + list(self.gnn_module.parameters())

        # set optimizer
        self.optimizer = optim.Adam(params=self.module_params,
                                    lr=tt.arg.lr,
                                    weight_decay=tt.arg.weight_decay)

        # set loss
        self.edge_loss = nn.BCELoss(reduction='none')

        self.node_loss = nn.CrossEntropyLoss(reduction='none')

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        val_acc = self.val_acc

        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways_train * tt.arg.num_shots_train
        num_queries = tt.arg.num_ways_train * 1
        num_samples = num_supports + num_queries
        support_edge_mask = torch.zeros(tt.arg.meta_batch_size, num_samples, num_samples).to(tt.arg.device)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask

        evaluation_mask = torch.ones(tt.arg.meta_batch_size, num_samples, num_samples).to(tt.arg.device)
        # for semi-supervised setting, ignore unlabeled support sets for evaluation
        for c in range(tt.arg.num_ways_train):
            evaluation_mask[:,
            ((c + 1) * tt.arg.num_shots_train - tt.arg.num_unlabeled):(c + 1) * tt.arg.num_shots_train,
            :num_supports] = 0
            evaluation_mask[:, :num_supports,
            ((c + 1) * tt.arg.num_shots_train - tt.arg.num_unlabeled):(c + 1) * tt.arg.num_shots_train] = 0

        # for each iteration
        for iter in range(self.global_step + 1, tt.arg.train_iteration + 1):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step = iter

            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader['train'].get_task_batch(num_tasks=tt.arg.meta_batch_size,
                                                                     num_ways=tt.arg.num_ways_train,
                                                                     num_shots=tt.arg.num_shots_train,
                                                                     seed=iter + tt.arg.seed)

            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)

            # set init edge
            init_edge = full_edge.clone()  # batch_size x 2 x num_samples x num_samples
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

            # for semi-supervised setting,
            for c in range(tt.arg.num_ways_train):
                init_edge[:, :, ((c+1) * tt.arg.num_shots_train - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_train, :num_supports] = 0.5
                init_edge[:, :, :num_supports, ((c+1) * tt.arg.num_shots_train - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_train] = 0.5

            # set as train mode
            self.enc_module.train()
            self.gnn_module.train()

            # (1) encode data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1) # batch_size x num_samples x featdim

            # (2) predict edge logit (consider only the last layer logit, num_tasks x 2 x num_samples x num_samples)
            if tt.arg.train_transductive:
                full_logit_layers = self.gnn_module(node_feat=full_data, edge_feat=init_edge)
            else:
                evaluation_mask[:, num_supports:, num_supports:] = 0 # ignore query-query edges, since it is non-transductive setting
                # input_node_feat: (batch_size x num_queries) x (num_support + 1) x featdim
                # input_edge_feat: (batch_size x num_queries) x 2 x (num_support + 1) x (num_support + 1)
                support_data = full_data[:, :num_supports] # batch_size x num_support x featdim
                query_data = full_data[:, num_supports:] # batch_size x num_query x featdim
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1, 1) # batch_size x num_queries x num_support x featdim
                support_data_tiled = support_data_tiled.view(tt.arg.meta_batch_size * num_queries, num_supports, -1) # (batch_size x num_queries) x num_support x featdim
                query_data_reshaped = query_data.contiguous().view(tt.arg.meta_batch_size * num_queries, -1).unsqueeze(1) # (batch_size x num_queries) x 1 x featdim
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped], 1) # (batch_size x num_queries) x (num_support + 1) x featdim

                input_edge_feat = 0.5 * torch.ones(tt.arg.meta_batch_size, 2, num_supports + 1, num_supports + 1).to(tt.arg.device) # batch_size x 2 x (num_support + 1) x (num_support + 1)

                input_edge_feat[:, :, :num_supports, :num_supports] = init_edge[:, :, :num_supports, :num_supports] # batch_size x 2 x (num_support + 1) x (num_support + 1)
                input_edge_feat = input_edge_feat.repeat(num_queries, 1, 1, 1) #(batch_size x num_queries) x 2 x (num_support + 1) x (num_support + 1)

                # logit: (batch_size x num_queries) x 2 x (num_support + 1) x (num_support + 1)
                logit_layers = self.gnn_module(node_feat=input_node_feat, edge_feat=input_edge_feat)

                logit_layers = [logit_layer.view(tt.arg.meta_batch_size, num_queries, 2, num_supports + 1, num_supports + 1) for logit_layer in logit_layers]

                # logit --> full_logit (batch_size x 2 x num_samples x num_samples)
                full_logit_layers = []
                for l in range(tt.arg.num_layers):
                    full_logit_layers.append(torch.zeros(tt.arg.meta_batch_size, 2, num_samples, num_samples).to(tt.arg.device))

                for l in range(tt.arg.num_layers):
                    full_logit_layers[l][:, :, :num_supports, :num_supports] = logit_layers[l][:, :, :, :num_supports, :num_supports].mean(1)
                    full_logit_layers[l][:, :, :num_supports, num_supports:] = logit_layers[l][:, :, :, :num_supports, -1].transpose(1, 2).transpose(2, 3)
                    full_logit_layers[l][:, :, num_supports:, :num_supports] = logit_layers[l][:, :, :, -1, :num_supports].transpose(1, 2)

            # (4) compute loss
            full_edge_loss_layers = [self.edge_loss((1-full_logit_layer[:, 0]), (1-full_edge[:, 0])) for full_logit_layer in full_logit_layers]

            # weighted edge loss for balancing pos/neg
            pos_query_edge_loss_layers = [torch.sum(full_edge_loss_layer * query_edge_mask * full_edge[:, 0] * evaluation_mask) / torch.sum(query_edge_mask * full_edge[:, 0] * evaluation_mask) for full_edge_loss_layer in full_edge_loss_layers]
            neg_query_edge_loss_layers = [torch.sum(full_edge_loss_layer * query_edge_mask * (1-full_edge[:, 0]) * evaluation_mask) / torch.sum(query_edge_mask * (1-full_edge[:, 0]) * evaluation_mask) for full_edge_loss_layer in full_edge_loss_layers]
            query_edge_loss_layers = [pos_query_edge_loss_layer + neg_query_edge_loss_layer for (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in zip(pos_query_edge_loss_layers, neg_query_edge_loss_layers)]

            # compute accuracy
            full_edge_accr_layers = [self.hit(full_logit_layer, 1-full_edge[:, 0].long()) for full_logit_layer in full_logit_layers]
            query_edge_accr_layers = [torch.sum(full_edge_accr_layer * query_edge_mask * evaluation_mask) / torch.sum(query_edge_mask * evaluation_mask) for full_edge_accr_layer in full_edge_accr_layers]

            # compute node loss & accuracy (num_tasks x num_quries x num_ways)
            query_node_pred_layers = [torch.bmm(full_logit_layer[:, 0, num_supports:, :num_supports], self.one_hot_encode(tt.arg.num_ways_train, support_label.long())) for full_logit_layer in full_logit_layers] # (num_tasks x num_quries x num_supports) * (num_tasks x num_supports x num_ways)
            query_node_accr_layers = [torch.eq(torch.max(query_node_pred_layer, -1)[1], query_label.long()).float().mean() for query_node_pred_layer in query_node_pred_layers]

            total_loss_layers = query_edge_loss_layers

            # update model
            total_loss = []
            for l in range(tt.arg.num_layers - 1):
                total_loss += [total_loss_layers[l].view(-1) * 0.5]
            total_loss += [total_loss_layers[-1].view(-1) * 1.0]
            total_loss = torch.mean(torch.cat(total_loss, 0))

            total_loss.backward()

            self.optimizer.step()

            # adjust learning rate
            self.adjust_learning_rate(optimizers=[self.optimizer],
                                      lr=tt.arg.lr,
                                      iter=self.global_step)

            # logging
            tt.log_scalar('train/edge_loss', query_edge_loss_layers[-1], self.global_step)
            tt.log_scalar('train/edge_accr', query_edge_accr_layers[-1], self.global_step)
            tt.log_scalar('train/node_accr', query_node_accr_layers[-1], self.global_step)

            # evaluation
            if self.global_step % tt.arg.test_interval == 0:
                val_acc = self.eval(partition='val')

                is_best = 0

                if val_acc >= self.val_acc:
                    self.val_acc = val_acc
                    is_best = 1

                tt.log_scalar('val/best_accr', self.val_acc, self.global_step)

                self.save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'gnn_module_state_dict': self.gnn_module.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                    }, is_best)

            tt.log_step(global_step=self.global_step)

    def eval(self, partition='test', log_flag=True):
        best_acc = 0
        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways_test * tt.arg.num_shots_test
        num_queries = tt.arg.num_ways_test * 1
        num_samples = num_supports + num_queries
        support_edge_mask = torch.zeros(tt.arg.test_batch_size, num_samples, num_samples).to(tt.arg.device)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask
        evaluation_mask = torch.ones(tt.arg.test_batch_size, num_samples, num_samples).to(tt.arg.device)
        # for semi-supervised setting, ignore unlabeled support sets for evaluation
        for c in range(tt.arg.num_ways_test):
            evaluation_mask[:,
            ((c + 1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c + 1) * tt.arg.num_shots_test,
            :num_supports] = 0
            evaluation_mask[:, :num_supports,
            ((c + 1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c + 1) * tt.arg.num_shots_test] = 0

        query_edge_losses = []
        query_edge_accrs = []
        query_node_accrs = []

        # for each iteration
        for iter in range(tt.arg.test_iteration//tt.arg.test_batch_size):
            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader[partition].get_task_batch(num_tasks=tt.arg.test_batch_size,
                                                                       num_ways=tt.arg.num_ways_test,
                                                                       num_shots=tt.arg.num_shots_test,
                                                                       seed=iter)

            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)

            # set init edge
            init_edge = full_edge.clone()
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

            # for semi-supervised setting,
            for c in range(tt.arg.num_ways_test):
                init_edge[:, :, ((c+1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_test, :num_supports] = 0.5
                init_edge[:, :, :num_supports, ((c+1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_test] = 0.5

            # set as train mode
            self.enc_module.eval()
            self.gnn_module.eval()

            # (1) encode data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)

            # (2) predict edge logit (consider only the last layer logit, num_tasks x 2 x num_samples x num_samples)
            if tt.arg.test_transductive:
                full_logit_all = self.gnn_module(node_feat=full_data, edge_feat=init_edge)
                full_logit = full_logit_all[-1]
            else:
                evaluation_mask[:, num_supports:, num_supports:] = 0  # ignore query-query edges, since it is non-transductive setting

                full_logit = torch.zeros(tt.arg.test_batch_size, 2, num_samples, num_samples).to(tt.arg.device)

                # input_node_feat: (batch_size x num_queries) x (num_support + 1) x featdim
                # input_edge_feat: (batch_size x num_queries) x 2 x (num_support + 1) x (num_support + 1)
                support_data = full_data[:, :num_supports] # batch_size x num_support x featdim
                query_data = full_data[:, num_supports:] # batch_size x num_query x featdim
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1, 1) # batch_size x num_queries x num_support x featdim
                support_data_tiled = support_data_tiled.view(tt.arg.test_batch_size * num_queries, num_supports, -1) # (batch_size x num_queries) x num_support x featdim
                query_data_reshaped = query_data.contiguous().view(tt.arg.test_batch_size * num_queries, -1).unsqueeze(1) # (batch_size x num_queries) x 1 x featdim
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped], 1) # (batch_size x num_queries) x (num_support + 1) x featdim

                input_edge_feat = 0.5 * torch.ones(tt.arg.test_batch_size, 2, num_supports + 1, num_supports + 1).to(tt.arg.device)  # batch_size x 2 x (num_support + 1) x (num_support + 1)

                input_edge_feat[:, :, :num_supports, :num_supports] = init_edge[:, :, :num_supports, :num_supports]  # batch_size x 2 x (num_support + 1) x (num_support + 1)
                input_edge_feat = input_edge_feat.repeat(num_queries, 1, 1, 1)  # (batch_size x num_queries) x 2 x (num_support + 1) x (num_support + 1)

                # logit: (batch_size x num_queries) x 2 x (num_support + 1) x (num_support + 1)
                logit = self.gnn_module(node_feat=input_node_feat, edge_feat=input_edge_feat)[-1]

                logit = logit.view(tt.arg.test_batch_size, num_queries, 2, num_supports + 1, num_supports + 1)

                # batch_size x num_queries x 2 x (num_support + 1) x (num_support + 1)
                # logit --> full_logit (batch_size x 2 x num_samples x num_samples)
                full_logit[:, :, :num_supports, :num_supports] = logit[:, :, :, :num_supports, :num_supports].mean(1)
                full_logit[:, :, :num_supports, num_supports:] = logit[:, :, :, :num_supports, -1].transpose(1, 2).transpose(2, 3)
                full_logit[:, :, num_supports:, :num_supports] = logit[:, :, :, -1, :num_supports].transpose(1, 2)

            # (4) compute loss
            full_edge_loss = self.edge_loss(1-full_logit[:, 0], 1-full_edge[:, 0])

            query_edge_loss =  torch.sum(full_edge_loss * query_edge_mask * evaluation_mask) / torch.sum(query_edge_mask * evaluation_mask)

            # weighted loss for balancing pos/neg
            pos_query_edge_loss = torch.sum(full_edge_loss * query_edge_mask * full_edge[:, 0] * evaluation_mask) / torch.sum(query_edge_mask * full_edge[:, 0] * evaluation_mask)
            neg_query_edge_loss = torch.sum(full_edge_loss * query_edge_mask * (1-full_edge[:, 0]) * evaluation_mask) / torch.sum(query_edge_mask * (1-full_edge[:, 0]) * evaluation_mask)
            query_edge_loss = pos_query_edge_loss + neg_query_edge_loss

            # compute accuracy
            full_edge_accr = self.hit(full_logit, 1-full_edge[:, 0].long())
            query_edge_accr = torch.sum(full_edge_accr * query_edge_mask * evaluation_mask) / torch.sum(query_edge_mask * evaluation_mask)

            # compute node accuracy (num_tasks x num_quries x num_ways)
            query_node_pred = torch.bmm(full_logit[:, 0, num_supports:, :num_supports], self.one_hot_encode(tt.arg.num_ways_test, support_label.long())) # (num_tasks x num_quries x num_supports) * (num_tasks x num_supports x num_ways)
            query_node_accr = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean()

            query_edge_losses += [query_edge_loss.item()]
            query_edge_accrs += [query_edge_accr.item()]
            query_node_accrs += [query_node_accr.item()]

        # logging
        if log_flag:
            tt.log('---------------------------')
            tt.log_scalar('{}/edge_loss'.format(partition), np.array(query_edge_losses).mean(), self.global_step)
            tt.log_scalar('{}/edge_accr'.format(partition), np.array(query_edge_accrs).mean(), self.global_step)
            tt.log_scalar('{}/node_accr'.format(partition), np.array(query_node_accrs).mean(), self.global_step)

            tt.log('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                   (iter,
                    np.array(query_node_accrs).mean() * 100,
                    np.array(query_node_accrs).std() * 100,
                    1.96 * np.array(query_node_accrs).std() / np.sqrt(float(len(np.array(query_node_accrs)))) * 100))
            tt.log('---------------------------')

        return np.array(query_node_accrs).mean()

    def adjust_learning_rate(self, optimizers, lr, iter):
        new_lr = lr * (0.5 ** (int(iter / tt.arg.dec_lr)))

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def label2edge(self, label):
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        # compute edge
        edge = torch.eq(label_i, label_j).float().to(tt.arg.device)

        # expand
        edge = edge.unsqueeze(1)
        edge = torch.cat([edge, 1 - edge], 1)
        return edge

    def hit(self, logit, label):
        pred = logit.max(1)[1]
        hit = torch.eq(pred, label).float()
        return hit

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].to(tt.arg.device)

    def save_checkpoint(self, state, is_best):
        torch.save(state, 'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar')
        if is_best:
            shutil.copyfile('asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar',
                            'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'model_best.pth.tar')

def set_exp_name():
    exp_name = 'D-{}'.format(tt.arg.dataset)
    exp_name += '_N-{}_K-{}_U-{}'.format(tt.arg.num_ways, tt.arg.num_shots, tt.arg.num_unlabeled)
    exp_name += '_L-{}_B-{}'.format(tt.arg.num_layers, tt.arg.meta_batch_size)
    exp_name += '_T-{}'.format(tt.arg.transductive)
    exp_name += '_SEED-{}'.format(tt.arg.seed)

    return exp_name

if __name__ == '__main__':

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
    tt.arg.test_interval = 5000 if tt.arg.test_interval is None else tt.arg.test_interval
    tt.arg.test_batch_size = 10
    tt.arg.log_step = 1000 if tt.arg.log_step is None else tt.arg.log_step

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 15000 if tt.arg.dataset == 'mini' else 30000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    tt.arg.experiment = set_exp_name() if tt.arg.experiment is None else tt.arg.experiment

    print(set_exp_name())

    #set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tt.arg.log_dir_user = tt.arg.log_dir if tt.arg.log_dir_user is None else tt.arg.log_dir_user
    tt.arg.log_dir = tt.arg.log_dir_user

    if not os.path.exists('asset/checkpoints'):
        os.makedirs('asset/checkpoints')
    if not os.path.exists('asset/checkpoints/' + tt.arg.experiment):
        os.makedirs('asset/checkpoints/' + tt.arg.experiment)


    enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)

    gnn_module = GraphNetwork(in_features=tt.arg.emb_size,
                              node_features=tt.arg.num_edge_features,
                              edge_features=tt.arg.num_node_features,
                              num_layers=tt.arg.num_layers,
                              dropout=tt.arg.dropout)

    if tt.arg.dataset == 'mini':
        train_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='val')
    elif tt.arg.dataset == 'tiered':
        train_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='train')
        valid_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='val')
    else:
        print('Unknown dataset!')

    data_loader = {'train': train_loader,
                   'val': valid_loader
                   }

    # create trainer
    trainer = ModelTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           data_loader=data_loader)

    trainer.train()
