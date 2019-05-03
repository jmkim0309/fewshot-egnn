from torchtools import nn, optim, tt


__author__ = 'namju.kim@kakaobrain.com'


class SupervisedTrainer(object):

    def __init__(self, model, data_loader, optimizer=None, criterion=None):
        self.global_step = 0
        self.model = model.to(tt.arg.device)
        self.data_loader = data_loader
        self.optimizer = optimizer or optim.Adam(model.parameters())
        self.criterion = criterion or nn.CrossEntropyLoss()

    def train(self, inputs):

        # split inputs
        x, y = inputs

        # forward
        if tt.arg.cuda:
            z = nn.DataParallel(self.model)(x)
        else:
            z = self.model(x)

        # loss
        loss = self.criterion(z, y)

        # accuracy
        acc = tt.accuracy(z, y)

        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # logging
        tt.log_scalar('loss', loss, self.global_step)
        tt.log_scalar('acc', acc, self.global_step)

    def epoch(self, ep_no=None):
        pass

    def run(self):

        # experiment name
        tt.arg.experiment = tt.arg.experiment or self.model.__class__.__name__.lower()

        # load model
        self.global_step = self.model.load_model()
        epoch, min_step = divmod(self.global_step, len(self.data_loader))

        # epochs
        while epoch < (tt.arg.epoch or 1):
            epoch += 1

            # iterations
            for step, inputs in enumerate(self.data_loader, min_step + 1):

                # check step counter
                if step > len(self.data_loader):
                    break

                # increase global step count
                self.global_step += 1

                # update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = tt.arg.lr

                # call train func
                if type(inputs) in [list, tuple]:
                    self.train([tt.var(d) for d in inputs])
                else:
                    self.train(tt.var(inputs))

                # logging
                tt.log_weight(self.model, global_step=self.global_step)
                tt.log_gradient(self.model, global_step=self.global_step)
                tt.log_step(epoch=epoch, global_step=self.global_step,
                            max_epoch=(tt.arg.epoch or 1), max_step=len(self.data_loader))

                # save model
                self.model.save_model(self.global_step)

            # epoch handler
            self.epoch(epoch)

        # save final model
        self.model.save_model(self.global_step, force=True)
