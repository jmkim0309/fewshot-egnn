from torchtools import nn


#
# Reshape layer for Sequential or ModuleList
#
class Reshape(nn.Module):

    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

    def extra_repr(self):
        return 'shape={}'.format(self.shape)