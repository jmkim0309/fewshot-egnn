from torchtools import tt


__author__ = 'namju.kim@kakaobrain.com'


def accuracy(prob, label, ignore_index=-100):

    # argmax
    pred = prob.max(1)[1].type_as(label)

    # masking
    mask = label.ne(ignore_index)
    pred = pred.masked_select(mask)
    label = label.masked_select(mask)

    # calc accuracy
    hit = tt.nvar(pred.eq(label).long().sum())
    acc = hit / label.size(0)
    return acc
