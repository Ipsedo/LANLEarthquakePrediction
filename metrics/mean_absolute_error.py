import torch as th


def mae(preds, targets):
    if len(preds.size()) != 1:
        raise ValueError("preds must have size of dim 1 ! (current %d)" % (len(preds.size())))
    if len(targets.size()) != 1:
        raise ValueError("targets must have size of dim 1 ! (current %d)" % (len(targets.size())))
    if preds.size(0) != targets.size(0):
        raise ValueError("preds and targets must have the same shape ! (preds.size(0) = %d and targets.size(0) = % d" % (preds.size(0), targets.size(0)))
    return th.div(th.sum(th.abs(targets - preds)), targets.size(0)).cpu().item()
