import torch


def accuracy(output, target, present_classes, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum(0)
            res.append(compute_per_class_metric(correct_k, target, present_classes))
        return [r.item() * 100. for r in res]


def rank(output, target, present_classes):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        pred = output.argsort(dim=1, descending=True)

        rank = pred.eq(target.view(-1, 1).expand_as(pred)).nonzero(as_tuple=False)[:, 1].float()
        return compute_per_class_metric(rank + 1, target, present_classes).item()


def compute_per_class_metric(metric, target, present_classes):
    acc_per_class = 0.
    for i in present_classes:
        idx = (target == i)
        e = torch.true_divide(torch.sum(metric[idx]), torch.sum(idx))
        acc_per_class += e
    return acc_per_class / len(present_classes)
