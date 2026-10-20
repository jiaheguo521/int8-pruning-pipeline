"""Recovering a pruned model: fine-tuning on labels, or distilling against the unpruned one.

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)   # after pruning
    history = int8_pruning.finetune(model, train_loader, 30, optimizer=opt)
    history = int8_pruning.distill(model, teacher, unlabelled_loader, 10, optimizer=opt)

Both train the model in place and return one `EpochResult` per epoch. Build the optimizer
after pruning: pruning replaces the parameter tensors.
"""
from int8_pruning.recovery._recovery import EpochResult, distill, finetune

__all__ = ["finetune", "distill", "EpochResult"]

# Report the public path, not the private module, in help() and tracebacks.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
