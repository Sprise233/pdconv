from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class PolyLRSchedulerWithWarmup(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, warmup_steps: int = 0, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps  # 新增 warmup 步数
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def get_lr(self, current_step):
        if current_step < self.warmup_steps:
            # 在 warmup 阶段，学习率线性增加
            return self.initial_lr * (current_step / self.warmup_steps)
        else:
            # 在 warmup 之后，使用多项式衰减
            return self.initial_lr * (1 - (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)) ** self.exponent

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.get_lr(current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
