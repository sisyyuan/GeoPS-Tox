import torch



def get_device(number=None):
    if isinstance(number, torch.device):
        return number
    if number is None or str(number).lower() == "cpu":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{number}")
    return torch.device("cpu")


class PenaltyWeightScheduler:
    def __init__(self, epoch_to_max, init_val, max_val):
        assert epoch_to_max >= 0
        self.epoch_to_max = epoch_to_max
        self.init_val = init_val
        self.max_val = max_val
        self.step_val = (self.max_val - self.init_val) / self.epoch_to_max if self.epoch_to_max > 0 else 0

    def step(self, epoch):
        if epoch < 0: 
            return self.init_val
        elif epoch >= self.epoch_to_max:
            return self.max_val
        else:
            return self.init_val + self.step_val * epoch
