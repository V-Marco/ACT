class TorchStandardScaler:
    def __init__(self):
        self._is_fit = False

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
        self._is_fit = True

    def transform(self, x):
        if self._is_fit:
            x -= self.mean
            x /= self.std + 1e-7
        return x


class TorchMinMaxColScaler:
    def __init__(self):
        self._is_fit = False

    def fit(self, x):
        self.min = x.min(dim=0)[0]
        self.max = x.max(dim=0)[0]
        self._is_fit = True

    def transform(self, x):
        if self._is_fit:
            x = (x - self.min + 1e-7) / (self.max + 1e-7 - self.min)
        return x


class TorchMinMaxScaler:
    def __init__(self):
        self._is_fit = False

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
        self._is_fit = True

    def transform(self, x):
        if self._is_fit:
            x = (x - self.min + 1e-7) / (self.max - self.min)
        return x