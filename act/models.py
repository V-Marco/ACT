import torch

"""
Set of extra models that could be used in ACT
"""

class SimpleNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, summary_features, seed):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, out_channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, X, summary_features):
        return self.model.forward(X)


class SimpleSummaryNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, summary_features):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(summary_features.shape[-1], 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, out_channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, X, summary_features):
        return self.model.forward(summary_features)


# @CHECK
class BranchingNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, summary_features):
        super().__init__()
        self.voltage_branch = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, out_channels),
        )
        self.summary_branch = torch.nn.Sequential(
            torch.nn.Linear(summary_features.shape[-1], out_channels)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X, summary_features):
        voltage_out = self.voltage_branch(X)
        summary_out = self.summary_branch(summary_features)
        return self.sigmoid(voltage_out + summary_out)


class EmbeddingNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, summary_features):
        super().__init__()
        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(64 + summary_features.shape[-1], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, out_channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, X, summary_features):
        embedding = self.embedder(X)
        return self.predictor(torch.cat((embedding, summary_features)))


class ConvolutionEmbeddingNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, summary_features):
        super().__init__()
        self.embedder = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=5, padding="same"
            ),
            torch.nn.Conv1d(
                in_channels=8, out_channels=8, kernel_size=5, padding="same"
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Conv1d(
                in_channels=8, out_channels=8, kernel_size=5, padding="same"
            ),
            torch.nn.Conv1d(
                in_channels=8, out_channels=1, kernel_size=5, padding="same"
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_channels, 64),
            torch.nn.ReLU(),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(64 + summary_features.shape[-1], 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, out_channels),
            torch.nn.Tanh(),
        )

    def forward(self, X, summary_features):
        # Potentially support batches
        if len(X.shape) == 1:
            X_res = X.reshape(1, 1, X.shape[0])
        if len(X.shape) == 2:
            X_res = X.reshape(X.shape[0], 1, X.shape[1])

        # The embedder's output is (1, ...), flatten for concatenation
        embedding = self.embedder(X_res)
        return self.predictor(torch.cat((summary_features, embedding), axis=1))


class ConvolutionNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, summary_features):
        super().__init__()
        self.embedder = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=5, padding="same"
            ),
            torch.nn.Conv1d(
                in_channels=8, out_channels=8, kernel_size=5, padding="same"
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Conv1d(
                in_channels=8, out_channels=8, kernel_size=5, padding="same"
            ),
            torch.nn.Conv1d(
                in_channels=8, out_channels=1, kernel_size=5, padding="same"
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.Tanh(),
        )

    def forward(self, X, summary_features):
        # Potentially support batches
        if len(X.shape) == 1:
            X_res = X.reshape(1, 1, X.shape[0])
        if len(X.shape) == 2:
            X_res = X.reshape(X.shape[0], 1, X.shape[1])

        # The embedder's output is (1, ...), flatten for concatenation
        return self.embedder(X_res)


class SummaryNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, summary_features):
        super().__init__()
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(summary_features.shape[-1], 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, out_channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, X, summary_features):
        return self.predictor(summary_features)
