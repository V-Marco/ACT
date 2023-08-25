import torch

# We can potentially add CNN layers before Linear layers


# @CHECK
class BranchingNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, summary_features):
        super().__init__()
        self.summary_features = summary_features
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

    def forward(self, X):
        voltage_out = self.voltage_branch(X)
        summary_out = self.summary_branch(self.summary_features)
        return self.sigmoid(voltage_out + summary_out)


# @CHECK
class EmbeddingNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, summary_features):
        super().__init__()
        self.summary_features = summary_features
        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(64 + summary_features.shape[-1], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.Sigmoid(),
        )

    def forward(self, X):
        embedding = self.embedder(X)
        return self.predictor(torch.cat(embedding, self.summary_features))
