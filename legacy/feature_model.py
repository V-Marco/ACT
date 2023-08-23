import numpy as np
import torch
from scipy import stats

# Information to calculate 1D CNN output size and maxpool output size.
# https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-cnn-26a14c2ea29
# Maxpool layer uses same resource but kernel size and stride are set to be the same unless otherwise specified.
# Padding set to 0 by default.


class DefaultSummaryModel(torch.nn.Module):
    def __init__(
        self,
        num_linear_layer_input_features: int,
        num_summary_features: int = 8,
        use_statistics: bool = True,
    ) -> None:
        """
        CNN to get summary features from current injections data.

        Parameters:
        ----------

        num_current_injections: int
            Number of current injections. Used as the number of input channels for the network.

        num_summary_features: int
            Number of summary features. Equal to the number of ouput dimensions of the network.

        hybrid: bool
            If true, concatenate the network's output with summary statistics of the input data (mean, variance and skew).
        """
        super().__init__()

        # Save for forward pass
        self.use_statistics = use_statistics
        self.num_linear_layer_input_features = num_linear_layer_input_features

        # Each layer is initialized individually to implement skip-connections
        # Dimensionalities a -> b below are arbitrary to keep track of changes
        self.conv1 = torch.nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=9, padding="same"
        )  # 1024 -> 1024
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)  # 1024 -> 512
        self.conv2 = torch.nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=7, padding="same"
        )  # 512 -> 512
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)  # 512 -> 256
        self.conv3 = torch.nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=5, padding="same"
        )  # 256 -> 256
        self.pool3 = torch.nn.MaxPool1d(kernel_size=2, stride=2)  # 256 -> 128
        self.conv4 = torch.nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=3, padding="same"
        )  # 128 -> 128
        self.pool4 = torch.nn.MaxPool1d(kernel_size=2, stride=2)  # 128 -> 64
        self.linear = torch.nn.Linear(
            in_features=num_linear_layer_input_features,
            out_features=num_summary_features,
        )

        print(
            f"Total number of summary features: {num_summary_features + use_statistics * 3}"
        )

    def forward(self, X):
        X = X.reshape((-1, 1, X.shape[0]))  # e.g., (batch_size, 1, 1024)

        out = self.pool1(torch.nn.functional.relu(self.conv1(X)))

        sk_con = self.pool2(out)
        out = self.pool2(torch.nn.functional.relu(self.conv2(out)))
        out = out + sk_con

        sk_con = self.pool3(out)
        out = self.pool3(torch.nn.functional.relu(self.conv3(out)))
        out = out + sk_con

        sk_con = self.pool4(out)
        out = self.pool4(torch.nn.functional.relu(self.conv4(out)))

        out = out.view(1, -1)
        out = self.linear(out)

        if self.use_statistics:
            out = torch.cat((out, self.compute_statistics(X)), 1)

        return out

    def compute_statistics(self, X: torch.tensor) -> torch.tensor:
        """
        Calculates mean, variance and skew of input data's columns.

        Parameters:
        ----------
        X: torch.tensor
            Data to compute on.

        Returns:
        ----------
        out: torch.tensor
            Tensor with summary statistics.
        """
        data = X.numpy()
        mean = np.mean(data, -1)
        variance = np.std(data, -1)
        skew = stats.skew(data, -1)

        return torch.from_numpy(np.concatenate((mean, variance, skew), -1))
