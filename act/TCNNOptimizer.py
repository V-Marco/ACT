import numpy as np
import torch
import copy
import tqdm

from act.act_types import SimulationConfig
from act.models import (
    ConvolutionEmbeddingNet,
)
from act import utils
from act.ACTOptimizer import (
    ACTOptimizer
)

from act.TorchScalers import (
    TorchMinMaxScaler,
    TorchMinMaxColScaler
)



class TCNNOptimizer(ACTOptimizer):
    def __init__(
        self,
        simulation_config: SimulationConfig,
        logger: object = None,
        set_passive_properties = True
    ):
        super().__init__(
            simulation_config=simulation_config,
            logger=logger,
            set_passive_properties=set_passive_properties
        )

        self.model = None
        self.model_pool = None

        self.voltage_data_scaler = TorchMinMaxScaler()
        self.summary_feature_scaler = TorchMinMaxColScaler()

        self.segregation_index = utils.get_segregation_index(simulation_config)
        self.hto_block_channels = []

    def optimize(self, target_V: torch.Tensor) -> torch.Tensor:
        self.load_params()

        (
        simulated_V_for_next_stage,
        ampl_next_stage, 
        spiking_ind,
        nonsaturated_ind
        ) = self.load_voltage_traces(target_V)
            
        summary_features, summary_feature_columns, coefs_loaded = self.load_summary_features(simulated_V_for_next_stage,
        spiking_ind,
        nonsaturated_ind
        )

        # make amp output a learned parameter (target params)
        param_samples_for_next_stage = torch.cat(
            (param_samples_for_next_stage, ampl_next_stage.reshape((-1, 1))), dim=1
        )

        self.model = self.init_nn_model(
            in_channels=target_V.shape[1],
            out_channels=self.num_params + 1,  # +1 to learn amp input
            summary_features=summary_features,
            model_class=self.model_class,
        )

        # Resample to match the length of target data
        resampled_data = self.resample_voltage(
            simulated_V_for_next_stage, target_V.shape[1]
        )

        lows = [p["low"] for p in self.config["optimization_parameters"]["params"]]
        highs = [p["high"] for p in self.config["optimization_parameters"]["params"]]

        lows.append(round(float(ampl_next_stage.min()), 4))
        highs.append(round(float(ampl_next_stage.max()), 4))
        # remove any remaining nan values
        summary_features[torch.isnan(summary_features)] = 0

        # Train model
        train_stats = self.train_model(
            resampled_data.float(),
            param_samples_for_next_stage,
            lows,
            highs,
            train_test_split=self.train_test_split,
            summary_features=summary_features,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            num_epochs=self.num_epochs,
            summary_feature_columns=summary_feature_columns,
        )

        target_summary_features = self.extract_target_v_summary_features(target_V)

        predictions = self.predict_with_model(
            target_V.float(), lows, highs, target_summary_features.float()
        )
        # predictions = torch.max(predictions, dim=0).values

        return predictions, train_stats

    def init_nn_model(
        self, in_channels: int, out_channels: int, summary_features, model_class=None
    ) -> torch.nn.Sequential:
        if model_class:
            print(f"Overriding model class to {model_class}")
            ModelClass = eval(model_class)  # dangerous but ok
        else:
            print(f"Using ConvolutionEmbeddingNet for model class")
            # ModelClass = SimpleNet
            # ModelClass = BranchingNet
            # ModelClass = EmbeddingNet
            ModelClass = ConvolutionEmbeddingNet
            # ModelClass = SummaryNet
            # ModelClass = ConvolutionNet

        model = ModelClass(in_channels, out_channels, summary_features)
        return model

    def train_model(
        self,
        voltage_data: torch.Tensor,
        target_params: torch.Tensor,
        lows,
        highs,
        summary_features,
        summary_feature_columns=[],
        train_test_split=0.85,
        batch_size=8,
        learning_rate=2e-5,
        weight_decay=1e-4,
        num_epochs=0,
    ) -> None:
        if not learning_rate:
            learning_rate = 2e-5
        if not weight_decay:
            weight_decay = 1e-4

        sigmoid_mins = torch.tensor(lows)
        sigmoid_maxs = torch.tensor(highs)

        stats = {
            "train_loss_batches": [],
            "train_loss": [],
            "test_loss": [],
            "train_size": 0,
            "test_size": 0,
            "feature_importance": {},
        }

        # cut the target_params for segregation
        if self.config["run_mode"] == "segregated":
            if self.config["segregation"][self.segregation_index].get(
                "use_hto_amps", False
            ):
                self.hto_block_channels = self.config["optimization_parameters"].get(
                    "hto_block_channels", []
                )
            # get all the indicies that we want to keep
            keep_ind = []
            for i, param in enumerate(self.params):
                if (
                    param not in self.preset_params
                    and param not in self.hto_block_channels
                ):
                    keep_ind.append(i)
            print(f"Training target param indicies {keep_ind} only for segregation")
            keep_ind.append(-1)  # we want to also keep the last element for amps
            print(f"With amps {keep_ind}")
            target_params = target_params[:, keep_ind]
            sigmoid_mins = sigmoid_mins[keep_ind]
            sigmoid_maxs = sigmoid_maxs[keep_ind]

        # shuffle the training data
        indexes = torch.randperm(voltage_data.shape[0])
        split_point = int(voltage_data.shape[0] * train_test_split)

        train_ind = indexes[:split_point]
        test_ind = indexes[split_point:]
        stats["train_size"] = len(train_ind)
        stats["test_size"] = len(test_ind)

        voltage_data_train = voltage_data[train_ind]
        voltage_data_test = voltage_data[test_ind]

        summary_features_train = summary_features[train_ind]
        summary_features_test = summary_features[test_ind]

        target_params_train = target_params[train_ind]
        target_params_test = target_params[test_ind]

        # Fit the training data, transform both train and test.
        # The fit is not applied to original dataset due to the possibility of data leakage
        self.voltage_data_scaler.fit(voltage_data_train)
        self.summary_feature_scaler.fit(summary_features_train)

        voltage_data_train = self.voltage_data_scaler.transform(voltage_data_train)
        voltage_data_test = self.voltage_data_scaler.transform(voltage_data_test)
        summary_features_train = self.summary_feature_scaler.transform(
            summary_features_train
        )
        summary_features_test = self.summary_feature_scaler.transform(
            summary_features_test
        )
    
        optim = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_fn = torch.nn.MSELoss()  # torch.nn.functional.l1_loss

        self.logger.info(
            f"Training a model with {optim} optimizer | lr = {learning_rate} | weight_decay = {weight_decay}."
        )
        self.logger.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        batch_start = torch.arange(0, len(voltage_data_train), batch_size)

        # Hold the best model
        best_mse = np.inf  # init to infinity
        best_weights = None

        for epoch in range(num_epochs):
            self.model.train()
            with tqdm.tqdm(
                batch_start, unit="batch", mininterval=0, disable=False
            ) as bar:
                bar.set_description(f"Epoch {epoch}/{num_epochs}")
                for start in bar:
                    voltage_data_batch = voltage_data_train[
                        start : start + batch_size
                    ]
                    summary_features_batch = summary_features_train[
                        start : start + batch_size
                    ]
                    target_params_batch = target_params_train[
                        start : start + batch_size
                    ]
                    # forward pass
                    pred = (
                        self.model(voltage_data_batch, summary_features_batch)
                        * (sigmoid_maxs - sigmoid_mins)
                        + sigmoid_mins
                    )
                    loss = loss_fn(pred, target_params_batch)
                    stats["train_loss_batches"].append(
                        float(loss.cpu().detach().numpy())
                    )

                    # backward pass
                    optim.zero_grad()  # this line is new, wasn't in last round
                    loss.backward()

                    # update weights
                    optim.step()

                    # print process
                    bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            self.model.eval()
            y_out = self.model(voltage_data_train, summary_features_train)
            y_pred = y_out * (sigmoid_maxs - sigmoid_mins) + sigmoid_mins
            target_params_train_norm = (target_params_train - sigmoid_mins) / (
                sigmoid_maxs - sigmoid_mins
            )
            mse = loss_fn(y_pred, target_params_train)
            # mse = loss_fn(y_out, target_params_train_norm)
            mse = float(mse)
            stats["train_loss"].append(mse)

            y_out = self.model(voltage_data_test, summary_features_test)
            y_pred = y_out * (sigmoid_maxs - sigmoid_mins) + sigmoid_mins
            target_params_test_norm = (target_params_test - sigmoid_mins) / (
                sigmoid_maxs - sigmoid_mins
            )
            mse = loss_fn(y_pred, target_params_test)
            # mse = loss_fn(y_out, target_params_test_norm)
            mse = float(mse)
            stats["test_loss"].append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(self.model.state_dict())

        # restore model and return best accuracy
        self.model.load_state_dict(best_weights)

        return stats

    def predict_with_model(
        self, target_V: torch.Tensor, lows, highs, summary_features
    ) -> torch.Tensor:
        sigmoid_mins = torch.tensor(lows)
        sigmoid_maxs = torch.tensor(highs)

        if self.config["run_mode"] == "segregated":
            output_ind = []  # these are the indices that the network returned
            for i, param in enumerate(self.params):
                if (
                    param not in self.preset_params
                    and param not in self.hto_block_channels
                ):
                    output_ind.append(i)
            output_ind.append(-1)
            sigmoid_mins = sigmoid_mins[output_ind]
            sigmoid_maxs = sigmoid_maxs[output_ind]

        ret = None
        
        self.model.eval()
        outs = []
        target_V_fit = self.voltage_data_scaler.transform(target_V)
        summary_features_fit = self.summary_feature_scaler.transform(
            summary_features
        )
        for i in range(target_V.shape[0]):
            out = (
                self.model(
                    target_V_fit[i].reshape(1, -1),
                    summary_features_fit[i].reshape(1, -1),
                )
                * (sigmoid_maxs - sigmoid_mins)
                + sigmoid_mins
            )
            outs.append(out.reshape(1, -1))

        ret = torch.cat(outs, dim=0)

        # return with preset params
        if self.config["run_mode"] == "segregated":
            seg_ret = torch.zeros((ret.shape[0], len(self.params) + 1))
            seg_ret[:, output_ind] = ret
            for param_ind, param in enumerate(self.params):
                if param in self.preset_params:
                    seg_ret[:, param_ind] = self.preset_params[param]
            return seg_ret
        else:
            return ret