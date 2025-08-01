import numpy as np
import torch

from act.act_types import SimulationConfig


from legacy.legacy_from_act import utils
from sklearn.ensemble import RandomForestRegressor
from proj.ACT.act.DataProcessor_old import DataProcessor
from act.ACTOptimizer import (
    ACTOptimizer
)

from act.TorchScalers import (
    TorchMinMaxScaler,
    TorchMinMaxColScaler
)


class RandomForestOptimizer(ACTOptimizer):
    def __init__(
        self,
        simulation_config: SimulationConfig,
        logger: object = None,
        set_passive_properties = True,
        n_estimators=5000,
        min_samples_split=2,
        max_depth=None,
        random_state=42
    ):
        super().__init__(
            simulation_config=simulation_config,
            logger=logger,
            set_passive_properties=set_passive_properties
        )

        self.model = None
        self.model_pool = None
        self.reg = RandomForestRegressor(
            n_estimators=n_estimators, 
            min_samples_split=min_samples_split, 
            max_depth=max_depth, 
            random_state=random_state
            )

        self.voltage_data_scaler = TorchMinMaxScaler()
        self.summary_feature_scaler = TorchMinMaxColScaler()

        self.segregation_index = utils.get_segregation_index(simulation_config)
        self.hto_block_channels = []


    def get_feature_importance(self, X_train, columns=[]):
        if not columns:
            columns = [f"feature_{i+1}" for i in range(X_train.shape[1])]
        f = dict(zip(columns, np.around(self.reg.feature_importances_ * 100, 2)))
        sf = {
            k: v for k, v in sorted(f.items(), key=lambda item: item[1], reverse=True)
        }
        for k, v in sf.items():
            print(k + " : " + str(v))
        return sf


    def optimize(self, target_V: torch.Tensor) -> torch.Tensor:

        # Load the data
        self.get_params()

        (
        simulated_V_for_next_stage,
        ampl_next_stage, 
        spiking_ind,
        nonsaturated_ind
        ) = self.get_voltage_traces(target_V)
            
        summary_features, summary_feature_columns, coefs_loaded = self.get_summary_features(simulated_V_for_next_stage, ampl_next_stage, spiking_ind,nonsaturated_ind)

        # make amp output a learned parameter (target params)
        param_samples_for_next_stage = torch.cat(
            (param_samples_for_next_stage, ampl_next_stage.reshape((-1, 1))), dim=1
        )

        # Resample to match the length of target data
        resampled_data = DataProcessor.resample_voltage(
            simulated_V_for_next_stage, target_V.shape[1]
        )

        lows = [p["low"] for p in self.config["optimization_parameters"]["params"]]
        highs = [p["high"] for p in self.config["optimization_parameters"]["params"]]

        lows.append(round(float(ampl_next_stage.min()), 4))
        highs.append(round(float(ampl_next_stage.max()), 4))
        # remove any remaining nan values
        summary_features[torch.isnan(summary_features)] = 0

        # Train model
        train_stats = self.fit(
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

        target_summary_features = self.get_summary_features(target_V, spiking_ind, nonsaturated_ind)

        predictions = self.predict(
            target_V.float(), lows, highs, target_summary_features.float()
        )
        # predictions = torch.max(predictions, dim=0).values

        return predictions, train_stats

    def fit(
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

        # Train the model
        self.reg.fit(summary_features_train.cpu().numpy(), target_params_train.cpu().numpy())

        # Evaluate the model
        stats["feature_importance"] = self.get_feature_importance(
            summary_features_train.cpu().numpy(),
            columns=summary_feature_columns
        )
        

        return stats

    def predict(
        self, summary_features, lows, highs
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

        ret = torch.tensor(
                self.reg.predict(summary_features.cpu().numpy())
            ).float()

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
        