from dataclasses import dataclass, field
from typing import List, Literal

import torch
from dataclasses_json import dataclass_json
from mdps import mdp_abc, mdp_utils
from torch import nn

from ...mdps.change_lane import ChangeLaneEnv


@dataclass_json
@dataclass
class LinearDQNConfig:
    out_features: List[int] = field()
    use_biases: List[bool] = field()
    use_batch_norms: List[bool] = field()
    use_activations: List[bool] = field()
    activation_type: Literal["relu", "tanh", "none"] = field()


class LinearDQN(nn.Module):
    model: nn.ModuleList

    def __init__(self, configs: LinearDQNConfig) -> None:
        super().__init__()
        model = nn.ModuleList([nn.Flatten()])
        for out_features, use_bias, use_batch_norm, use_activation in zip(
                configs.out_features, configs.use_biases,
                configs.use_batch_norms, configs.use_activations):
            # add linear layer
            model.append(nn.LazyLinear(out_features, use_bias))
            # add batch norm
            if use_batch_norm:
                model.append(nn.LazyBatchNorm1d())
            # add activation
            if configs.activation_type != "none" and use_activation:
                if configs.activation_type == "relu":
                    model.append(nn.ReLU())
                elif configs.activation_type == "tanh":
                    model.append(nn.Tanh())
        model.append(nn.LazyLinear(ChangeLaneEnv.N_ACTIONS))
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x
        for layer in self.model:
            output = layer(output)
        return output

    def init(self, env: mdp_abc.Environment):
        state = env.reset()
        dummy_input = mdp_utils.states_to_torch([state],
                                                dtype=torch.float,
                                                device=torch.device("cpu"))
        self.forward(dummy_input)
