from typing import Any, Callable, Optional
from torch import nn
import torch

from ...mdps.change_lane import ChangeLaneEnv


class LinearDQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(
                ChangeLaneEnv.STATES_SHAPE[0] * ChangeLaneEnv.STATES_SHAPE[1],
                100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, ChangeLaneEnv.N_ACTIONS))

    def reset_parameters(self,
                         initializer: Optional[Callable[[torch.Tensor],
                                                        Any]] = None):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                if initializer is None:
                    m.reset_parameters()
                else:
                    initializer(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output
