from torch import nn
import torch

from ..mdps.highway.change_lane import ChangeLaneEnv


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output
