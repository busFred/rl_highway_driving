from typing import Sequence

import numpy as np
import torch

from .mdp_abc import State


def states_to_torch(states: Sequence[State], dtype: torch.dtype,
                    device: torch.device) -> torch.Tensor:
    states_np: np.ndarray = np.asarray(
        [state.get_np_state() for state in states])
    states_torch: torch.Tensor = torch.tensor(states_np,
                                              dtype=dtype,
                                              device=device)
    return states_torch
