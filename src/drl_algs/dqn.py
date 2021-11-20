import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from dataclasses_json import dataclass_json
from drl_utils import buff_utils
from drl_utils.buff_utils import RLDataset
from mdps import mdp_utils
from mdps.mdp_abc import (Action, DiscreteAction, DiscreteEnvironment, Metrics,
                          PolicyBase, State)
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


@dataclass_json
@dataclass
class DQNConfig:
    discount: float = field(default=1.0)
    epsilon: float = field(default=0.5)
    epsilon_decay: float = field(default=0.99)
    epsilon_update_episodes: int = field(default=20)
    n_episodes: int = field(default=1000)
    max_buff_size: int = field(default=10000)
    batch_size: int = field(default=100)
    targ_update_episodes: int = field(default=20)


class DQN(PolicyBase):

    ActionType: Type[DiscreteAction]

    _dqn: nn.Module
    _dtype: torch.dtype
    _device: torch.device

    @property
    def dqn(self) -> nn.Module:
        """Get the dqn.

        Returns:
            dqn (nn.Module): The self.dqn.
        """
        self._dqn.to(device=self._device, dtype=self._dtype)
        return self._dqn

    def __init__(
        self,
        env: DiscreteEnvironment,
        dqn_net: nn.Module,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.ActionType = env.ActionType
        self._dqn = dqn_net
        self._dtype = dtype
        self._device = device
        self._dqn.eval()

    def sample_action(self, state: State) -> DiscreteAction:
        next_q_vals: torch.Tensor = self.predict_q_vals([state])
        action = self.ActionType(next_q_vals.argmax(1)[0].item())
        return action

    def predict_q_vals(self, states: Sequence[State]) -> torch.Tensor:
        """Predict and select the q_vals predictions made by self.dqn.

        Args:
            states (Sequence[State]): (n_states, ) States to be predicted.

        Returns:
            q_vals (torch.Tensor): (n_states, n_actions) The predicted q values.
        """
        states_tensor: torch.Tensor = mdp_utils.states_to_torch(
            states=states, dtype=self._dtype, device=self._device)
        return self._predict_q_vals(states=states_tensor)

    def _predict_q_vals(self, states: torch.Tensor) -> torch.Tensor:
        """Predict and select the q_vals predictions made by self.dqn.

        Args:
            states (torch.Tensor): (n_states, n_state_features) States to be predicted.

        Returns:
            q_vals (torch.Tensor): (n_states, n_actions) The predicted q values.
        """
        states = states.to(device=self._device, dtype=self._dtype)
        # (n_states, n_actions)
        q_vals: torch.Tensor = self.dqn(states)
        return q_vals


class DQNTrain(DQN, pl.LightningModule):

    env: DiscreteEnvironment
    dqn_config: DQNConfig
    optimizer: torch.optim.Optimizer
    _targ_dqn: nn.Module

    # initialized in on_fit_start
    curr_epsilon: float
    buff: RLDataset
    curr_state: State
    is_terminal: bool
    max_episode_steps: int
    n_val_episodes: int

    @property
    def targ_dqn(self) -> nn.Module:
        """Get the targ_dqn.

        Raises:
            ValueError: if self.targ_dqn is None.

        Returns:
            targ_dqn (nn.Module): The target self.dqn.
        """
        self._targ_dqn.to(device=self._device, dtype=self._dtype)
        return self._targ_dqn

    def __init__(
        self,
        env: DiscreteEnvironment,
        dqn_net: nn.Module,
        dqn_config: DQNConfig,
        optimizer: torch.optim.Optimizer,
        max_episode_steps: int = 30,
        n_val_episodes: int = 10,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__(env=env, dqn_net=dqn_net, dtype=dtype, device=device)
        self.env = env
        self.dqn_config = dqn_config
        self.optimizer = optimizer
        self._targ_dqn = copy.deepcopy(self._dqn)
        self.max_episode_steps = max_episode_steps
        self.n_val_episodes = n_val_episodes

    # pl method override

    def configure_optimizers(self):
        return self.optimizer

    # initialize varaibles for dqn
    # train
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.buff, batch_size=self.dqn_config.batch_size)

    def on_fit_start(self) -> None:
        self.curr_epsilon = self.dqn_config.epsilon
        # initialize replay buffer with random policy
        self.buff = RLDataset(max_size=self.dqn_config.max_buff_size,
                              batch_size=self.dqn_config.batch_size,
                              dtype=self._dtype,
                              device=self._device)
        buff_utils.populate_replay_buffer(
            buff=self.buff,
            env=self.env,
            policy=self.env.get_random_policy(),
            target_size=self.dqn_config.batch_size)
        self.curr_state = self.env.reset()
        self.is_terminal = False
        return super().on_fit_start()

    # on episode start
    def on_epoch_start(self) -> None:
        # reset the enviornment on each epoch
        self._curr_state = self.env.reset()
        self.is_terminal = False

    # on step start
    def on_train_batch_start(self,
                             batch: Any,
                             batch_idx: int,
                             unused: Optional[int] = 0) -> Union[int, None]:
        if batch_idx >= self.max_episode_steps or self.is_terminal:
            return -1
        return None

    # deep q step
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        states, actions, next_states, next_rewards, is_terminals = batch
        self.eval()
        action, next_state, next_reward, is_terminal = self._eps_greedy_step()
        self.buff.add_experience(state=self.curr_state,
                                 action=action,
                                 next_state=next_state,
                                 next_reward=next_reward,
                                 is_terminal=is_terminal)
        self.train()
        # dqn.optimizer.zero_grad()
        target_q_vals: torch.Tensor = self._compute_target(
            next_states=next_states,
            next_rewards=next_rewards,
            is_terminals=is_terminals)
        # (n_states, n_actions)
        pred_q_vals: torch.Tensor = self._predict_q_vals(states=states)
        # (n_states, 1)
        pred_q_vals = pred_q_vals.gather(1, actions.type(dtype=torch.int64))
        error: torch.Tensor = nn.SmoothL1Loss()(pred_q_vals, target_q_vals)
        self.curr_state = next_state
        self.is_terminal = is_terminal
        return error

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % self.dqn_config.epsilon_update_episodes == 0:
            self.curr_epsilon = self.curr_epsilon * self.dqn_config.epsilon_decay
        if self.current_epoch % self.dqn_config.targ_update_episodes == 0:
            self._update_targ()
        return super().on_train_epoch_end()

    # val
    def val_dataloader(self) -> DataLoader:
        # just a dummy
        return DataLoader(dataset=TensorDataset(
            torch.tensor(np.array([self.n_val_episodes]))),
                          batch_size=1)

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx):
        metrics: Sequence[Metrics] = mdp_utils.simulate_episodes(
            self.env,
            policy=self,
            max_episode_steps=self.max_episode_steps,
            n_episodes=self.n_val_episodes,
            to_visualize=False)
        return metrics

    def validation_epoch_end(self, outputs: Sequence[Sequence[Metrics]]):
        metrics_seq: Sequence[Metrics] = outputs[0]
        metrics_dict = self.env.summarize_metrics_seq(metrics_seq)
        self.log_dict(metrics_dict, prog_bar=True)

    def train(self, mode: bool = True) -> "DQNTrain":
        if mode == True:
            if self._targ_dqn is None:
                self._update_targ()
            self.dqn.train()
        else:
            self.dqn.eval()
        return self

    def eval(self) -> "DQNTrain":
        """Set the dqn to eval mode.
        """
        self.dqn.eval()
        return self

    # dqn specific method

    @torch.no_grad()
    def _compute_target(self, next_states: torch.Tensor,
                        next_rewards: torch.Tensor,
                        is_terminals: torch.Tensor) -> torch.Tensor:
        """Compute target q values given a tensor representation for states.

        Args:
            states (torch.Tensor): (n_states, n_state_features)
            next_rewards (torch.Tensor): (n_states, 1)
            is_terminals (torch.Tensor): (n_states, )
            dqn_config (DQNConfig): dqn hyperparameter

        Returns:
            target_q_vals (torch.Tensor): (n_states, 1) The target value.
        """
        next_states.to(dtype=self._dtype, device=self._device)
        # (n_states, n_actions)
        next_q_vals: torch.Tensor = self.targ_dqn(next_states)
        next_q_vals = next_q_vals.detach()
        # select next action greedily
        # (n_states, 1)
        next_q_vals = next_q_vals.max(1, keepdim=True)[0]
        # current terminal states has no future reward
        next_q_vals[is_terminals] = 0.0
        # (n_states, 1)
        target_q_vals: torch.Tensor = next_rewards + self.dqn_config.discount * next_q_vals
        return target_q_vals

    @torch.no_grad()
    def _eps_greedy_step(self) -> Tuple[Action, State, float, bool]:
        """Take a step based on epsilon greedy policy.

        Args:
            env (DiscreteEnvironment): A discrete enviornment.
            state (State): The current state of the enviornment.
            dqn (DQN): The dqn.
            dqn_config (DQNConfig): The hyperparameters for the dqn.

        Returns:
            action (DiscreteAction): The current action.
            next_state (State): The next state.
            next_reward (float): The next reward.
            is_terminal (bool): Whether next state is terminal.
        """
        is_random: bool = np.random.uniform(0, 1) < self.dqn_config.epsilon
        policy = self
        if is_random:
            # ignore type error
            policy = self.env.get_random_policy()
        # (1, n_actions)
        action = policy.sample_action(self.curr_state)
        next_state, next_reward, is_terminal = self.env.step(
            action, to_visualize=False)
        return action, next_state, next_reward, is_terminal

    def _update_targ(self):
        """Update target network.
        """
        self.targ_dqn.load_state_dict(self.dqn.state_dict())
        self.targ_dqn.eval()
        for param in self.targ_dqn.parameters():
            param.requires_grad = False
