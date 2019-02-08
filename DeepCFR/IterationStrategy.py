# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch
from torch.nn import functional as F

from PokerRL.rl import rl_util
from PokerRL.rl.neural.DuelingQNet import DuelingQNet


class IterationStrategy:

    def __init__(self, t_prof, owner, env_bldr, device, cfr_iter):
        self._t_prof = t_prof
        self._owner = owner
        self._env_bldr = env_bldr
        self._device = device
        self._cfr_iter = cfr_iter

        self._adv_net = None
        self._all_range_idxs = torch.arange(self._env_bldr.rules.RANGE_SIZE, device=self._device, dtype=torch.long)

    @property
    def owner(self):
        return self._owner

    @property
    def cfr_iteration(self):
        return self._cfr_iter

    @property
    def device(self):
        return self._device

    def reset(self):
        self._adv_net = None

    def get_action(self, pub_obses, range_idxs, legal_actions_lists):
        a_probs = self.get_a_probs(pub_obses=pub_obses, range_idxs=range_idxs,
                                   legal_actions_lists=legal_actions_lists)

        return torch.multinomial(torch.from_numpy(a_probs), num_samples=1).cpu().numpy()

    def get_a_probs2(self, pub_obses, range_idxs, legal_action_masks, to_np=True):
        """
        Args:
            pub_obses (list):               batch (list) of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (list):              batch (list) of range_idxs (one for each pub_obs) [2, 421, 58, 912, ...]
            legal_action_masks (Torch.tensor)
        """

        with torch.no_grad():
            bs = len(range_idxs)

            if self._adv_net is None:  # at iteration 0
                uniform_even_legal = legal_action_masks / (legal_action_masks.sum(-1)
                                                           .unsqueeze(-1)
                                                           .expand_as(legal_action_masks))
                if to_np:
                    return uniform_even_legal.cpu().numpy()
                return uniform_even_legal
            else:
                range_idxs = torch.tensor(range_idxs, dtype=torch.long, device=self._device)

                advantages = self._adv_net(pub_obses=pub_obses,
                                           range_idxs=range_idxs,
                                           legal_action_masks=legal_action_masks)

                # """"""""""""""""""""
                relu_advantages = F.relu(advantages, inplace=False)  # Cause the sum of *positive* regret matters in CFR
                sum_pos_adv_expanded = relu_advantages.sum(1).unsqueeze(-1).expand_as(relu_advantages)

                # """"""""""""""""""""
                # In case all negative
                # """"""""""""""""""""
                best_legal_deterministic = torch.zeros((bs, self._env_bldr.N_ACTIONS,), dtype=torch.float32,
                                                       device=self._device)
                bests = torch.argmax(
                    torch.where(legal_action_masks.byte(), advantages, torch.full_like(advantages, fill_value=-10e20))
                    , dim=1
                )
                _batch_arranged = torch.arange(bs, device=self._device, dtype=torch.long)
                best_legal_deterministic[_batch_arranged, bests] = 1

                # """"""""""""""""""""
                # Strat
                # """"""""""""""""""""
                strategy = torch.where(
                    sum_pos_adv_expanded > 0,
                    relu_advantages / sum_pos_adv_expanded,
                    best_legal_deterministic
                )

                if to_np:
                    strategy = strategy.cpu().numpy()
                return strategy

    def get_a_probs(self, pub_obses, range_idxs, legal_actions_lists, to_np=True):
        """
        Args:
            pub_obses (list):               batch (list) of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (list):              batch (list) of range_idxs (one for each pub_obs) [2, 421, 58, 912, ...]
            legal_actions_lists (list):     batch (list) of lists of integers that represent legal actions
        """

        with torch.no_grad():
            masks = rl_util.batch_get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                              legal_actions_lists=legal_actions_lists,
                                                              device=self._device, dtype=torch.float32)
            return self.get_a_probs2(pub_obses=pub_obses,
                                     range_idxs=range_idxs,
                                     legal_action_masks=masks,
                                     to_np=to_np)

    def get_a_probs_for_each_hand(self, pub_obs, legal_actions_list):
        """
        Args:
            pub_obs (np.array(shape=(seq_len, n_features,)))
            legal_actions_list (list):      list of ints representing legal actions
        """

        if self._t_prof.DEBUGGING:
            assert isinstance(pub_obs, np.ndarray)
            assert len(pub_obs.shape) == 2, "all hands have the same public obs"
            assert isinstance(legal_actions_list[0],
                              int), "all hands do the same actions. no need to batch, just parse int"

        return self._get_a_probs_of_hands(pub_obs=pub_obs, legal_actions_list=legal_actions_list,
                                          range_idxs_tensor=self._all_range_idxs)

    def get_a_probs_for_each_hand_in_list(self, pub_obs, range_idxs, legal_actions_list):
        """
        Args:
            pub_obs (np.array(shape=(seq_len, n_features,)))
            range_idxs (np.ndarray):        list of range_idxs to evaluate in public state ""pub_obs""
            legal_actions_list (list):      list of ints representing legal actions
        """

        if self._t_prof.DEBUGGING:
            assert isinstance(pub_obs, np.ndarray)
            assert isinstance(range_idxs, np.ndarray)
            assert len(pub_obs.shape) == 2, "all hands have the same public obs"
            assert isinstance(legal_actions_list[0], int), "all hands can do the same actions. no need to batch"

        return self._get_a_probs_of_hands(pub_obs=pub_obs, legal_actions_list=legal_actions_list,
                                          range_idxs_tensor=torch.from_numpy(range_idxs).to(dtype=torch.long,
                                                                                            device=self._device))

    def _get_a_probs_of_hands(self, pub_obs, range_idxs_tensor, legal_actions_list):
        with torch.no_grad():
            n_hands = range_idxs_tensor.size(0)

            if self._adv_net is None:  # at iteration 0
                uniform_even_legal = torch.zeros((self._env_bldr.N_ACTIONS,), dtype=torch.float32, device=self._device)
                uniform_even_legal[legal_actions_list] = 1.0 / len(legal_actions_list)  # always >0
                uniform_even_legal = uniform_even_legal.unsqueeze(0).expand(n_hands, self._env_bldr.N_ACTIONS)
                return uniform_even_legal.cpu().numpy()

            else:
                legal_action_masks = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                                         legal_actions_list=legal_actions_list,
                                                                         device=self._device, dtype=torch.float32)
                legal_action_masks = legal_action_masks.unsqueeze(0).expand(n_hands, -1)

                advantages = self._adv_net(pub_obses=[pub_obs] * n_hands,
                                           range_idxs=range_idxs_tensor,
                                           legal_action_masks=legal_action_masks)

                # """"""""""""""""""""
                relu_advantages = F.relu(advantages, inplace=False)  # Cause the sum of *positive* regret matters in CFR
                sum_pos_adv_expanded = relu_advantages.sum(1).unsqueeze(-1).expand_as(relu_advantages)

                # """"""""""""""""""""
                # In case all negative
                # """"""""""""""""""""
                best_legal_deterministic = torch.zeros((n_hands, self._env_bldr.N_ACTIONS,), dtype=torch.float32,
                                                       device=self._device)
                bests = torch.argmax(
                    torch.where(legal_action_masks.byte(), advantages, torch.full_like(advantages, fill_value=-10e20)),
                    dim=1
                )

                _batch_arranged = torch.arange(n_hands, device=self._device, dtype=torch.long)
                best_legal_deterministic[_batch_arranged, bests] = 1

                # """"""""""""""""""""
                # Strategy
                # """"""""""""""""""""
                strategy = torch.where(
                    sum_pos_adv_expanded > 0,
                    relu_advantages / sum_pos_adv_expanded,
                    best_legal_deterministic,
                )

                return strategy.cpu().numpy()

    def state_dict(self):
        return {
            "owner": self._owner,
            "net": self.net_state_dict(),
            "iter": self._cfr_iter,
        }

    @staticmethod
    def build_from_state_dict(t_prof, env_bldr, device, state):
        s = IterationStrategy(t_prof=t_prof, env_bldr=env_bldr, device=device,
                              owner=state["owner"], cfr_iter=state["iter"])
        s.load_state_dict(state=state)  # loads net state
        return s

    def load_state_dict(self, state):
        assert self._owner == state["owner"]
        self.load_net_state_dict(state["net"])
        self._cfr_iter = state["iter"]

    def net_state_dict(self):
        """ This just wraps the net.state_dict() with the option of returning None if net is None """
        if self._adv_net is None:
            return None
        return self._adv_net.state_dict()

    def load_net_state_dict(self, state_dict):
        if state_dict is None:
            return  # if this happens (should only for iteration 0), this class will return random actions.
        else:
            self._adv_net = DuelingQNet(q_args=self._t_prof.module_args["adv_training"].adv_net_args,
                                        env_bldr=self._env_bldr, device=self._device)
            self._adv_net.load_state_dict(state_dict)
            self._adv_net.to(self._device)

        self._adv_net.eval()
        for param in self._adv_net.parameters():
            param.requires_grad = False

    def get_copy(self, device=None):
        _device = self._device if device is None else device
        return IterationStrategy.build_from_state_dict(t_prof=self._t_prof, env_bldr=self._env_bldr,
                                                       device=_device, state=self.state_dict())
