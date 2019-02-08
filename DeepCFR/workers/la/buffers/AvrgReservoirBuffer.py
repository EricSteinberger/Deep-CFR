import torch

from DeepCFR.workers.la.buffers._ReservoirBufferBase import ReservoirBufferBase as _ResBufBase
from PokerRL.rl import rl_util


class AvrgReservoirBuffer(_ResBufBase):
    """
    Reservoir buffer to store state+action samples for the average strategy network
    """

    def __init__(self, owner, nn_type, max_size, env_bldr, iter_weighting_exponent):
        super().__init__(owner=owner, max_size=max_size, env_bldr=env_bldr, nn_type=nn_type,
                         iter_weighting_exponent=iter_weighting_exponent)

        self._a_probs_buffer = torch.zeros((max_size, env_bldr.N_ACTIONS), dtype=torch.float32, device=self.device)

    def add(self, pub_obs, range_idx, legal_actions_list, a_probs, iteration):
        if self.size < self._max_size:
            self._add(idx=self.size,
                      pub_obs=pub_obs,
                      range_idx=range_idx,
                      legal_action_mask=self._get_mask(legal_actions_list),
                      action_probs=a_probs,
                      iteration=iteration)
            self.size += 1

        elif self._should_add():
            self._add(idx=self._random_idx(),
                      pub_obs=pub_obs,
                      range_idx=range_idx,
                      legal_action_mask=self._get_mask(legal_actions_list),
                      action_probs=a_probs,
                      iteration=iteration)

        self.n_entries_seen += 1

    def sample(self, batch_size, device):
        indices = torch.randint(0, self.size, (batch_size,), dtype=torch.long, device=self.device)

        if self._nn_type == "recurrent":
            obses = self._pub_obs_buffer[indices.cpu().numpy()]
        elif self._nn_type == "feedforward":
            obses = self._pub_obs_buffer[indices].to(device)
        else:
            raise NotImplementedError

        return \
            obses, \
            self._range_idx_buffer[indices].to(device), \
            self._legal_action_mask_buffer[indices].to(device), \
            self._a_probs_buffer[indices].to(device), \
            self._iteration_buffer[indices].to(device) / self._last_cfr_iteration_seen

    def _add(self, idx, pub_obs, range_idx, legal_action_mask, action_probs, iteration):
        if self._nn_type == "feedforward":
            pub_obs = torch.from_numpy(pub_obs)

        self._pub_obs_buffer[idx] = pub_obs
        self._range_idx_buffer[idx] = range_idx
        self._legal_action_mask_buffer[idx] = legal_action_mask
        self._a_probs_buffer[idx] = action_probs

        # In "https://arxiv.org/pdf/1811.00164.pdf", Brown et al. weight by floor((t+1)/2), but we assume that
        # this is due to incrementation happening for every alternating update. We count one iteration as an
        # update for both plyrs.
        self._iteration_buffer[idx] = float(iteration) ** self._iter_weighting_exponent
        self._last_cfr_iteration_seen = iteration

    def _get_mask(self, legal_actions_list):
        return rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                   legal_actions_list=legal_actions_list,
                                                   device=self.device, dtype=torch.float32)

    def state_dict(self):
        return {
            "base": super().state_dict(),
            "a_probs": self._a_probs_buffer,
        }

    def load_state_dict(self, state):
        super().load_state_dict(state["base"])
        self._a_probs_buffer = state["a_probs"]
