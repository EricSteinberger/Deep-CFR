import torch
import torch.nn.functional as nnf

from PokerRL.rl import rl_util
from PokerRL.rl.neural.AvrgStrategyNet import AvrgStrategyNet
from PokerRL.rl.neural.NetWrapperBase import NetWrapperArgsBase as _NetWrapperArgsBase
from PokerRL.rl.neural.NetWrapperBase import NetWrapperBase as _NetWrapperBase


class AvrgWrapper(_NetWrapperBase):

    def __init__(self, owner, env_bldr, avrg_training_args, device):
        super().__init__(
            net=AvrgStrategyNet(avrg_net_args=avrg_training_args.avrg_net_args, env_bldr=env_bldr, device=device),
            env_bldr=env_bldr,
            args=avrg_training_args,
            owner=owner,
            device=device
        )
        self._all_range_idxs = torch.arange(self._env_bldr.rules.RANGE_SIZE, device=self.device, dtype=torch.long)

    def get_a_probs(self, pub_obses, range_idxs, legal_actions_lists):
        """
        Args:
            pub_obses (list):             list of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (np.ndarray):    array of range_idxs (one for each pub_obs) tensor([2, 421, 58, 912, ...])
            legal_actions_lists (list:  list of lists. each 2nd level lists contains ints representing legal actions
        """
        with torch.no_grad():
            masks = rl_util.batch_get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                              legal_actions_lists=legal_actions_lists,
                                                              device=self.device)
            masks = masks.view(1, -1)
            return self.get_a_probs2(pub_obses=pub_obses, range_idxs=range_idxs, legal_action_masks=masks)

    def get_a_probs2(self, pub_obses, range_idxs, legal_action_masks):
        with torch.no_grad():
            pred = self._net(pub_obses=pub_obses,
                             range_idxs=torch.from_numpy(range_idxs).to(dtype=torch.long, device=self.device),
                             legal_action_masks=legal_action_masks)

            return nnf.softmax(pred, dim=-1).cpu().numpy()

    def get_a_probs_for_each_hand(self, pub_obs, legal_actions_list):
        with torch.no_grad():
            mask = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                       legal_actions_list=legal_actions_list,
                                                       device=self.device, dtype=torch.uint8)
            mask = mask.unsqueeze(0).expand(self._env_bldr.rules.RANGE_SIZE, -1)

            pred = self._net(pub_obses=[pub_obs] * self._env_bldr.rules.RANGE_SIZE,
                             range_idxs=self._all_range_idxs,
                             legal_action_masks=mask)

            return nnf.softmax(pred, dim=1).cpu().numpy()

    def _mini_batch_loop(self, buffer, grad_mngr):
        batch_pub_obs, \
        batch_range_idxs, \
        batch_legal_action_masks, \
        batch_a_probs, \
        batch_loss_weight, \
            = buffer.sample(device=self.device, batch_size=self._args.batch_size)

        # [batch_size, n_actions]
        strat_pred = self._net(pub_obses=batch_pub_obs,
                               range_idxs=batch_range_idxs,
                               legal_action_masks=batch_legal_action_masks)
        strat_pred = nnf.softmax(strat_pred, dim=-1)
        grad_mngr.backprop(pred=strat_pred, target=batch_a_probs,
                           loss_weights=batch_loss_weight.unsqueeze(-1).expand_as(batch_a_probs))


class AvrgTrainingArgs(_NetWrapperArgsBase):

    def __init__(self,
                 avrg_net_args,
                 n_batches_avrg_training=1000,
                 batch_size=4096,
                 n_mini_batches_per_update=1,
                 optim_str="adam",
                 loss_str="weighted_mse",
                 lr=0.001,
                 grad_norm_clipping=10.0,
                 device_training="cpu",
                 max_buffer_size=2e6,
                 lr_patience=100,
                 init_avrg_model="random",
                 ):
        super().__init__(batch_size=batch_size,
                         n_mini_batches_per_update=n_mini_batches_per_update,
                         optim_str=optim_str,
                         loss_str=loss_str,
                         lr=lr,
                         grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)

        self.avrg_net_args = avrg_net_args
        self.n_batches_avrg_training = n_batches_avrg_training
        self.max_buffer_size = int(max_buffer_size)
        self.lr_patience = lr_patience
        self.init_avrg_model = init_avrg_model
