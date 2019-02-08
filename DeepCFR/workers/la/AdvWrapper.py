import torch

from PokerRL.rl.neural.DuelingQNet import DuelingQNet
from PokerRL.rl.neural.NetWrapperBase import NetWrapperArgsBase as _NetWrapperArgsBase
from PokerRL.rl.neural.NetWrapperBase import NetWrapperBase as _NetWrapperBase


class AdvWrapper(_NetWrapperBase):

    def __init__(self, env_bldr, adv_training_args, owner, device):
        super().__init__(
            net=DuelingQNet(env_bldr=env_bldr, q_args=adv_training_args.adv_net_args, device=device),
            env_bldr=env_bldr,
            args=adv_training_args,
            owner=owner,
            device=device
        )

    def get_advantages(self, pub_obses, range_idxs, legal_action_mask):
        self._net.eval()
        with torch.no_grad():
            return self._net(pub_obses=pub_obses, range_idxs=range_idxs, legal_action_masks=legal_action_mask)

    def _mini_batch_loop(self, buffer, grad_mngr):
        batch_pub_obs, \
        batch_range_idxs, \
        batch_legal_action_masks, \
        batch_adv, \
        batch_loss_weight, \
            = buffer.sample(device=self.device, batch_size=self._args.batch_size)

        # [batch_size, n_actions]
        adv_pred = self._net(pub_obses=batch_pub_obs,
                             range_idxs=batch_range_idxs,
                             legal_action_masks=batch_legal_action_masks)

        grad_mngr.backprop(pred=adv_pred, target=batch_adv,
                           loss_weights=batch_loss_weight.unsqueeze(-1).expand_as(batch_adv))


class AdvTrainingArgs(_NetWrapperArgsBase):

    def __init__(self,
                 adv_net_args,
                 n_batches_adv_training=1000,
                 batch_size=4096,
                 n_mini_batches_per_update=1,
                 optim_str="adam",
                 loss_str="weighted_mse",
                 lr=0.001,
                 grad_norm_clipping=10.0,
                 device_training="cpu",
                 max_buffer_size=2e6,
                 lr_patience=100,
                 init_adv_model="last",
                 ):
        super().__init__(batch_size=batch_size, n_mini_batches_per_update=n_mini_batches_per_update,
                         optim_str=optim_str, loss_str=loss_str, lr=lr, grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)
        self.adv_net_args = adv_net_args
        self.n_batches_adv_training = n_batches_adv_training
        self.lr_patience = lr_patience
        self.max_buffer_size = int(max_buffer_size)
        self.init_adv_model = init_adv_model
