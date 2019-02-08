import os
import pickle

import psutil
from torch.optim import lr_scheduler

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.ParameterServerBase import ParameterServerBase
from PokerRL.rl.neural.AvrgStrategyNet import AvrgStrategyNet
from PokerRL.rl.neural.DuelingQNet import DuelingQNet


class ParameterServer(ParameterServerBase):

    def __init__(self, t_prof, owner, chief_handle):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle)

        self.owner = owner
        self._adv_args = t_prof.module_args["adv_training"]

        self._adv_net = self._get_new_adv_net()
        self._adv_optim, self._adv_lr_scheduler = self._get_new_adv_optim()

        if self._t_prof.log_verbose:
            self._exp_mem_usage = self._ray.get(
                self._ray.remote(self._chief_handle.create_experiment,
                                 self._t_prof.name + "_PS" + str(owner) + "_Memory_Usage"))

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]
            self._avrg_net = self._get_new_avrg_net()
            self._avrg_optim, self._avrg_lr_scheduler = self._get_new_avrg_optim()

    # ______________________________________________ API to pull from PS _______________________________________________

    def get_adv_weights(self):
        self._adv_net.zero_grad()
        return self._ray.state_dict_to_numpy(self._adv_net.state_dict())

    def get_avrg_weights(self):
        self._avrg_net.zero_grad()
        return self._ray.state_dict_to_numpy(self._avrg_net.state_dict())

    # ____________________________________________ API to make PS compute ______________________________________________
    def apply_grads_adv(self, list_of_grads):
        self._apply_grads(list_of_grads=list_of_grads, optimizer=self._adv_optim, net=self._adv_net,
                          grad_norm_clip=self._adv_args.grad_norm_clipping)

    def apply_grads_avrg(self, list_of_grads):
        self._apply_grads(list_of_grads=list_of_grads, optimizer=self._avrg_optim, net=self._avrg_net,
                          grad_norm_clip=self._avrg_args.grad_norm_clipping)

    def reset_adv_net(self, cfr_iter):
        if self._adv_args.init_adv_model == "last":
            self._adv_net.zero_grad()
            if not self._t_prof.online:
                self._adv_optim, self._adv_lr_scheduler = self._get_new_adv_optim()
        elif self._adv_args.init_adv_model == "random":
            self._adv_net = self._get_new_adv_net()
            self._adv_optim, self._adv_lr_scheduler = self._get_new_adv_optim()
        else:
            raise ValueError(self._adv_args.init_adv_model)

        if self._t_prof.log_verbose and (cfr_iter % 3 == 0):
            # Logs
            process = psutil.Process(os.getpid())
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_mem_usage, "Debug/MemoryUsage/PS", cfr_iter,
                             process.memory_info().rss)

    def reset_avrg_net(self):
        if self._avrg_args.init_avrg_model == "last":
            self._avrg_net.zero_grad()
            if not self._t_prof.online:
                self._avrg_optim, self._avrg_lr_scheduler = self._get_new_avrg_optim()

        elif self._avrg_args.init_avrg_model == "random":
            self._avrg_net = self._get_new_avrg_net()
            self._avrg_optim, self._avrg_lr_scheduler = self._get_new_avrg_optim()

        else:
            raise ValueError(self._avrg_args.init_avrg_model)

    def step_scheduler_adv(self, loss):
        self._adv_lr_scheduler.step(loss)

    def step_scheduler_avrg(self, loss):
        self._avrg_lr_scheduler.step(loss)

    # ______________________________________________ API for checkpointing _____________________________________________
    def checkpoint(self, curr_step):
        state = {
            "adv_net": self._adv_net.state_dict(),
            "adv_optim": self._adv_optim.state_dict(),
            "adv_lr_sched": self._adv_lr_scheduler.state_dict(),
            "seat_id": self.owner,
        }
        if self._AVRG:
            state["avrg_net"] = self._avrg_net.state_dict()
            state["avrg_optim"] = self._avrg_optim.state_dict()
            state["avrg_lr_sched"] = self._avrg_lr_scheduler.state_dict()

        with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                 cls=self.__class__, worker_id="P" + str(self.owner)),
                  "wb") as pkl_file:
            pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, name_to_load, step):
        with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                 cls=self.__class__, worker_id="P" + str(self.owner)),
                  "rb") as pkl_file:
            state = pickle.load(pkl_file)

            assert self.owner == state["seat_id"]

        self._adv_net.load_state_dict(state["adv_net"])
        self._adv_optim.load_state_dict(state["adv_optim"])
        self._adv_lr_scheduler.load_state_dict(state["adv_lr_sched"])
        if self._AVRG:
            self._avrg_net.load_state_dict(state["avrg_net"])
            self._avrg_optim.load_state_dict(state["avrg_optim"])
            self._avrg_lr_scheduler.load_state_dict(state["avrg_lr_sched"])

    # __________________________________________________________________________________________________________________
    def _get_new_adv_net(self):
        return DuelingQNet(q_args=self._adv_args.adv_net_args, env_bldr=self._env_bldr, device=self._device)

    def _get_new_avrg_net(self):
        return AvrgStrategyNet(avrg_net_args=self._avrg_args.avrg_net_args, env_bldr=self._env_bldr,
                               device=self._device)

    def _get_new_adv_optim(self):
        opt = rl_util.str_to_optim_cls(self._adv_args.optim_str)(self._adv_net.parameters(), lr=self._adv_args.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=opt,
                                                   threshold=0.001,
                                                   factor=0.5,
                                                   patience=self._adv_args.lr_patience,
                                                   min_lr=0.00002)
        return opt, scheduler

    def _get_new_avrg_optim(self):
        opt = rl_util.str_to_optim_cls(self._avrg_args.optim_str)(self._avrg_net.parameters(), lr=self._avrg_args.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=opt,
                                                   threshold=0.0001,
                                                   factor=0.5,
                                                   patience=self._avrg_args.lr_patience,
                                                   min_lr=0.00002)
        return opt, scheduler
