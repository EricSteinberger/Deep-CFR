import copy
import os
import pickle
from os.path import join as ospj

import psutil

from DeepCFR.IterationStrategy import IterationStrategy
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.StrategyBuffer import StrategyBuffer
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase as _ChiefBase
from PokerRL.util import file_util


class Chief(_ChiefBase):

    def __init__(self, t_prof):
        super().__init__(t_prof=t_prof)
        self._ps_handles = None
        self._la_handles = None
        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)

        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo
        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            self._strategy_buffers = [
                StrategyBuffer(t_prof=t_prof,
                               owner=p,
                               env_bldr=self._env_bldr,
                               max_size=None,
                               device=self._t_prof.device_inference)
                for p in range(t_prof.n_seats)
            ]

            if self._t_prof.log_verbose:
                self._exp_mem_usage = self.create_experiment(self._t_prof.name + " Chief_Memory_Usage")

    def set_la_handles(self, *la_handles):
        self._la_handles = list(la_handles)

    def set_ps_handle(self, *ps_handles):
        self._ps_handles = list(ps_handles)

    def update_alive_las(self, alive_la_handles):
        self._la_handles = alive_la_handles

    # ____________________________________________________ Strategy ____________________________________________________
    def pull_current_eval_strategy(self, last_iteration_receiver_has):
        """
        Args:
            last_iteration_receiver_has (list):     None or int for each player
        """
        d = {}
        last_cfr_iter_got_self = None

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            d[EvalAgentDeepCFR.EVAL_MODE_AVRG_NET] = self._pull_avrg_net_eval_strat()

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            d[EvalAgentDeepCFR.EVAL_MODE_SINGLE], last_cfr_iter_got_self = self._pull_single_eval_strat(
                last_iteration_receiver_has)

        return d, last_cfr_iter_got_self

    def _pull_avrg_net_eval_strat(self):
        return [
            self._ray.get(self._ray.remote(ps.get_avrg_weights))
            for ps in self._ps_handles
        ]

    def _pull_single_eval_strat(self, last_iteration_receiver_has):
        """
        Args:
            last_iteration_receiver_has (list):     None or int for each player
        """
        buf_sizes = [
            self._strategy_buffers[_p_id].size
            for _p_id in range(self._t_prof.n_seats)
        ]

        _first_cfr_iter_to_get = [
            0 if last_iteration_receiver_has[_p_id] is None else last_iteration_receiver_has[_p_id]
            for _p_id in range(self._t_prof.n_seats)
        ]

        def _to_torch(cum_strat_state_dict):
            cum_strat_state_dict["net"] = self._ray.state_dict_to_numpy(cum_strat_state_dict["net"])
            return cum_strat_state_dict

        state_dicts = [
            [
                _to_torch(self._strategy_buffers[_p_id].get(i).state_dict())
                for i in range(_first_cfr_iter_to_get[_p_id], self._strategy_buffers[_p_id].size)
            ]
            for _p_id in range(self._t_prof.n_seats)
        ]

        return state_dicts, \
               buf_sizes

    # Only applicable to SINGLE
    def add_new_iteration_strategy_model(self, owner, adv_net_state_dict, cfr_iter):
        iter_strat = IterationStrategy(t_prof=self._t_prof, env_bldr=self._env_bldr, owner=owner,
                                         device=self._t_prof.device_inference, cfr_iter=cfr_iter)

        iter_strat.load_net_state_dict(
            self._ray.state_dict_to_torch(adv_net_state_dict, device=self._t_prof.device_inference))
        self._strategy_buffers[iter_strat.owner].add(iteration_strat=iter_strat)

        #  Store to disk
        if self._t_prof.export_each_net:
            path = ospj(self._t_prof.path_strategy_nets, self._t_prof.name)
            file_util.create_dir_if_not_exist(path)
            file_util.do_pickle(obj=iter_strat.state_dict(),
                                path=path,
                                file_name=str(iter_strat.cfr_iteration) + "_P" + str(iter_strat.owner) + ".pkl"
                                )

        if self._t_prof.log_verbose:
            if owner == 1:
                # Logs
                process = psutil.Process(os.getpid())
                self.add_scalar(self._exp_mem_usage, "Debug/Memory Usage/Chief", cfr_iter, process.memory_info().rss)

    # ________________________________ Store a pickled API class to play against the AI ________________________________
    def export_agent(self, step):
        _dir = ospj(self._t_prof.path_agent_export_storage, str(self._t_prof.name), str(step))
        file_util.create_dir_if_not_exist(_dir)

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            MODE = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET

            t_prof = copy.deepcopy(self._t_prof)
            t_prof.eval_modes_of_algo = [MODE]

            eval_agent = EvalAgentDeepCFR(t_prof=t_prof)
            eval_agent.reset()

            w = {EvalAgentDeepCFR.EVAL_MODE_AVRG_NET: self._pull_avrg_net_eval_strat()}
            eval_agent.update_weights(w)
            eval_agent.set_mode(mode=MODE)
            eval_agent.store_to_disk(path=_dir, file_name="eval_agent" + MODE)

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            MODE = EvalAgentDeepCFR.EVAL_MODE_SINGLE
            t_prof = copy.deepcopy(self._t_prof)
            t_prof.eval_modes_of_algo = [MODE]

            eval_agent = EvalAgentDeepCFR(t_prof=t_prof)
            eval_agent.reset()

            eval_agent._strategy_buffers = self._strategy_buffers  # could copy - it's just for the export, so it's ok
            eval_agent.set_mode(mode=MODE)
            eval_agent.store_to_disk(path=_dir, file_name="eval_agent" + MODE)

    # __________________________________________________ Checkpointing _________________________________________________
    def checkpoint(self, curr_step):
        if self._SINGLE:
            for p_id in range(self._t_prof.n_seats):
                state = {
                    "strat_buffer": self._strategy_buffers[p_id].state_dict(),
                }

                with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                         cls=self.__class__, worker_id="P" + str(p_id)),
                          "wb") as pkl_file:
                    pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, name_to_load, step):
        if self._SINGLE:
            for p_id in range(self._t_prof.n_seats):
                with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                         cls=self.__class__, worker_id="P" + str(p_id)),
                          "rb") as pkl_file:
                    state = pickle.load(pkl_file)
                    self._strategy_buffers[p_id].load_state_dict(state["strat_buffer"])
