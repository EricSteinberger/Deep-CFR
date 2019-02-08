import os
import pickle

import psutil

from DeepCFR.IterationStrategy import IterationStrategy
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.la.buffers.AdvReservoirBuffer import AdvReservoirBuffer
from DeepCFR.workers.la.AdvWrapper import AdvWrapper
from DeepCFR.workers.la.buffers.AvrgReservoirBuffer import AvrgReservoirBuffer
from DeepCFR.workers.la.AvrgWrapper import AvrgWrapper
from DeepCFR.workers.la.sampling_algorithms.MultiOutcomeSampler import MultiOutcomeSampler
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase


class LearnerActor(WorkerBase):

    def __init__(self, t_prof, worker_id, chief_handle):
        super().__init__(t_prof=t_prof)

        self._adv_args = t_prof.module_args["adv_training"]

        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)
        self._id = worker_id
        self._chief_handle = chief_handle

        self._adv_buffers = [
            AdvReservoirBuffer(owner=p, env_bldr=self._env_bldr, max_size=self._adv_args.max_buffer_size,
                               nn_type=t_prof.nn_type,
                               iter_weighting_exponent=self._t_prof.iter_weighting_exponent)
            for p in range(self._t_prof.n_seats)
        ]

        self._adv_wrappers = [
            AdvWrapper(owner=p,
                       env_bldr=self._env_bldr,
                       adv_training_args=self._adv_args,
                       device=self._adv_args.device_training)
            for p in range(self._t_prof.n_seats)
        ]

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]

            self._avrg_buffers = [
                AvrgReservoirBuffer(owner=p, env_bldr=self._env_bldr, max_size=self._avrg_args.max_buffer_size,
                                    nn_type=t_prof.nn_type,
                                    iter_weighting_exponent=self._t_prof.iter_weighting_exponent)
                for p in range(self._t_prof.n_seats)
            ]

            self._avrg_wrappers = [
                AvrgWrapper(owner=p,
                            env_bldr=self._env_bldr,
                            avrg_training_args=self._avrg_args,
                            device=self._avrg_args.device_training)
                for p in range(self._t_prof.n_seats)
            ]

            if self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=self._avrg_buffers,
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples)
            else:
                raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")
        else:
            if self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=None,
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples)
            else:
                raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")

        if self._t_prof.log_verbose:
            self._exp_mem_usage = self._ray.get(
                self._ray.remote(self._chief_handle.create_experiment,
                                 self._t_prof.name + "_LA" + str(worker_id) + "_Memory_Usage"))
            self._exps_adv_buffer_size = self._ray.get(
                [
                    self._ray.remote(self._chief_handle.create_experiment,
                                     self._t_prof.name + "_LA" + str(worker_id) + "_P" + str(p) + "_ADV_BufSize")
                    for p in range(self._t_prof.n_seats)
                ]
            )
            if self._AVRG:
                self._exps_avrg_buffer_size = self._ray.get(
                    [
                        self._ray.remote(self._chief_handle.create_experiment,
                                         self._t_prof.name + "_LA" + str(worker_id) + "_P" + str(p) + "_AVRG_BufSize")
                        for p in range(self._t_prof.n_seats)
                    ]
                )

    def generate_data(self, traverser, cfr_iter):
        iteration_strats = [
            IterationStrategy(t_prof=self._t_prof, env_bldr=self._env_bldr, owner=p,
                              device=self._t_prof.device_inference, cfr_iter=cfr_iter)
            for p in range(self._t_prof.n_seats)
        ]
        for s in iteration_strats:
            s.load_net_state_dict(state_dict=self._adv_wrappers[s.owner].net_state_dict())

        self._data_sampler.generate(n_traversals=self._t_prof.n_traversals_per_iter,
                                    traverser=traverser,
                                    iteration_strats=iteration_strats,
                                    cfr_iter=cfr_iter,
                                    )

        # Log after both players generated data
        if self._t_prof.log_verbose and traverser == 1 and (cfr_iter % 3 == 0):
            for p in range(self._t_prof.n_seats):
                self._ray.remote(self._chief_handle.add_scalar,
                                 self._exps_adv_buffer_size[p], "Debug/BufferSize", cfr_iter,
                                 self._adv_buffers[p].size)
                if self._AVRG:
                    self._ray.remote(self._chief_handle.add_scalar,
                                     self._exps_avrg_buffer_size[p], "Debug/BufferSize", cfr_iter,
                                     self._avrg_buffers[p].size)

            process = psutil.Process(os.getpid())
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_mem_usage, "Debug/MemoryUsage/LA", cfr_iter,
                             process.memory_info().rss)

    def update(self, adv_state_dicts=None, avrg_state_dicts=None):
        """
        Args:
            adv_state_dicts (list):         Optional. if not None:
                                                        expects a list of neural net state dicts or None for each player
                                                        in order of their seat_ids. This allows updating only some
                                                        players.

            avrg_state_dicts (list):         Optional. if not None:
                                                        expects a list of neural net state dicts or None for each player
                                                        in order of their seat_ids. This allows updating only some
                                                        players.
        """
        for p_id in range(self._t_prof.n_seats):
            if adv_state_dicts[p_id] is not None:
                self._adv_wrappers[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(self._ray.get(adv_state_dicts[p_id]),
                                                             device=self._adv_wrappers[p_id].device))

            if avrg_state_dicts[p_id] is not None:
                self._avrg_wrappers[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(self._ray.get(avrg_state_dicts[p_id]),
                                                             device=self._avrg_wrappers[p_id].device))

    def get_loss_last_batch_adv(self, p_id):
        return self._adv_wrappers[p_id].loss_last_batch

    def get_loss_last_batch_avrg(self, p_id):
        return self._avrg_wrappers[p_id].loss_last_batch

    def get_adv_grads(self, p_id):
        return self._ray.grads_to_numpy(
            self._adv_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._adv_buffers[p_id]))

    def get_avrg_grads(self, p_id):
        return self._ray.grads_to_numpy(
            self._avrg_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._avrg_buffers[p_id]))

    def checkpoint(self, curr_step):
        for p_id in range(self._env_bldr.N_SEATS):
            state = {
                "adv_buffer": self._adv_buffers[p_id].state_dict(),
                "adv_wrappers": self._adv_wrappers[p_id].state_dict(),
                "p_id": p_id,
            }
            if self._AVRG:
                state["avrg_buffer"] = self._avrg_buffers[p_id].state_dict()
                state["avrg_wrappers"] = self._avrg_wrappers[p_id].state_dict()

            with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                     cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id)),
                      "wb") as pkl_file:
                pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, name_to_load, step):
        for p_id in range(self._env_bldr.N_SEATS):
            with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                     cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id)),
                      "rb") as pkl_file:
                state = pickle.load(pkl_file)

                assert state["p_id"] == p_id

                self._adv_buffers[p_id].load_state_dict(state["adv_buffer"])
                self._adv_wrappers[p_id].load_state_dict(state["adv_wrappers"])
                if self._AVRG:
                    self._avrg_buffers[p_id].load_state_dict(state["avrg_buffer"])
                    self._avrg_wrappers[p_id].load_state_dict(state["avrg_wrappers"])
