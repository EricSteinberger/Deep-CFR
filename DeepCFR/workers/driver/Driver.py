from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.driver._HighLevelAlgo import HighLevelAlgo
from PokerRL.rl.base_cls.workers.DriverBase import DriverBase


class Driver(DriverBase):

    def __init__(self, t_prof, eval_methods, n_iterations=None, iteration_to_import=None, name_to_import=None):
        if t_prof.DISTRIBUTED:
            from DeepCFR.workers.chief.dist import Chief
            from DeepCFR.workers.la.dist import LearnerActor
            from DeepCFR.workers.ps.dist import ParameterServer

        else:
            from DeepCFR.workers.chief.local import Chief
            from DeepCFR.workers.la.local import LearnerActor
            from DeepCFR.workers.ps.local import ParameterServer

        super().__init__(t_prof=t_prof, eval_methods=eval_methods, n_iterations=n_iterations,
                         iteration_to_import=iteration_to_import, name_to_import=name_to_import,
                         chief_cls=Chief, eval_agent_cls=EvalAgentDeepCFR)

        if "h2h" in list(eval_methods.keys()):
            assert EvalAgentDeepCFR.EVAL_MODE_SINGLE in t_prof.eval_modes_of_algo
            assert EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in t_prof.eval_modes_of_algo
            self._ray.remote(self.eval_masters["h2h"][0].set_modes,
                             [EvalAgentDeepCFR.EVAL_MODE_SINGLE, EvalAgentDeepCFR.EVAL_MODE_AVRG_NET]
                             )

        print("Creating LAs...")
        self.la_handles = [
            self._ray.create_worker(LearnerActor,
                                    t_prof,
                                    i,
                                    self.chief_handle)
            for i in range(t_prof.n_learner_actors)
        ]

        print("Creating Parameter Servers...")
        self.ps_handles = [
            self._ray.create_worker(ParameterServer,
                                    t_prof,
                                    p,
                                    self.chief_handle)
            for p in range(t_prof.n_seats)
        ]

        self._ray.wait([
            self._ray.remote(self.chief_handle.set_ps_handle,
                             *self.ps_handles),
            self._ray.remote(self.chief_handle.set_la_handles,
                             *self.la_handles)
        ])

        print("Created and initialized Workers")

        self.algo = HighLevelAlgo(t_prof=t_prof,
                                  la_handles=self.la_handles,
                                  ps_handles=self.ps_handles,
                                  chief_handle=self.chief_handle)

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        self._maybe_load_checkpoint_init()

    def run(self):
        print("Setting stuff up...")

        # """"""""""""""""
        # Init globally
        # """"""""""""""""
        self.algo.init()

        print("Starting Training...")
        for _iter_nr in range(10000000 if self.n_iterations is None else self.n_iterations):
            print("Iteration: ", self._cfr_iter)

            # """"""""""""""""
            # Maybe train AVRG
            # """"""""""""""""
            avrg_times = None
            if self._AVRG and self._any_eval_needs_avrg_net():
                avrg_times = self.algo.train_average_nets(cfr_iter=_iter_nr)

            # """"""""""""""""
            # Eval
            # """"""""""""""""
            # Evaluate. Sync & Lock, then train while evaluating on other workers
            self.evaluate()

            # """"""""""""""""
            # Log
            # """"""""""""""""
            if self._cfr_iter % self._t_prof.log_export_freq == 0:
                self.save_logs()
            self.periodically_export_eval_agent()

            # """"""""""""""""
            # Iteration
            # """"""""""""""""
            iter_times = self.algo.run_one_iter_alternating_update(cfr_iter=self._cfr_iter)

            print(
                "Generating Data: ", str(iter_times["t_generating_data"]) + "s.",
                "  ||  Trained ADV", str(iter_times["t_computation_adv"]) + "s.",
                "  ||  Synced ADV", str(iter_times["t_syncing_adv"]) + "s.",
                "\n"
            )
            if self._AVRG and avrg_times:
                print(
                    "Trained AVRG", str(avrg_times["t_computation_avrg"]) + "s.",
                    "  ||  Synced AVRG", str(avrg_times["t_syncing_avrg"]) + "s.",
                    "\n"
                )

            self._cfr_iter += 1

            # """"""""""""""""
            # Checkpoint
            # """"""""""""""""
            self.periodically_checkpoint()

    def _any_eval_needs_avrg_net(self):
        for e in list(self.eval_masters.values()):
            if self._cfr_iter % e[1] == 0:
                return True
        return False

    def checkpoint(self, **kwargs):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]:
            self._ray.wait([
                self._ray.remote(w.checkpoint,
                                 self._cfr_iter)
            ])

        # Delete past checkpoints
        s = [self._cfr_iter]
        if self._cfr_iter > self._t_prof.checkpoint_freq + 1:
            s.append(self._cfr_iter - self._t_prof.checkpoint_freq)

        self._delete_past_checkpoints(steps_not_to_delete=s)

    def load_checkpoint(self, step, name_to_load):
        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles + [self.chief_handle]:
            self._ray.wait([
                self._ray.remote(w.load_checkpoint,
                                 name_to_load, step)
            ])
