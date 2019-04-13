# Copyright (c) 2019 Eric Steinberger


import copy

import numpy as np
from PokerRL.game import Poker
from PokerRL.game._.tree._.nodes import PlayerActionNode
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.EvalAgentBase import EvalAgentBase as _EvalAgentBase
from PokerRL.rl.errors import UnknownModeError

from DeepCFR.IterationStrategy import IterationStrategy
from DeepCFR.StrategyBuffer import StrategyBuffer
from DeepCFR.workers.la.AvrgWrapper import AvrgWrapper

NP_FLOAT_TYPE = np.float64  # Use 64 for extra stability in big games


class EvalAgentDeepCFR(_EvalAgentBase):
    EVAL_MODE_AVRG_NET = "AVRG_NET"
    EVAL_MODE_SINGLE = "SINGLE"
    ALL_MODES = [EVAL_MODE_AVRG_NET, EVAL_MODE_SINGLE]

    def __init__(self, t_prof, mode=None, device=None):
        super().__init__(t_prof=t_prof, mode=mode, device=device)
        self.avrg_args = t_prof.module_args["avrg_training"]

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self.t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self.t_prof.eval_modes_of_algo

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            self.avrg_net_policies = [
                AvrgWrapper(avrg_training_args=self.avrg_args, owner=p, env_bldr=self.env_bldr, device=self.device)
                for p in range(t_prof.n_seats)
            ]
            for pol in self.avrg_net_policies:
                pol.eval()

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            self._strategy_buffers = [
                StrategyBuffer(t_prof=t_prof,
                               owner=p,
                               env_bldr=self.env_bldr,
                               max_size=self.t_prof.eval_agent_max_strat_buf_size,
                               device=self.device)
                for p in range(t_prof.n_seats)
            ]

            # Iteration whose strategies is used for current episode for each seat.
            # Only applicable if trajectory-sampling SD-CFR is used.
            self._episode_net_idxs = [
                None
                for p in range(self.env_bldr.N_SEATS)
            ]

            # Track history
            self._a_history = None

    def can_compute_mode(self):
        """ All modes are always computable (i.e. not dependent on iteration etc.)"""
        return True

    # ___________________________ Overrides to track history for reach computation in SD-CFR ___________________________
    def notify_of_reset(self):
        if self._mode == self.EVAL_MODE_SINGLE:
            self._reset_action_history()
            self._sample_new_strategy()
        super().notify_of_reset()

    def reset(self, deck_state_dict=None):
        if self._mode == self.EVAL_MODE_SINGLE:
            self._reset_action_history()
            self._sample_new_strategy()
        super().reset(deck_state_dict=deck_state_dict)

    def set_to_public_tree_node_state(self, node):
        if self._mode == self.EVAL_MODE_SINGLE:
            # """""""""""""""""""""""""""""
            # Set history to correct state
            # """""""""""""""""""""""""""""
            self._reset_action_history()
            relevant_nodes_in_forward_order = []
            _node = node
            while _node is not None:
                if isinstance(_node, PlayerActionNode) and _node.p_id_acted_last == node.p_id_acting_next:
                    relevant_nodes_in_forward_order.insert(0, _node)
                _node = _node.parent

            for _node in relevant_nodes_in_forward_order:
                super().set_to_public_tree_node_state(node=_node.parent)
                self._add_history_entry(p_id_acting=_node.p_id_acted_last, action_hes_gonna_do=_node.action)

        # """""""""""""""""""""""""""""
        # Set env wrapper to correct
        # state.
        # """""""""""""""""""""""""""""
        super().set_to_public_tree_node_state(node=node)

    def get_a_probs_for_each_hand(self):
        """ BEFORE CALLING, NOTIFY EVALAGENT OF THE PAST ACTIONS / ACTIONSEQUENCE!!!!! """
        pub_obs = self._internal_env_wrapper.get_current_obs()
        legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._mode == self.EVAL_MODE_AVRG_NET:
            return self.avrg_net_policies[p_id_acting].get_a_probs_for_each_hand(pub_obs=pub_obs,
                                                                                 legal_actions_list=legal_actions_list)

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        elif self._mode == self.EVAL_MODE_SINGLE:

            unif_rand_legal = np.full(
                shape=self.env_bldr.N_ACTIONS,
                fill_value=1.0 / len(legal_actions_list)
            ) * rl_util.get_legal_action_mask_np(n_actions=self.env_bldr.N_ACTIONS,
                                                 legal_actions_list=legal_actions_list,
                                                 dtype=np.float32)

            n_models = self._strategy_buffers[p_id_acting].size
            if n_models == 0:
                return np.repeat(np.expand_dims(unif_rand_legal, axis=0),
                                 repeats=self.env_bldr.rules.RANGE_SIZE, axis=0)
            else:
                # Dim: [model_idx, range_idx]
                reaches = self._get_reach_for_each_model_each_hand(p_id_acting=p_id_acting)

                # """"""""""""""""""""""
                # Compute strategy for
                # all infosets with
                # reach >0. Initialize
                # All others stay unif.
                # """"""""""""""""""""""
                contrib_each_model = np.zeros(
                    shape=(n_models, self.env_bldr.rules.RANGE_SIZE, self.env_bldr.N_ACTIONS),
                    dtype=NP_FLOAT_TYPE
                )

                for m_i, (strat, weight) in enumerate(self._strategy_buffers[p_id_acting].get_strats_and_weights()):
                    range_idxs = np.nonzero(reaches[m_i])[0]
                    if range_idxs.shape[0] > 0:
                        a_probs_m = strat.get_a_probs_for_each_hand_in_list(
                            pub_obs=pub_obs,
                            range_idxs=range_idxs,
                            legal_actions_list=legal_actions_list
                        )
                        contrib_each_model[m_i, range_idxs] = a_probs_m * weight

                # Dim: [range_idx, action_p]
                a_probs = (np.sum(contrib_each_model * np.expand_dims(reaches, axis=2), axis=0)).astype(NP_FLOAT_TYPE)

                # Dim: [range_idx]
                a_probs_sum = np.expand_dims(np.sum(a_probs, axis=1), axis=1)

                # Dim: [range_idx, action_p]
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.where(a_probs_sum == 0,
                                    np.repeat(np.expand_dims(unif_rand_legal, axis=0),
                                              repeats=self._internal_env_wrapper.env.RANGE_SIZE, axis=0),
                                    a_probs / a_probs_sum
                                    )

        else:
            raise UnknownModeError(self._mode)

    def get_a_probs(self):
        pub_obs = self._internal_env_wrapper.get_current_obs()
        legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        range_idx = self._internal_env_wrapper.env.get_range_idx(p_id=p_id_acting)

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._mode == self.EVAL_MODE_AVRG_NET:
            return self.avrg_net_policies[p_id_acting].get_a_probs(
                pub_obses=[pub_obs],
                range_idxs=np.array([range_idx], dtype=np.int32),
                legal_actions_lists=[legal_actions_list]
            )[0]

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        elif self._mode == self.EVAL_MODE_SINGLE:

            if self._strategy_buffers[p_id_acting].size == 0:
                unif_rand_legal = np.full(
                    shape=self.env_bldr.N_ACTIONS,
                    fill_value=1.0 / len(legal_actions_list)
                ) * rl_util.get_legal_action_mask_np(n_actions=self.env_bldr.N_ACTIONS,
                                                     legal_actions_list=legal_actions_list,
                                                     dtype=np.float32)
                return unif_rand_legal
            else:
                # """""""""""""""""""""
                # Weighted by Iteration
                # """"""""""""""""""""""
                # Dim: [model_idx, action_p]
                a_probs_each_model = np.array([
                    weight * strat.get_a_probs(pub_obses=[pub_obs],
                                               range_idxs=[range_idx],
                                               legal_actions_lists=[legal_actions_list]
                                               )[0]
                    for strat, weight in self._strategy_buffers[p_id_acting].get_strats_and_weights()
                ])

                # """"""""""""""""""""""
                # Weighted by Reach
                # """"""""""""""""""""""
                a_probs_each_model *= np.expand_dims(self._get_reach_for_each_model(
                    p_id_acting=p_id_acting,
                    range_idx=range_idx,
                ), axis=2)

                # """"""""""""""""""""""
                # Normalize
                # """"""""""""""""""""""
                # Dim: [action_p]
                a_probs = np.sum(a_probs_each_model, axis=0)

                # Dim: []
                a_probs_sum = np.sum(a_probs)

                # Dim: [action_p]
                return a_probs / a_probs_sum

        else:
            raise UnknownModeError(self._mode)

    def get_action(self, step_env=True, need_probs=False):
        """ !! BEFORE CALLING, NOTIFY EVALAGENT OF THE PAST ACTIONS / ACTIONSEQUENCE !! """

        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        range_idx = self._internal_env_wrapper.env.get_range_idx(p_id=p_id_acting)

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._mode == self.EVAL_MODE_AVRG_NET:
            if need_probs:  # only do if necessary
                a_probs_all_hands = self.get_a_probs_for_each_hand()
                a_probs = a_probs_all_hands[range_idx]
            else:
                a_probs_all_hands = None  # not needed

                a_probs = self.avrg_net_policies[p_id_acting].get_a_probs(
                    pub_obses=[self._internal_env_wrapper.get_current_obs()],
                    range_idxs=np.array([range_idx], dtype=np.int32),
                    legal_actions_lists=[self._internal_env_wrapper.env.get_legal_actions()]
                )[0]

            action = np.random.choice(np.arange(self.env_bldr.N_ACTIONS), p=a_probs)

            if step_env:
                self._internal_env_wrapper.step(action=action)

            return action, a_probs_all_hands

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        elif self._mode == self.EVAL_MODE_SINGLE:
            if need_probs:
                a_probs_all_hands = self.get_a_probs_for_each_hand()
            else:
                a_probs_all_hands = None  # not needed

            legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()

            if self._episode_net_idxs[p_id_acting] is None:  # Iteration 0
                action = legal_actions_list[np.random.randint(len(legal_actions_list))]
            else:  # Iteration > 0
                action = self._strategy_buffers[p_id_acting].get(self._episode_net_idxs[p_id_acting]).get_action(
                    pub_obses=[self._internal_env_wrapper.get_current_obs()],
                    range_idxs=[range_idx],
                    legal_actions_lists=[legal_actions_list],
                )[0].item()

            if step_env:
                # add to history before modifying env state
                self._add_history_entry(p_id_acting=p_id_acting, action_hes_gonna_do=action)

                # make INTERNAL step to keep up with the game state.
                self._internal_env_wrapper.step(action=action)

            return action, a_probs_all_hands
        else:
            raise UnknownModeError(self._mode)

    def get_action_frac_tuple(self, step_env):
        a_idx_raw = self.get_action(step_env=step_env, need_probs=False)[0]

        if self.env_bldr.env_cls.IS_FIXED_LIMIT_GAME:
            return a_idx_raw, -1
        else:
            if a_idx_raw >= 2:
                frac = self.env_bldr.env_args.bet_sizes_list_as_frac_of_pot[a_idx_raw - 2]
                return [Poker.BET_RAISE, frac]
            return [a_idx_raw, -1]

    def update_weights(self, weights_for_eval_agent):

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            avrg_weights = weights_for_eval_agent[self.EVAL_MODE_AVRG_NET]

            for p in range(self.t_prof.n_seats):
                self.avrg_net_policies[p].load_net_state_dict(self.ray.state_dict_to_torch(avrg_weights[p],
                                                                                           device=self.device))
                self.avrg_net_policies[p].eval()

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            list_of_new_iter_strat_state_dicts = copy.deepcopy(weights_for_eval_agent[self.EVAL_MODE_SINGLE])

            for p in range(self.t_prof.n_seats):
                for state in list_of_new_iter_strat_state_dicts[p]:
                    state["net"] = self.ray.state_dict_to_torch(state["net"], device=self.device)

                    _iter_strat = IterationStrategy.build_from_state_dict(state=state, t_prof=self.t_prof,
                                                                          env_bldr=self.env_bldr,
                                                                          device=self.device)

                    self._strategy_buffers[p].add(iteration_strat=_iter_strat)

    def _state_dict(self):
        d = {}

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            d["avrg_nets"] = [pol.net_state_dict() for pol in self.avrg_net_policies]

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            d["strategy_buffers"] = [self._strategy_buffers[p].state_dict() for p in range(self.t_prof.n_seats)]
            d["curr_net_idxs"] = copy.deepcopy(self._episode_net_idxs)
            d["history"] = copy.deepcopy(self._a_history)

        return d

    def _load_state_dict(self, state):
        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            for i in range(self.t_prof.n_seats):
                self.avrg_net_policies[i].load_net_state_dict(state["avrg_nets"][i])

        # """"""""""""""""""""""""""""
        # SD-CFR
        # """"""""""""""""""""""""""""
        if self._SINGLE:
            for p in range(self.t_prof.n_seats):
                self._strategy_buffers[p].load_state_dict(state=state["strategy_buffers"][p])
            self._a_history = copy.deepcopy(state['history'])
            self._episode_net_idxs = copy.deepcopy(state['curr_net_idxs'])

    # _____________________________________________ SD-CFR specific _____________________________________________
    def _add_history_entry(self, p_id_acting, action_hes_gonna_do):
        self._a_history[p_id_acting]["pub_obs_batch"].append(self._internal_env_wrapper.get_current_obs())
        self._a_history[p_id_acting]["legal_action_list_batch"].append(
            self._internal_env_wrapper.env.get_legal_actions())
        self._a_history[p_id_acting]["a_batch"].append(action_hes_gonna_do)
        self._a_history[p_id_acting]["len"] += 1

    def _get_reach_for_each_model(self, p_id_acting, range_idx):
        models = self._strategy_buffers[p_id_acting].strategies

        H = self._a_history[p_id_acting]
        if H['len'] == 0:
            # Dim: [model_idx]
            return np.ones(shape=(len(models)), dtype=np.float32)

        # """"""""""""""""""""""
        # Batch calls history
        # and computes product
        # of result
        # """"""""""""""""""""""
        # Dim: [model_idx, history_time_step]
        prob_a_each_model_each_timestep = np.array(
            [
                model.get_a_probs(
                    pub_obses=H['pub_obs_batch'],
                    range_idxs=[range_idx] * H['len'],
                    legal_actions_lists=H['legal_action_list_batch'],
                )[np.arange(len(models)), H['a_batch']]

                for model in models
            ]
        )
        # Dim: [model_idx]
        return np.prod(a=prob_a_each_model_each_timestep, axis=1)

    def _get_reach_for_each_model_each_hand(self, p_id_acting):
        # Probability that each model would perform action a (from history) with each hand
        models = self._strategy_buffers[p_id_acting].strategies

        # Dim: [model_idx, range_idx]
        reaches = np.empty(shape=(len(models), self.env_bldr.rules.RANGE_SIZE,), dtype=NP_FLOAT_TYPE)

        H = self._a_history[p_id_acting]

        for m_i, model in enumerate(models):
            non_zero_hands = list(range(self.env_bldr.rules.RANGE_SIZE))

            # """"""""""""""""""""""
            # Batch calls hands but
            # not history timesteps.
            # """"""""""""""""""""""
            reach_hist = np.zeros(shape=(H['len'], self.env_bldr.rules.RANGE_SIZE), dtype=NP_FLOAT_TYPE)
            for hist_idx in range(H['len']):
                if len(non_zero_hands) == 0:
                    break

                # Dim: [model_idx, RANGE_SIZE]
                p_m_a = model.get_a_probs_for_each_hand_in_list(
                    pub_obs=H['pub_obs_batch'][hist_idx],
                    legal_actions_list=H['legal_action_list_batch'][hist_idx],
                    range_idxs=np.array(non_zero_hands),
                )[:, H['a_batch'][hist_idx]]

                reach_hist[hist_idx, non_zero_hands] = p_m_a * len(H['legal_action_list_batch'][hist_idx])
                # collect zeros to avoid unnecessary future queries
                for h_idx in reversed(range(len(non_zero_hands))):
                    if p_m_a[h_idx] == 0:
                        del non_zero_hands[h_idx]

            reaches[m_i] = np.prod(reach_hist, axis=0)

        # Dim: [model_idx, RANGE_SIZE]
        return reaches

    def _sample_new_strategy(self):
        """
        Sample one current strategy from the buffer to play by this episode
        """
        self._episode_net_idxs = [
            self._strategy_buffers[p].sample_strat_idx_weighted()
            for p in range(self.env_bldr.N_SEATS)
        ]

    def _reset_action_history(self):
        self._a_history = {
            p_id: {
                "pub_obs_batch": [],
                "legal_action_list_batch": [],
                "a_batch": [],
                "len": 0,
            }
            for p_id in range(self.env_bldr.N_SEATS)
        }
