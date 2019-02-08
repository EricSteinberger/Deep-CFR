import torch

from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs


class SamplerBase:

    def __init__(self, env_bldr, adv_buffers, avrg_buffers=None, ):
        self._env_bldr = env_bldr
        self._adv_buffers = adv_buffers
        self._avrg_buffers = avrg_buffers
        self._env_wrapper = self._env_bldr.get_new_wrapper(is_evaluating=False)

    def _traverser_act(self, start_state_dict, traverser, trav_depth, plyrs_range_idxs, iteration_strats, cfr_iter):
        raise NotImplementedError

    def generate(self, n_traversals, traverser, iteration_strats, cfr_iter, ):
        for _ in range(n_traversals):
            self._traverse_once(traverser=traverser, iteration_strats=iteration_strats, cfr_iter=cfr_iter)

    def _traverse_once(self, traverser, iteration_strats, cfr_iter, ):
        """
        Args:
            traverser (int):                    seat id of the traverser
            iteration_strats (IterationStrategy):
            cfr_iter (int):                  current iteration of Deep CFR
        """
        self._env_wrapper.reset()
        self._recursive_traversal(start_state_dict=self._env_wrapper.state_dict(),
                                  traverser=traverser,
                                  trav_depth=0,
                                  plyrs_range_idxs=[
                                      self._env_wrapper.env.get_range_idx(p_id=p_id)
                                      for p_id in range(self._env_bldr.N_SEATS)
                                  ],
                                  iteration_strats=iteration_strats,
                                  cfr_iter=cfr_iter,
                                  )

    def _recursive_traversal(self, start_state_dict, traverser, trav_depth, plyrs_range_idxs, iteration_strats,
                             cfr_iter):
        """
        assumes passed state_dict is NOT done!
        """

        if start_state_dict["base"]["env"][EnvDictIdxs.current_player] == traverser:
            return self._traverser_act(start_state_dict=start_state_dict,
                                       traverser=traverser,
                                       trav_depth=trav_depth,
                                       plyrs_range_idxs=plyrs_range_idxs,
                                       iteration_strats=iteration_strats,
                                       cfr_iter=cfr_iter)

        return self._any_non_traverser_act(start_state_dict=start_state_dict,
                                           traverser=traverser,
                                           trav_depth=trav_depth,
                                           plyrs_range_idxs=plyrs_range_idxs,
                                           iteration_strats=iteration_strats,
                                           cfr_iter=cfr_iter)

    def _any_non_traverser_act(self, start_state_dict, traverser, plyrs_range_idxs, trav_depth, iteration_strats,
                               cfr_iter):
        self._env_wrapper.load_state_dict(start_state_dict)
        p_id_acting = self._env_wrapper.env.current_player.seat_id

        current_pub_obs = self._env_wrapper.get_current_obs()
        range_idx = plyrs_range_idxs[p_id_acting]
        legal_actions_list = self._env_wrapper.env.get_legal_actions()

        # """""""""""""""""""""""""
        # The players strategy
        # """""""""""""""""""""""""
        a_probs = iteration_strats[p_id_acting].get_a_probs(
            pub_obses=[current_pub_obs],
            range_idxs=[range_idx],
            legal_actions_lists=[legal_actions_list],
            to_np=False
        )[0]

        # """""""""""""""""""""""""
        # Adds to opponent's
        # average buffer if
        # applicable
        # """""""""""""""""""""""""
        if self._avrg_buffers is not None:
            self._avrg_buffers[p_id_acting].add(
                pub_obs=current_pub_obs,
                range_idx=range_idx,
                legal_actions_list=legal_actions_list,
                a_probs=a_probs.to(self._avrg_buffers[p_id_acting].device).squeeze(),
                iteration=cfr_iter + 1)

        # """""""""""""""""""""""""
        # Execute action from strat
        # """""""""""""""""""""""""
        a = torch.multinomial(a_probs.cpu(), num_samples=1).item()
        _obs, _rew_for_all, _done, _info = self._env_wrapper.step(a)
        _rew_traverser = _rew_for_all[traverser]

        # """""""""""""""""""""""""
        # Recurse or Return if done
        # """""""""""""""""""""""""
        if _done:
            return _rew_traverser
        return _rew_traverser + self._recursive_traversal(
            start_state_dict=self._env_wrapper.state_dict(),
            traverser=traverser,
            trav_depth=trav_depth,
            plyrs_range_idxs=plyrs_range_idxs,
            iteration_strats=iteration_strats,
            cfr_iter=cfr_iter
        )
