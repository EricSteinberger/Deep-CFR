# Copyright (c) 2019 Eric Steinberger


import time
from os.path import dirname, abspath

import numpy as np

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR

# These two eval agents HAVE TO come from the same training run and iteration for this analysis to make sense.
path_to_dcfr_eval_agent = dirname(abspath(__file__)) + "/trained_agents/Example_FHP_AVRG_NET.pkl"
path_to_sdcfr_eval_agent = dirname(abspath(__file__)) + "/trained_agents/Example_FHP_SINGLE.pkl"

N_GAMES = 200000
MAX_DEPTH = 6  # This is a constant for FHP


def stats(data):
    if len(data) == 0:
        return {
            "mean": None,
            "std": None,
            "conf": None,
            "N": None,
        }

    mean = np.mean(data).item()
    std = np.std(data).item()
    conf = 1.96 * std / np.sqrt(len(data))
    return {
        "mean": float(mean),
        "std": float(std),
        "conf": float(conf),
        "N": len(data),
    }


if __name__ == '__main__':
    eval_agent_dcfr = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_dcfr_eval_agent)
    eval_agent_sdcfr = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_sdcfr_eval_agent)
    assert eval_agent_dcfr.t_prof.name == eval_agent_sdcfr.t_prof.name

    env_bldr = eval_agent_dcfr.env_bldr
    env = env_bldr.get_new_env(is_evaluating=False)

    strategy_differences = {
        r: {
            depth: []
            for depth in range(MAX_DEPTH)
        }
        for r in env_bldr.rules.ALL_ROUNDS_LIST
    }

    start_time = time.time()

    for sample in range(1, N_GAMES + 1):
        for p_eval in range(2):
            obs, rew, done, info = env.reset()
            depth = 0

            eval_agent_sdcfr.reset(deck_state_dict=env.cards_state_dict())
            eval_agent_dcfr.reset(deck_state_dict=env.cards_state_dict())

            while not done:
                p_id_acting = env.current_player.seat_id
                legal_actions_list = env.get_legal_actions()

                # Step according to agent strategy
                if p_id_acting == p_eval:

                    # Compare Action-probability distribution
                    a_dist_sdcfr = eval_agent_sdcfr.get_a_probs()
                    a_dist_dcfr = eval_agent_dcfr.get_a_probs()
                    strategy_differences[env.current_round][depth] \
                        .append(np.sum(np.abs(a_dist_sdcfr - a_dist_dcfr)))
                    a = eval_agent_sdcfr.get_action(step_env=False)[0]

                # Handle env stepping along random trajectories
                else:
                    a = legal_actions_list[np.random.randint(len(legal_actions_list))]

                obs, rew, done, info = env.step(a)
                eval_agent_sdcfr.notify_of_action(p_id_acted=p_id_acting, action_he_did=a)
                eval_agent_dcfr.notify_of_action(p_id_acted=p_id_acting, action_he_did=a)

                depth += 1

        if sample % 10 == 0:
            print("Sample:", sample,
                  "\t  Time remaining: ",
                  str("%.0f" % ((time.time() - start_time) * (N_GAMES - sample) / sample)) + "s")

    avg_strat_diff = {
        r: {
            depth: stats(strategy_differences[r][depth])
            for depth in range(MAX_DEPTH)
        }
        for r in env_bldr.rules.ALL_ROUNDS_LIST
    }

    print("Average absolute difference:")
    for r, v in avg_strat_diff.items():
        for depth in range(MAX_DEPTH):
            if v[depth]["mean"] is not None:
                print("Round:", r,
                      "\t   Depth: ", "%.5f" % depth,
                      "\t   Dif Mean:", "%.5f" % v[depth]["mean"], "+/-", "%.5f" % v[depth]["conf"],
                      "\t   Dif STD:", "%.5f" % v[depth]["std"],
                      "\t   n_data:", "%.5f" % v[depth]["N"],
                      )
