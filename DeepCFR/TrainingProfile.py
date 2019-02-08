# Copyright (c) 2019 Eric Steinberger


import copy

import torch
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLLeduc
from PokerRL.game.wrappers import HistoryEnvBuilder, FlatLimitPokerEnvBuilder
from PokerRL.rl.base_cls.TrainingProfileBase import TrainingProfileBase
from PokerRL.rl.neural.AvrgStrategyNet import AvrgNetArgs
from PokerRL.rl.neural.DuelingQNet import DuelingQArgs

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.la.AdvWrapper import AdvTrainingArgs
from DeepCFR.workers.la.AvrgWrapper import AvrgTrainingArgs


class TrainingProfile(TrainingProfileBase):

    def __init__(self,

                 # ------ General
                 name="",
                 log_verbose=True,
                 log_export_freq=1,
                 checkpoint_freq=99999999,
                 eval_agent_export_freq=999999999,
                 n_learner_actor_workers=8,
                 max_n_las_sync_simultaneously=10,
                 nn_type="feedforward",  # "recurrent" or "feedforward"

                 # ------ Computing
                 path_data=None,
                 local_crayon_server_docker_address="localhost",
                 device_inference="cpu",
                 device_training="cpu",
                 device_parameter_server="cpu",
                 DISTRIBUTED=False,
                 CLUSTER=False,
                 DEBUGGING=False,

                 # ------ Env
                 game_cls=DiscretizedNLLeduc,
                 n_seats=2,
                 agent_bet_set=bet_sets.B_2,
                 start_chips=None,
                 chip_randomness=(0, 0),
                 uniform_action_interpolation=False,
                 use_simplified_headsup_obs=True,

                 # ------ Evaluation
                 eval_modes_of_algo=(EvalAgentDeepCFR.EVAL_MODE_SINGLE,),
                 eval_stack_sizes=None,

                 # ------ General Deep CFR params
                 n_traversals_per_iter=30000,
                 online=False,
                 iter_weighting_exponent=1.0,
                 n_actions_traverser_samples=3,

                 sampler="mo",

                 # --- Adv Hyperparameters
                 n_batches_adv_training=5000,
                 init_adv_model="random",

                 rnn_cls_str_adv="lstm",
                 rnn_units_adv=128,
                 rnn_stack_adv=1,
                 dropout_adv=0.0,
                 use_pre_layers_adv=False,
                 n_cards_state_units_adv=96,
                 n_merge_and_table_layer_units_adv=32,
                 n_units_final_adv=64,
                 mini_batch_size_adv=4096,
                 n_mini_batches_per_la_per_update_adv=1,
                 optimizer_adv="adam",
                 loss_adv="weighted_mse",
                 lr_adv=0.001,
                 grad_norm_clipping_adv=10.0,
                 lr_patience_adv=999999999,
                 normalize_last_layer_FLAT_adv=True,

                 max_buffer_size_adv=3e6,

                 # ------ SPECIFIC TO AVRG NET
                 n_batches_avrg_training=15000,
                 init_avrg_model="random",

                 rnn_cls_str_avrg="lstm",
                 rnn_units_avrg=128,
                 rnn_stack_avrg=1,
                 dropout_avrg=0.0,
                 use_pre_layers_avrg=False,
                 n_cards_state_units_avrg=96,
                 n_merge_and_table_layer_units_avrg=32,
                 n_units_final_avrg=64,
                 mini_batch_size_avrg=4096,
                 n_mini_batches_per_la_per_update_avrg=1,
                 loss_avrg="weighted_mse",
                 optimizer_avrg="adam",
                 lr_avrg=0.001,
                 grad_norm_clipping_avrg=10.0,
                 lr_patience_avrg=999999999,
                 normalize_last_layer_FLAT_avrg=True,

                 max_buffer_size_avrg=3e6,

                 # ------ SPECIFIC TO SINGLE
                 export_each_net=False,
                 eval_agent_max_strat_buf_size=None,

                 # ------ Optional
                 lbr_args=None,
                 rl_br_args=None,
                 h2h_args=None,

                 ):
        print(" ************************** Initing args for: ", name, "  **************************")

        if nn_type == "recurrent":
            from PokerRL.rl.neural.MainPokerModuleRNN import MPMArgsRNN

            env_bldr_cls = HistoryEnvBuilder

            mpm_args_adv = MPMArgsRNN(rnn_cls_str=rnn_cls_str_adv,
                                      rnn_units=rnn_units_adv,
                                      rnn_stack=rnn_stack_adv,
                                      rnn_dropout=dropout_adv,
                                      use_pre_layers=use_pre_layers_adv,
                                      n_cards_state_units=n_cards_state_units_adv,
                                      n_merge_and_table_layer_units=n_merge_and_table_layer_units_adv)
            mpm_args_avrg = MPMArgsRNN(rnn_cls_str=rnn_cls_str_avrg,
                                       rnn_units=rnn_units_avrg,
                                       rnn_stack=rnn_stack_avrg,
                                       rnn_dropout=dropout_avrg,
                                       use_pre_layers=use_pre_layers_avrg,
                                       n_cards_state_units=n_cards_state_units_avrg,
                                       n_merge_and_table_layer_units=n_merge_and_table_layer_units_avrg)

        elif nn_type == "feedforward":
            from PokerRL.rl.neural.MainPokerModuleFLAT import MPMArgsFLAT

            env_bldr_cls = FlatLimitPokerEnvBuilder

            mpm_args_adv = MPMArgsFLAT(use_pre_layers=use_pre_layers_adv,
                                       card_block_units=n_cards_state_units_adv,
                                       other_units=n_merge_and_table_layer_units_adv,
                                       normalize=normalize_last_layer_FLAT_adv)
            mpm_args_avrg = MPMArgsFLAT(use_pre_layers=use_pre_layers_avrg,
                                        card_block_units=n_cards_state_units_avrg,
                                        other_units=n_merge_and_table_layer_units_avrg,
                                        normalize=normalize_last_layer_FLAT_avrg)

        else:
            raise ValueError(nn_type)

        super().__init__(
            name=name,
            log_verbose=log_verbose,
            log_export_freq=log_export_freq,
            checkpoint_freq=checkpoint_freq,
            eval_agent_export_freq=eval_agent_export_freq,
            path_data=path_data,
            game_cls=game_cls,
            env_bldr_cls=env_bldr_cls,
            start_chips=start_chips,
            eval_modes_of_algo=eval_modes_of_algo,
            eval_stack_sizes=eval_stack_sizes,

            DEBUGGING=DEBUGGING,
            DISTRIBUTED=DISTRIBUTED,
            CLUSTER=CLUSTER,
            device_inference=device_inference,
            local_crayon_server_docker_address=local_crayon_server_docker_address,

            module_args={
                "adv_training": AdvTrainingArgs(
                    adv_net_args=DuelingQArgs(
                        mpm_args=mpm_args_adv,
                        n_units_final=n_units_final_adv,
                    ),
                    n_batches_adv_training=n_batches_adv_training,
                    init_adv_model=init_adv_model,
                    batch_size=mini_batch_size_adv,
                    n_mini_batches_per_update=n_mini_batches_per_la_per_update_adv,
                    optim_str=optimizer_adv,
                    loss_str=loss_adv,
                    lr=lr_adv,
                    grad_norm_clipping=grad_norm_clipping_adv,
                    device_training=device_training,
                    max_buffer_size=max_buffer_size_adv,
                    lr_patience=lr_patience_adv,
                ),
                "avrg_training": AvrgTrainingArgs(
                    avrg_net_args=AvrgNetArgs(
                        mpm_args=mpm_args_avrg,
                        n_units_final=n_units_final_avrg,
                    ),
                    n_batches_avrg_training=n_batches_avrg_training,
                    init_avrg_model=init_avrg_model,
                    batch_size=mini_batch_size_avrg,
                    n_mini_batches_per_update=n_mini_batches_per_la_per_update_avrg,
                    loss_str=loss_avrg,
                    optim_str=optimizer_avrg,
                    lr=lr_avrg,
                    grad_norm_clipping=grad_norm_clipping_avrg,
                    device_training=device_training,
                    max_buffer_size=max_buffer_size_avrg,
                    lr_patience=lr_patience_avrg,
                ),
                "env": game_cls.ARGS_CLS(
                    n_seats=n_seats,
                    starting_stack_sizes_list=[start_chips for _ in range(n_seats)],
                    bet_sizes_list_as_frac_of_pot=copy.deepcopy(agent_bet_set),
                    stack_randomization_range=chip_randomness,
                    use_simplified_headsup_obs=use_simplified_headsup_obs,
                    uniform_action_interpolation=uniform_action_interpolation
                ),
                "lbr": lbr_args,
                "rlbr": rl_br_args,
                "h2h": h2h_args,
            }
        )

        self.nn_type = nn_type
        self.online = online
        self.n_traversals_per_iter = n_traversals_per_iter
        self.iter_weighting_exponent = iter_weighting_exponent
        self.sampler = sampler
        self.n_actions_traverser_samples = n_actions_traverser_samples

        # SINGLE
        self.export_each_net = export_each_net
        self.eval_agent_max_strat_buf_size = eval_agent_max_strat_buf_size

        # Different for dist and local
        if DISTRIBUTED or CLUSTER:
            print("Running with ", n_learner_actor_workers, "LearnerActor Workers.")
            self.n_learner_actors = n_learner_actor_workers
        else:
            self.n_learner_actors = 1
        self.max_n_las_sync_simultaneously = max_n_las_sync_simultaneously

        assert isinstance(device_parameter_server, str), "Please pass a string (either 'cpu' or 'cuda')!"
        self.device_parameter_server = torch.device(device_parameter_server)
