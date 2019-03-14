from PokerRL.game.games import StandardLeduc

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="SD-CFR_LEDUC_BUF_1000",
                                         nn_type="feedforward",
                                         max_buffer_size_adv=1e6,
                                         max_buffer_size_avrg=1e6,
                                         eval_agent_export_freq=999999,
                                         n_traversals_per_iter=1500,
                                         n_batches_adv_training=750,
                                         n_batches_avrg_training=5000,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_merge_and_table_layer_units_avrg=64,
                                         n_units_final_adv=64,
                                         n_units_final_avrg=64,
                                         mini_batch_size_adv=2048,
                                         mini_batch_size_avrg=2048,
                                         init_adv_model="last",
                                         init_avrg_model="random",
                                         use_pre_layers_adv=False,
                                         use_pre_layers_avrg=False,
                                         eval_agent_max_strat_buf_size=1000,

                                         game_cls=StandardLeduc,

                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                         ),

                                         DISTRIBUTED=False,
                                         log_verbose=False,
                                         ),
                  eval_methods={
                      "br": 15,
                  },
                  n_iterations=None)
    ctrl.run()
