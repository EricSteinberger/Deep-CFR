# Deep CFR & Single Deep CFR
A scalable implementation of [Deep CFR](https://arxiv.org/pdf/1811.00164.pdf) [1] and its successor
[Single Deep CFR (SD-CFR)](https://arxiv.org/pdf/1901.07621.pdf) [2] in the
[PokerRL](https://github.com/TinkeringCode/PokerRL) framework.

This codebase is designed for:
- Researchers to compare new methods to these baselines.
- Anyone wanting to learn about Deep RL in imperfect information games.

This implementation seamlessly be runs on your local machine and on hundreds of cores on AWS.

### Reproducing Results from Single Deep CFR (Steinberger 2019) [2]
The run-script `DeepCFR/paper_experiment_sdcfr_vs_deepcfr_h2h.py` launches one run of the Head-to-Head performance comparison
between Single Deep CFR and Deep CFR as presented in [2]. We ran the experiments on an m5.12xlarge instance where
we disabled hyper-threading. We set the instance up for distributed runs as explained in
[PokerRL](https://github.com/TinkeringCode/PokerRL). To reproduce, you can simply clone this repository onto the
instance and start the script via
```
git clone https://github.com/TinkeringCode/Deep-CFR.git
cd Deep-CFR
python paper_experiment_sdcfr_vs_deepcfr_h2h.py
```
and watch the results coming in at `INSTANCE_IP:8888` in your browser.

Very Important Notes:
- This implementation defines an iteration as one sequential update for BOTH players. Thus, **iteration 300 in the plot in [2]
  is equivalent to iteration 150 in the Tensorboard logs!**
- Results on iteration 0 have no meaning since they compare a random neural network to an exactly uniform strategy.
 
The action-probability comparison was conducted on a single CPU using `analyze_sdcfr_vs_dcfr_strategy.py`.


## (Single) Deep CFR on your Local Machine

### Install locally
This codebase only supports Linux officially (Mac has not been tested).

First, please install Docker and download the [PyCrayon](https://github.com/torrvision/crayon) container. For dependency
management we recommend Miniconda. To install the dependencies, simply activate your conda environment and then run

```
conda install pytorch=0.4.1 -c pytorch -y
pip install PokerRL
```


### Running experiments locally
Before starting (Single) Deep CFR, please spin up the log server by
```
docker run -d -p 8888:8888 -p 8889:8889 --name crayon alband/crayon
docker start crayon
```

You can now view logs at `localhost:8888` in your browser. To run Deep CFR or SD-CFR with custom hyperparameters in
any Poker game supported by PokerRL, build a script similar to `DeepCFR/leduc_example.py`. Run-scripts define
the hyperparameters, the game to be played, and the evaluation metrics. Here is a very minimalistic example showing a
few of the available settings:

```
from PokerRL.game.games import StandardLeduc  # or any other game

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="SD-CFR_LEDUC_EXAMPLE",
    
                                         eval_agent_export_freq=20,  # export API to play against the agent
                                         
                                         nn_type="feedforward", # we also support recurrent nets
                                         max_buffer_size_adv=3e6,
                                         n_traversals_per_iter=1500,
                                         n_batches_adv_training=750,
                                         init_adv_model="last", # "last" or "random"

                                         game_cls=StandardLeduc, # The game to play     
                                         
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # Single Deep CFR (SD-CFR)
                                         ),

                                         DISTRIBUTED=False, # Run locally
                                         ),
                  eval_methods={
                      "br": 3, # evaluate Best Response every 3 iterations.
                  })
    ctrl.run()
```
Note that you can specify one or both averaging methods under `eval_modes_of_algo`.
Choosing both is useful to compare them as they will share the value networks! However, we showed in [2] that SD-CFR
is expected to perform better, is faster, and requires less memory.
                                         

## Cloud & Clusters
For deployment on AWS, whether single-core, many-core distributed, or on a cluster, please first follow
the tutorial in the corresponding section of [PokerRL](https://github.com/TinkeringCode/PokerRL)'s README.

We recommend forking this repository so you can write your own scripts but still have remote access through git.
In your run-script set either the `DISTRIBUTED` or the `CLUSTER` option of the TrainingProfile to True
(see e.g. `DeepCFR/paper_experiment_sdcfr_vs_deepcfr_h2h.py`).
Moreover, you should specify the number of `LearnerActor` and evaluator workers (if applicable) you want to deploy.
Note that hyperparmeters ending with "_per_la" (e.g. the batch size) are effectively multiplied by the number of
workers. 

When running in DISTRIBUTED mode (i.e. one machine, many cores), simply ssh onto your AWS instance, get your code
onto it (e.g. through git cloning your forked repo) and start your run-script.
To fire up a cluster, define a `.yaml` cluster configuration that properly sets up your workers. Each of them
should have a copy of your forked repo as well as all dependencies on it.
Use `ray up ...` in an ssh session to the head of the cluster to start the job - more detailed instructions about 
the underlying framework we use for distributed computing can be found at [ray](https://github.com/ray-project/ray).





## Citing
If you use this repository in your research, you can cite it by citing PokerRL as follows:
```
@misc{steinberger2019pokerrl,
    author = {Eric Steinberger},
    title = {PokerRL},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/TinkeringCode/PokerRL}},
}
```




## Authors
* **Eric Steinberger**





## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.





## References
[1] Brown, Noam, et al. "Deep Counterfactual Regret Minimization." arXiv preprint arXiv:1811.00164 (2018).

[2] Steinberger, Eric. "Single Deep Counterfactual Regret Minimization." arXiv preprint arXiv:1901.07621 (2019).
