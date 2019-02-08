import ray
import torch

from DeepCFR.workers.la.local import LearnerActor as LocalLearnerActor


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class LearnerActor(LocalLearnerActor):

    def __init__(self, t_prof, worker_id, chief_handle):
        LocalLearnerActor.__init__(self, t_prof=t_prof, worker_id=worker_id, chief_handle=chief_handle)
