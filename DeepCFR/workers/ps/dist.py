import ray
import torch

from DeepCFR.workers.ps.local import ParameterServer as _LocalParameterServer


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class ParameterServer(_LocalParameterServer):

    def __init__(self, t_prof, owner, chief_handle):
        super().__init__(t_prof=t_prof, owner=owner, chief_handle=chief_handle)
