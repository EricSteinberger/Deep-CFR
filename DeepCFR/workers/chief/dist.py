import ray

from DeepCFR.workers.chief.local import Chief as _LocalChief


@ray.remote(num_cpus=1)
class Chief(_LocalChief):

    def __init__(self, t_prof):
        _LocalChief.__init__(self, t_prof=t_prof)
