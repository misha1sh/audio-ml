import torch.distributed.rpc as rpc
import dill
import os
import time

def _run_remote(data):
    func, args, kwargs = dill.loads(data)
    return func(*args, **kwargs)

def run_rpc(func_to_call, worker_name, func, *args, **kwargs):
    data_to_send = dill.dumps((func, args, kwargs))
    return func_to_call(worker_name, _run_remote, (data_to_send, ))


def add_numbers(a, b):
    return a + b

if __name__ == '__main__':
    print("starting rpc")
    init_method = os.getenv('INIT_METHOD', 'tcp://51.250.75.187:2345')
    rpc.init_rpc("worker", rpc.BackendType.TENSORPIPE, rank=1, world_size=2, rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method, num_worker_threads=1))
    print("rpc connected")
    time.sleep(1e9)