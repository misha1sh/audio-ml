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
    Dprint("rpc connected")
    time.sleep(1e9)