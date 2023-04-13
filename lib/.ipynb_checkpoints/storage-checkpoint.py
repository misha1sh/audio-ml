from pathlib import Path
from collections import defaultdict
import pickle
import torch
import os
import asyncio
import shutil

class Storage:
    def __init__(self, path):
        self.path = Path(path)
        self.chunks_count = defaultdict(int)
        self.future = None

    def clear(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        self.chunks_count = defaultdict(int)
        self.future = None

    def store(self, name, data): 
        chunk_id = str(self.chunks_count[name])
        os.makedirs(self.path / chunk_id, exist_ok=True)
        
        if isinstance(data, torch.Tensor):
            file_ext = '.pt'
        else:
            file_ext = '.pickle'

        with open(self.path / chunk_id / f'{name}_{chunk_id}{file_ext}', 'wb') as file:
            if isinstance(data, torch.Tensor):
                torch.save(data, file)
            else:
                pickle.dump(data, file)
            self.chunks_count[name] += 1

    def get_chunks_count(self, name):
        return self.chunks_count[name]

    def start_loading_async(self, name, chunk_id, *args, **kwargs):
        if chunk_id >= self.chunks_count[name]: return
        loop = asyncio.get_running_loop()
        def task():
            return self._read_file(name, chunk_id, *args, **kwargs)
        self.future = {
            "name": name,
            "chunk_id": int(chunk_id),
            "future": loop.run_in_executor(None, task)

            #asyncio.create_task(self._read_file_async(*self._find_file(name, chunk_id)))
        }

    def _find_file(self, name, chunk_id):
        if chunk_id >= self.get_chunks_count(name):
            raise ValueError(f"Too far chunk {name} {chunk_id}  {self.get_chunks_count(chunk_id)}")

        filepath = str(self.path / str(chunk_id) / f'{name}_{chunk_id}')
        if os.path.isfile(filepath + '.pickle'):
            return filepath + '.pickle', pickle.load
        elif os.path.isfile(filepath + '.pt'):
            return filepath + '.pt', torch.load
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    def _read_file(self, name, chunk_id, *args, **kwargs):
        filepath, method = self._find_file(name, chunk_id)
        with open(filepath, "rb") as file:
            return method(file, *args, **kwargs)

    def get(self, name, chunk_id, *args, **kwargs):
        # if self.future and self.future["name"] == name and self.future["chunk_id"] == int(chunk_id):
        #     res = await self.future["future"]
        #     self.future  = None
        #     return res
        return self._read_file(name, chunk_id, *args, *kwargs)

# storage = Storage("cache/storage")
# storage.clear()
# storage.store("test", "1")
# storage.start_loading_async("test", 0)
# await storage.get("test", 0)
