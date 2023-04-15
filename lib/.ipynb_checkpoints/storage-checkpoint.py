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

    def clear(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        self.chunks_count = defaultdict(int)
        self.future = None

    def store(self, name, chunk_id, data): 
        chunk_id = str(chunk_id)
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


    def _find_file(self, name, chunk_id):
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
        return self._read_file(name, chunk_id, *args, *kwargs)

