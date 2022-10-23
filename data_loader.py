from cacher import root, file_cached, mem_cached, clear_cache
import urllib
import os
import zipfile
import fma.utils as fma
import librosa
import audioread.exceptions

from utils import ProgressParallel
from joblib import delayed

BASE_URL = "https://os.unil.cloud.switch.ch/fma/"

CHECKSUMS = {
    "fma_metadata.zip": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
    "fma_small.zip": "ade154f733639d52e35e32f5593efe5be76c6d70",
    "fma_medium.zip": "c67b69ea232021025fca9231fc1c7c1a063ab50b",
    "fma_large.zip": "497109f4dd721066b5ce5e5f250ec604dc78939e",
    "fma_full.zip": "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab",
}


def download_file(file):
    file_path = os.path.join(root, file)
    if os.path.isfile(file_path):
        return file_path
    print("donwloading", file, "to", file_path)
    url = BASE_URL + file
    urllib.request.urlretrieve(url, file_path)
    return file_path

def unzip_file(file_path):
    unzipped_dir = os.path.splitext(file_path)[0]
    if os.path.isdir(unzipped_dir):
        return unzipped_dir

    print("extracting", file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    return unzipped_dir

@mem_cached("fma_tracks_list")
@file_cached("fma_tracks_list")
def fma_tracks_list(type='small'):
    fma_metadata_dir = unzip_file(download_file("fma_metadata.zip"))

    tracks = fma.load(os.path.join(fma_metadata_dir, 'tracks.csv'))
    small = tracks[tracks['set', 'subset'] <= type]
    return list(small.track.index)



def read_wav(directory_path, track_id, sample_rate, length_seconds, length):
    warnings.filterwarnings('ignore', message=".*PySoundFile.*")
    try:
        file_path = fma.get_audio_path(directory_path, track_id)
        data, sr = librosa.load(file_path, sr=sample_rate, mono=True, duration=length_seconds + 0.01)
        if len(data) < length:
            return None
        return data[0: length]
    except audioread.exceptions.NoBackendError:
        return None

@mem_cached("read_wavs")
@file_cached("read_wavs")
def read_wavs(type, sample_rate, length_seconds, length):
    directory_path = unzip_file(download_file("fma_" + type + ".zip"))
    track_ids = fma_tracks_list(type=type)

    tasks = [delayed(read_wav)(directory_path, track_id, sample_rate, length_seconds, length) for track_id in track_ids]

    dataset = ProgressParallel(n_jobs=8, total=len(tasks))(tasks)
    dataset = [i for i in dataset if i is not None]
    return np.array(dataset)
