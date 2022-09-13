from scipy.io import loadmat, savemat
from pathlib import Path
from random import randrange
from pandas import DataFrame
from .utils import make_graph
from tqdm import tqdm, trange

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
import torch_geometric.transforms as T

DATADIR=Path("/home/hussain/data/event_based_sign_lang/")
RAWDIR = DATADIR / "raw"
PROCDIR = DATADIR / "processed"

def load_sample(letter, sample_id=None):
    if sample_id is None:
        sample_id = randrange(1, 4200)
        print(f"returning sample {sample_id}")
    mat_dir = RAWDIR / letter / f"{letter}_{str(sample_id).zfill(4)}.mat"
    mat = loadmat(mat_dir)
    mat.pop("__header__")
    mat.pop("__version__")
    mat.pop("__globals__")
    #flatten cols
    mat = {key: mat[key].ravel() for key in mat.keys()}
    return DataFrame(mat)

def save_graph(graph, letter, sample_id):
    mat_dir = PROCDIR / letter / f"{letter}_{str(sample_id).zfill(4)}.mat"
    savemat(mat_dir, {k: v for k, v in graph.items()})

def process(letter, beta=0.5e-6, n_samples=8000, radius=5, max_n_neighbors=32):
    for sample_id in trange(1, 4201):
        events_df = load_sample(letter, sample_id)
        graph = make_graph(events_df, beta=beta, n_samples=n_samples, radius=radius, max_n_neighbors=max_n_neighbors)
        save_graph(graph, letter, sample_id)