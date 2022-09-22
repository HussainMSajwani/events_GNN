from string import ascii_lowercase
from sys import prefix
import numpy as np

from pathlib import Path
from scipy.io import loadmat, savemat
from functools import reduce
import os.path as osp

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.nn.pool import radius_graph
import torch_geometric.transforms as T
from torch_geometric.utils import remove_isolated_nodes
from tqdm.auto import tqdm


#torch.set_default_dtype(torch.float32) 

letters_idx = {letter: idx for idx, letter in enumerate(ascii_lowercase)}

hpc = False

if hpc:
    datadir = Path("/l/proj/kuin0009/hussain/events/data")
else:
    datadir = Path("/home/hussain/data/event_based_sign_lang/")


class ASLDataset(Dataset):
    def __init__(
        self, 
        root=datadir, 
        letters = "all", 
        beta=0.2e-2, 
        n_samples=8000, 
        radius=5, 
        max_n_neighbors=32,
        overwrite_processing=False,
        train_val_test = "train",
        transform=None, 
        pre_transform=None, 
        pre_filter=None
        ):

        if letters == "all":
            self.letters = list(ascii_lowercase)
        else:
            #print(letters)
            self.letters = letters
        
        self.beta = beta
        self.n_samples = n_samples
        self.radius = radius
        self.max_n_neighbors = max_n_neighbors
        self.overwrite_processing = overwrite_processing
        self.train_val_test = train_val_test

        root = root / train_val_test

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return self.root / 'raw'

    @property
    def processed_dir(self) -> str:
        return self.root / 'processed'
        # in super() _process is called when PROC_DIR is empty.


    @property
    def raw_file_names(self):
        all_files = []
        for letter in self.letters:
            prefix = self.raw_dir / letter 
            all_files += [prefix / f"{letter}_{str(sample_id).zfill(4)}.mat" for sample_id in range(1, 4201)]
        return all_files

    @property
    def processed_file_names(self):
        all_files = []
        for letter in self.letters:
            prefix = self.processed_dir / letter 
            all_files += [prefix / f"{letter}_{str(sample_id).zfill(4)}.mat" for sample_id in range(1, 4201)]
        return all_files


    def process(self):
        letter_idx = reduce(
            list.__add__, 
            map(
                lambda let: 4200*[let], 
                self.letters
                )
            )
        iter = tqdm(self.raw_paths, desc='processing')
        iter = zip(letter_idx, iter)
        iter = enumerate(iter)

        for idx, (letter, raw_path) in iter:
            # Read data from `raw_path`.

            sample_id = idx % 4200 + 1
            save_dir = self.processed_dir / letter / f"{letter}_{str(sample_id).zfill(4)}.pt"

            if osp.exists(save_dir) and not self.overwrite_processing:
                #if the file exists and we don't want to overwrite, then skip
                continue

            mat = self._get_mat(raw_path)
            mat = {key: mat[key].ravel() for key in mat.keys()}

            ts = mat["ts"]*self.beta
            st_pos = np.array([mat["x"], mat["y"], ts]).T
           #st_pos = np.array([mat["x"], mat["y"]]).T
            st_pos = torch.from_numpy(st_pos.astype(np.float32))

            pol = 2*mat["pol"].astype(np.float32) - 1
            #print(pol)
            pol = torch.from_numpy(pol.reshape(-1, 1))
            
            y = letters_idx[letter]

            data = Data(pos=st_pos, y=y, x=pol)
            sampler = T.FixedPoints(num=self.n_samples, allow_duplicates=False, replace=False)
            data = sampler(data)

            edge_index = radius_graph(data['pos'], r=self.radius, max_num_neighbors=self.max_n_neighbors)
            edge_index, _, mask = remove_isolated_nodes(edge_index=edge_index, num_nodes=data.x.shape[0])

            data.edge_index = edge_index
            #print(mask.shape, data.x.shape, data.edge_index.shape)
            

            pseudo_maker = T.Cartesian(cat=False, norm=True)
            data = pseudo_maker(data)
            
            data.x = data.x[mask]


            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self._pre_transform(data)
            torch.save(data.to("cpu"), save_dir)
            #print(type())
            #savemat(save_dir, {k: v.numpy() if type(v) == torch.Tensor else v for k,v in data.items()})


    def len(self):
        return len(self.processed_file_names)


    def get(self, idx):
        sample_id = idx % 4200 + 1
        letter = self.letters[idx // 4200]
        save_dir = self.processed_dir / letter / f"{letter}_{str(sample_id).zfill(4)}.pt"
        data = torch.load(save_dir)
        #data = self._get_mat(save_dir)
        #data = {k: torch.from_numpy(v) for k, v in data.items()}
        #data = Data(**data)
        return data

    def _get_mat(self, raw_path):
        mat = loadmat(raw_path)
        mat.pop("__header__")
        mat.pop("__version__")
        mat.pop("__globals__")
        return mat