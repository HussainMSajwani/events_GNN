from pathlib import Path
import shutil
from tqdm.auto import tqdm
import numpy as np
import json

def random_split(x, prop=[0.80, 0.1, 0.1]):
    assert sum(prop) == 1, "sum of prop must be 1"
    n = len(x)
    np.random.shuffle(x)
    
    train_idx = int(prop[0]*n) 
    val_idx = train_idx + int(prop[1]*n) 
    out = np.split(x, [train_idx, val_idx])
    return {'train': out[0].tolist(), 'val': out[1].tolist(), 'test': out[2].tolist()}
    
zpad = lambda x, n: str(x).zfill(n)

np.random.seed(0)
letters = ['b', 'd']
datadir = Path("/l/proj/kuin0009/hussain/events/data")
new_dir = datadir.parent / 'data1'
split_json = {}
for letter in letters:
    split = random_split(np.arange(1, 4201))
    split_json[letter] = split
    for key, part in split.items():
        for i, idx in enumerate(tqdm(part, desc=f'{letter}_{key}')):
            raw = datadir / 'raw' / letter / f'{letter}_{zpad(idx, 4)}.mat'
            proc = datadir / 'processed' / letter / f'{letter}_{zpad(idx, 4)}.pt'
            
            shutil.copy2(raw, new_dir / key / 'raw' / letter / f'{letter}_{zpad(i, 4)}.mat')
            shutil.copy2(proc, new_dir / key / 'processed' / letter / f'{letter}_{zpad(i, 4)}.pt')

with open('train_val_test_indices.json') as f:
    data = json.load(f)

data.update(split_json)
with open('train_val_test_indices.json', 'w') as f:
    json.dump(data, f, indent=4)