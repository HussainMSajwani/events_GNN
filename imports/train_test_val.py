from pathlib import Path
import shutil
from tqdm.auto import tqdm
import numpy as np

def random_split(x, prop=[0.80, 0.1, 0.1]):
    assert sum(prop) == 1, "sum of prop must be 1"
    n = len(x)
    np.random.shuffle(x)
    
    train_idx = int(prop[0]*n) 
    val_idx = train_idx + int(prop[1]*n) 
    out = np.split(x, [train_idx, val_idx])
    return {'train': out[0], 'val': out[1], 'test': out[2]}
    
zpad = lambda x, n: str(x).zfill(n)

np.random.seed(0)
letters = ['a']
datadir = Path("/home/hussain/data/event_based_sign_lang/")
new_dir = datadir.parent / 'event_based_sign_lang'
for letter in letters:
    split = random_split(np.arange(1, 4201))
    for key, part in split.items():
        for i, idx in enumerate(tqdm(part, desc=f'{letter}_{key}')):
            raw = datadir / 'raw' / letter / f'{letter}_{str(idx).zfill(4)}.mat'
            proc = datadir / 'processed' / letter / f'{letter}_{str(idx).zfill(4)}.pt'
            
            shutil.move(raw, new_dir / key / 'raw' / letter / f'{letter}_{zpad(i, 4)}.mat')
            shutil.move(proc, new_dir / key / 'processed' / letter / f'{letter}_{zpad(i, 4)}.pt')