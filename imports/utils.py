from random import randrange
import numpy as np
from matplotlib.pyplot import imshow

import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
import torch_geometric.transforms as T

def visualize_frames(events_df, start=0, finish=None):
    H, W = 240, 180
    if finish is None:
        finish = events_df.shape[0]
    img = np.zeros((H, W, 3))
    for i in range(start, finish):
        pol, ts, x, y = events_df.iloc[i, :]
        if pol==1:
            img[x, y, 0] = 255 #pol=1 events are drawn in red
        elif pol==0:
            img[x, y, 2] = 255 #pol=1 events are drawn in blue
    imshow(img)
    
def make_graph(events_df, beta=0.5e-6, n_samples=8000, radius=5, max_n_neighbors=32):
    ts = events_df["ts"]*beta
    st_pos = np.array([events_df["x"], events_df["y"], ts]).T
    st_pos = torch.from_numpy(st_pos)
    pol = torch.from_numpy(2*events_df["pol"].astype(int).values - 1)
    
    data = Data(x=pol, pos=st_pos, num_nodes=len(st_pos))
    #subsample data
    sampler = T.FixedPoints(num=n_samples, allow_duplicates=False, replace=False)
    sampled = sampler(data)
    edge_index = radius_graph(sampled["pos"], r=radius, max_num_neighbors=max_n_neighbors)
    data.edge_index = edge_index

    sampled_transformed = transforms(sampled)
    return sampled_transformed



#NVS paper Test.py 
transforms = T.Compose([
    T.Cartesian(cat=False, norm=False), 
    T.RandomFlip(axis=0, p=0.3), 
    T.RandomScale([0.95,0.999]),
    T.RandomFlip(axis=1, p=0.2)]
)