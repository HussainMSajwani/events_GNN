from imports.ASLDataset import ASLDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from imports.NVSModel import Net
import torch_geometric.transforms as T
from torch import autograd
from tqdm import tqdm
import numpy as np
import sys
from time import time 
from pathlib import Path

hpc = False

if hpc:
    pwd = Path("/l/proj/kuin0009/hussain/events/events_GNN")
else:
    pwd = Path("/home/hussain/papers_reproduction/sign_language")

letters=['a', 'b']

now = time()

args = {
    '1':{
        'layers': [64, 128, 256, 512],
        'voxels': [20, 30, 50, 80]
        },
    '2':{
        'layers': [64, 128, 256],
        'voxels': [20, 30, 50]
        },
    '3':{
        'layers': [64, 128, 256, 512, 512],
        'voxels': [20, 30, 50, 80, 80]
        },
    '4':{
        'layers': [64, 128, 256, 512, 512, 1024],
        'voxels': [20, 30, 50, 80, 80, 100]
        }
}

run = sys.argv[1]
run_dict = args[run]

def train(model, epoch, train_loader):
    model.train()

    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    if epoch == 110:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    for i, data in enumerate(tqdm(train_loader, desc='batches')):
        with autograd.detect_anomaly():
            data = data.to(device)
            og_data_x={key: data[key].detach().clone() for key in data.keys}
            #print(data.y)
            optimizer.zero_grad()
            end_point = model(data)
            
            loss = F.nll_loss(end_point, data.y)
            if np.isnan(loss.item()):
                print(og_data_x)
                torch.save(og_data_x, pwd / 'results' / run / 'nan.pt')
            pred = end_point.max(1)[1]
            acc = (pred.eq(data.y).sum().item())/len(data.y)
            
            loss.backward()
            optimizer.step()
            
            print({'epoch': epoch,'batch': i + 1,'loss': loss.item(),'acc': acc})
            with open('./log', 'a') as f:
                f.writelines(f'{epoch},{i+1},{loss.item()},{acc}\n')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = Net(
    len(letters),
    layer_sizes=run_dict['layers'],
    voxel_sizes=run_dict['voxels']
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

dataset = ASLDataset(
    letters=letters, 
    overwrite_processing=False, 
    transform=T.Cartesian(cat=False)
    )

n = len(dataset)
data_train, data_test = torch.utils.data.random_split(
    dataset, 
    [int(n*0.8), int(n*0.2)], 
    generator=torch.Generator().manual_seed(42)
)

test_data_idx = data_test.indices
with open(pwd / 'results' / run /'test_data_idx.txt', 'w') as f:
    f.writelines(str(test_data_idx))

dl = DataLoader(data_train, 1, shuffle=True)
train_loader = dl

for epoch in range(1, 150):
    train(model, epoch, train_loader)

    torch.save(model, pwd / 'results' / run / 'model.pkl')