from imports.ASLDataset import ASLDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from imports.NVSModel import Net
import torch_geometric.transforms as T
from torch import autograd
from tqdm import tqdm
import numpy as np


def train(model, epoch, train_loader):
    model.train()

    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    if epoch == 110:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    for i, data in enumerate(tqdm(train_loader, desc='batches')):
        data = data.to(device)
        og_data_x={key: data[key].detach().clone() for key in data.keys}
        #print(data.y)
        optimizer.zero_grad()
        end_point = model(data)
        
        loss = F.nll_loss(end_point, data.y)
        if np.isnan(loss.item()):
            print(og_data_x)
            torch.save(og_data_x, '/home/hussain/papers_reproduction/sign_language/nan.pt')
        pred = end_point.max(1)[1]
        acc = (pred.eq(data.y).sum().item())/len(data.y)
        
        loss.backward()
        optimizer.step()
        
        print({'epoch': epoch,'batch': i + 1,'loss': loss.item(),'acc': acc})
        with open('./log', 'a') as f:
            f.writelines(f'{epoch},{i+1},{loss.item()},{acc}\n')

letters=['a', 'b']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(len(letters)).to(device)
print(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

dataset = ASLDataset(letters=letters, overwrite_processing=False, transform=T.Cartesian(cat=False))
n = len(dataset)
data_train, data_test = torch.utils.data.random_split(
    dataset, 
    [int(n*0.8), int(n*0.2)], 
    generator=torch.Generator().manual_seed(42)
)

test_data_idx = data_test.indices
with open('test_data_idx.txt', 'w') as f:
    f.writelines(str(test_data_idx))

dl = DataLoader(data_train, 15, shuffle=True)
train_loader = dl

for epoch in range(1, 15):
    train(model, epoch, train_loader)

    torch.save(model, '/home/hussain/papers_reproduction/sign_language/model.pkl')