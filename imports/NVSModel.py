import torch
from torch import autograd
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x
from tqdm import tqdm

transform = T.Cartesian(cat=False)

class Net(torch.nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = SplineConv(64, 128, dim=3, kernel_size=5)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.conv3 = SplineConv(128, 256, dim=3, kernel_size=5)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.conv4 = SplineConv(256, 512, dim=3, kernel_size=5)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.fc1 = torch.nn.Linear(64 * 512, 1024)
        self.fc2 = torch.nn.Linear(1024, n_classes)

    def forward(self, data):
        inp = self.conv1(data.x, data.edge_index, data.edge_attr)
        data.x = F.elu(inp)
        data.x = self.bn1(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=[4,4])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn2(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=[6,6])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn3(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=[20,20])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn4(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=[32,32])
        x, _ = max_pool_x(cluster, data.x, data.batch, size=64)

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


def train(model, epoch, train_loader, device):
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
            #print(data.y)
            optimizer.zero_grad()
            end_point = model(data)
            
            loss = F.nll_loss(end_point, data.y)
            pred = end_point.max(1)[1]
            acc = (pred.eq(data.y).sum().item())/len(data.y)
            
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print({'epoch': epoch,'batch': i + 1,'loss': loss.item(),'acc': acc})