from turtle import forward
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x


transform = T.Cartesian(cat=False)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim=3, kernel_size=5):
        super(ConvBlock, self).__init__()

        self.conv = SplineConv(in_channels, out_channels, dim=dim, kernel_size=kernel_size)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, data):
        convd = self.conv(data.x, data.edge_index, data.edge_attr)
        data.x = F.elu(convd)
        data.x = self.bn(data.x)

        return data

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = SplineConv(in_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)
        self.left_conv2 = SplineConv(out_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)
        
        self.shortcut_conv = SplineConv(in_channel, out_channel, dim=3, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)
        
     
    def forward(self, data):
        
        data.x = F.elu(self.left_bn2(self.left_conv2(F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr))),
                                            data.edge_index, data.edge_attr)) + 
                       self.shortcut_bn(self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))
        
        return data


class Net(torch.nn.Module):
    
    def __init__(
        self, 
        n_classes, 
        layer_sizes=[64, 128, 256, 512],
        voxel_sizes=[4, 6, 20, 32]
        ):
        super(Net, self).__init__()
        
        self.res_chunk = torch.nn.ModuleDict({})
        self.layer_sizes = layer_sizes
        self.voxel_sizes = voxel_sizes

        self.conv1 = ConvBlock(1, layer_sizes[0]) #Net.spline_block(1, 64)
        previous = layer_sizes[0]

        for i, layer_size in enumerate(layer_sizes[1:]):
            res_block = ResidualBlock(previous, layer_size)
            self.res_chunk.update({f'res_block{i+1}': res_block})
            previous = layer_size

        self.fc1 = torch.nn.Linear(64 * layer_sizes[-1], 1024)
        self.fc2 = torch.nn.Linear(1024, n_classes)

    def forward(self, data):
        # layer_size, voxel_size
        # 64, 4
        # 128, 6
        # 256, 20
        # 512, 32
        data = self.conv1(data)
        cluster = voxel_grid(data.pos, batch=data.batch, size=2*[self.voxel_sizes[0]])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        iter_layer_params = enumerate(zip(self.layer_sizes[1:], self.voxel_sizes[1:]))
        for i, (layer_size, voxel_size) in iter_layer_params:
            data = self.res_chunk[f'res_block{i+1}'](data)
            cluster = voxel_grid(data.pos, batch=data.batch, size=2*[voxel_size])
            if layer_size == self.layer_sizes[-1]:
                x, _ = max_pool_x(cluster, data.x, data.batch, size=64)
            else:
                data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

