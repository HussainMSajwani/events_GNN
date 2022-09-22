import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x
from torch_geometric.data import Data


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
        
        print('res forward', data)
        return data


#https://github.com/uzh-rpg/aegnn/blob/master/aegnn/models/layer/max_pool.py
class MaxPooling(torch.nn.Module):

    def __init__(self, voxel_size, transform = None):
        super(MaxPooling, self).__init__()
        self.voxel_size = voxel_size
        self.transform = transform

    def forward(self, x, pos, batch = None, edge_index = None, return_data_obj = True):
        assert edge_index is not None, "edge_index must not be None"

        cluster = voxel_grid(pos, batch=batch, size=self.voxel_size)
        data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch).to('cuda')
        data = max_pool(cluster, data=data, transform=self.transform)  # transform for new edge attributes
        if return_data_obj:
            return data
        else:
            return data.x, data.pos, getattr(data, "batch"), data.edge_index, data.edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size})"


#https://github.com/uzh-rpg/aegnn/blob/master/aegnn/models/layer/max_pool_x.py
class MaxPoolingX(torch.nn.Module):

    def __init__(self, voxel_size, size: int):
        super(MaxPoolingX, self).__init__()
        self.voxel_size = voxel_size
        self.size = size

    def forward(self, x, pos, batch = None):
        cluster = voxel_grid(pos, batch=batch, size=self.voxel_size)
        x, _ = max_pool_x(cluster, x, batch, size=self.size)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"

class ResGNet(torch.nn.Module):
    
    def __init__(
        self, 
        n_classes, 
        layer_sizes=[64, 128, 256, 512],
        voxel_sizes=[20, 30, 50, 80]
        ):
        super(ResGNet, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.voxel_sizes = voxel_sizes

        self.n_graph_layers = len(layer_sizes)

        self.conv_block1 = ConvBlock(1, layer_sizes[0]) #Net.spline_block(1, 64)
        self.pool_block1 = MaxPooling(voxel_size=[0])

        self.res_blocks = torch.nn.ModuleDict({})
        self.pool_blocks = torch.nn.ModuleDict({})

        for layer in range(1, self.n_graph_layers):
            res_block = ResidualBlock(
                in_channel = self.layer_sizes[layer - 1], 
                out_channel = self.layer_sizes[layer]
                )

            self.res_blocks.update({f'res_block{layer+1}': res_block})

            if not layer+1 == self.n_graph_layers:
                pool_block = MaxPooling(
                    voxel_size = self.voxel_sizes[layer], 
                    transform = T.Cartesian(cat=False)
                    )
            else:
                pool_block = MaxPoolingX(self.voxel_sizes[layer], 64)

            self.pool_blocks.update({f'pool_block{layer+1}': pool_block})

        self.fc1 = torch.nn.Linear(64 * layer_sizes[-1], 1024)
        self.fc2 = torch.nn.Linear(1024, n_classes)

    def forward(self, data):
        data = self.conv_block1(data)
        data = self.pool_block1(data.x, data.pos, data.batch, data.edge_index)

        for layer in range(1, self.n_graph_layers):
            data = self.res_blocks[f'res_block{layer+1}'](data)
            if not layer+1 == self.n_graph_layers:
                data = self.pool_blocks[f'pool_block{layer+1}'](data.x, data.pos, data.batch, data.edge_index)
            else:
                x = self.pool_blocks[f'pool_block{layer+1}'](data.x, data.pos, data.batch)

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net(torch.nn.Module):
    def __init__(
        self, 
        n_classes, 
        layer_sizes=[64, 128, 256, 512],
        voxel_sizes=[20, 30, 50, 80]
        ):
        super(Net, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.voxel_sizes = voxel_sizes

        self.n_graph_layers = len(layer_sizes)

        self.conv_blocks = torch.nn.ModuleDict({})
        self.pool_blocks = torch.nn.ModuleDict({})

        for layer in range(self.n_graph_layers):
            in_channel = self.layer_sizes[layer-1] if not layer == 0 else 1
            conv_block = ConvBlock(
                in_channels=in_channel,
                out_channels=self.layer_sizes[layer]
            )

            self.conv_blocks.update({f'conv_block{layer+1}': conv_block})

            if not layer+1 == self.n_graph_layers:
                pool_block = MaxPooling(
                    voxel_size = self.voxel_sizes[layer], 
                    transform = T.Cartesian(cat=False)
                    )
            else:
                pool_block = MaxPoolingX(self.voxel_sizes[layer], 64)

            self.pool_blocks.update({f'pool_block{layer+1}': pool_block})

        self.fc1 = torch.nn.Linear(64 * layer_sizes[-1], 1024)
        self.fc2 = torch.nn.Linear(1024, n_classes)

    def forward(self, data):

        for layer in range(self.n_graph_layers):
            data = self.conv_blocks[f'conv_block{layer+1}'](data)
            if not layer+1 == self.n_graph_layers:
                data = self.pool_blocks[f'pool_block{layer+1}'](data.x, data.pos, data.batch, data.edge_index)
            else:
                x = self.pool_blocks[f'pool_block{layer+1}'](data.x, data.pos, data.batch)

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class og_Net(torch.nn.Module):
    def __init__(self):
        super(og_Net, self).__init__()
        self.conv1 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = SplineConv(64, 128, dim=3, kernel_size=5)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.conv3 = SplineConv(128, 256, dim=3, kernel_size=5)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.conv4 = SplineConv(256, 512, dim=3, kernel_size=5)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.fc1 = torch.nn.Linear(64 * 512, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn1(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=20)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn2(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=30)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn3(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=50)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.bn4(data.x)
        cluster = voxel_grid(data.pos, batch=data.batch, size=100)
        x, batch = max_pool_x(cluster, data.x, batch=data.batch, size=64)

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


