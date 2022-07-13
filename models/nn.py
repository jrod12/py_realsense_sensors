from torch import nn
import config as cfg
import sys
if sys.platform != 'win32':
    import spconv
else:
    raise ValueError('cannot run sparse network on Windows')
import torch
import utils
from models.regressor import Regressor
import numpy as np
torch.set_num_threads(cfg.TORCH_NUM_THREADS)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.net = spconv.SparseSequential(

            spconv.SparseConv3d(1, 5, (3, 3, 3), 1),
            spconv.SparseMaxPool3d(2, 2, 2),

            spconv.SparseConv3d(5, 5, (3, 3, 3), 1),
            spconv.SparseMaxPool3d(2, 2, 2),

            spconv.SparseConv3d(5, 5, (3, 3, 3), 1),
            spconv.SparseMaxPool3d(2, 2, 2),

            spconv.SparseConv3d(5, 5, (3, 3, 3), 1),
            spconv.SparseMaxPool3d(2, 2, 2),

            spconv.SparseConv3d(5, 5, (3, 3, 3), 1),
            spconv.SparseMaxPool3d(2, 2, 2),

            spconv.SparseConv3d(5, 5, (2, 2, 2), 1),
            spconv.SparseMaxPool3d(2, 2, 2),

            spconv.SparseConv3d(5, 5, (2, 2, 2), 1),
            spconv.SparseMaxPool3d(2, 2, 2),

            spconv.ToDense(),
        )

        self.fc1 = nn.Linear(4160, 30)
        self.fc2 = nn.ELU(alpha=1e-5)
        self.fc3 = nn.Linear(30, 3)

    def forward(self, x: torch.Tensor):
        x_sp_old = spconv.SparseConvTensor(x._values(),
                                           torch.transpose((x._indices()), 1, 0).to(
                                               dtype=torch.int32).contiguous().detach().clone(),
                                           x.shape[1:-1],
                                           x.shape[0])

        x = self.net(x_sp_old)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)

        return output


class ConvNetRegressor(Regressor):
    def __init__(self):
        if cfg.CAMERA == 'left':
            path = 'models/conv_net_model_18_epochs_LEFT.ckpt'
        elif cfg.CAMERA == 'right':
            path = 'models/conv_net_model_13_epochs_RIGHT.ckpt'
        else:
            raise ValueError('Incorrect camera name [config.ini]')
        self.regressor = torch.load(path)
        self.data_transformation = utils.DataTransformation()

    def predict(self, x):
        x = self.data_transformation.sparse_points(x)
        x = x.reshape((1,)+x.shape+(1,))
        i_train = torch.LongTensor(torch.from_numpy(x.coords[0:-1, :]))
        val_train = torch.FloatTensor(torch.from_numpy(np.ones((i_train.shape[1], 1), dtype=np.single)))

        x = torch.sparse.LongTensor(i_train, val_train, (
            1, x.shape[1], x.shape[2], x.shape[3], 1))

        return self.regressor.forward(x).detach().numpy()


