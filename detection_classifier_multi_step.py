import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
import scipy
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from transport_env.MultiAgentNetworkEnv import MultiAgentTransportationNetworkEnvironment
from util.torch.gcn import GraphConvolutionLayer


class Classifier(torch.nn.Module):
    def __init__(self, gcn, adj, device):
        super(Classifier, self).__init__()
        self.device = device
        self.register_buffer('adjacency_list', torch.tensor(adj, dtype=torch.long, device=self.device).t().contiguous())
        self.gcn = gcn

        if self.gcn:
            self.model = pyg.nn.Sequential('x, edge_index', [
                (pyg.nn.GCNConv(1, 4), 'x, edge_index -> x'),
                torch.nn.ReLU(),
                (pyg.nn.GCNConv(4, 4), 'x, edge_index -> x'),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(76 * 4, 1),
                torch.nn.Sigmoid()
            ])

        else:
            self.model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(76, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 1),
                torch.nn.Sigmoid()
            )

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        print(self.model)

        self.to(device)

    def forward(self, x):
        if self.gcn:
            return self.model(x, self.adjacency_list)
        else:
            return self.model(x)


class MultiClassClassifier(torch.nn.Module):
    def __init__(self, gcn, adj, device, window_size=5):
        super(MultiClassClassifier, self).__init__()
        self.device = device
        self.window_size = window_size

        self.instance_model = Classifier(gcn, adj, device)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(window_size, 1),
            torch.nn.Sigmoid(),
        )

        self.criterion = torch.nn.HuberLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        print(self.model)

        self.to(device)

    def forward(self, x):
        out = torch.zeros((x.shape[0], self.window_size, 1), device=self.device)
        for i in range(self.window_size):
            out[:, i, :] = self.instance_model(x[:, i, :, :])
        return self.model(torch.squeeze(out))
        

    def update(self, x, y):
        y_pred = self.forward(x)
        _loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        _loss.backward()
        self.optimizer.step()

        return _loss.cpu().detach().numpy()


if __name__ == '__main__':
    device = torch.device('cpu')
    df = pd.read_csv('samples.csv')
    env = MultiAgentTransportationNetworkEnvironment(
        dict(
            n_components=4,
            network=dict(
                method='network_file',
                city='SiouxFalls',
                # city='Anaheim',
            ),
            epsilon=30,
            norm=1,
        )
    )

    adjacency_list = env.get_adjacency_list()
    window_size = 5

    # put x as all columns but last
    # x = torch.unsqueeze(torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32, device=device), dim=2)
    # y = torch.unsqueeze(torch.tensor(df.iloc[:, -1].values, dtype=torch.float32, device=device), dim=1)

    x = df.iloc[:, :-1].values
    y = df.iloc[:, [-1]].values
    train_count = int(x.shape[0] * 0.85)

    x_multi = np.lib.stride_tricks.sliding_window_view(x, (window_size, x.shape[1])).squeeze()
    y_multi = np.lib.stride_tricks.sliding_window_view(y, (window_size, y.shape[1])).squeeze()
    y_multi = np.max(y_multi, axis=1, keepdims=True)

    x_multi = torch.tensor(x_multi, dtype=torch.float32, device=device)
    y_multi = torch.tensor(y_multi, dtype=torch.float32, device=device)

    x_train = x_multi[:train_count, :, :, np.newaxis]
    y_train = y_multi[:train_count, :]
    x_test = x_multi[train_count:, :, :, np.newaxis]
    y_test = y_multi[train_count:, :]

    classifier = MultiClassClassifier(gcn=True, adj=adjacency_list, device=device, window_size=window_size)
    data = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=128)

    try:
        for step in (pbar := tqdm(range(128))):
            for batch, (x_batch, y_batch) in enumerate(data):
                loss = classifier.update(x_batch, y_batch)

            y_pred = classifier.forward(x_train)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_train.cpu().detach().numpy().astype(np.int32)
            fpr, tpr, thres = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            pbar.set_description(f'loss: {loss:.4f} - auc: {roc_auc:.4f}')

    except KeyboardInterrupt:
        print('interrupted by user... evaluating model...')

    # evaluate the model on the last 10% of the data
    y_pred = classifier.forward(x_test)
    loss = classifier.criterion(y_pred, y_test)
    print(f'loss: {loss:.4f}')

    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_test.cpu().detach().numpy().astype(np.int32)
    fpr, tpr, thres = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.figtext(0.5, 0.01, f'loss: {loss:.4f}', wrap=True, horizontalalignment='center', fontsize=12)
    # save figure as timestamped png file
    plt.savefig(f'roc_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}_multi.png')
    plt.show()

