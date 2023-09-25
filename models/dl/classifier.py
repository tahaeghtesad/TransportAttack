import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class Classifier(torch.nn.Module):
    def __init__(self, config, device, observation_space_shape):
        super(Classifier, self).__init__()
        self.device = device
        self.config = config

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(observation_space_shape), 76 * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(76 * 4, 1),
            torch.nn.Sigmoid()
        )

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['classifier_config']['lr'])

        self.to(device)

    def forward_single(self, edge_travel_time):
        x = torch.unsqueeze(torch.tensor(edge_travel_time, dtype=torch.float32).to(self.device), dim=0)
        return torch.gt(self.model(x), self.config['classifier_config']['threshold']).cpu().data.numpy()[0][0]

    def forward(self, edge_travel_times):
        x = torch.tensor(edge_travel_times, dtype=torch.float32).to(self.device)
        return torch.squeeze(self.model(x)).cpu().data.numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def __update(self, edge_travel_times, labels):
        y_pred = self.model.forward(edge_travel_times)
        _loss = self.criterion(y_pred, labels)
        self.optimizer.zero_grad()
        _loss.backward()
        self.optimizer.step()

        return _loss.cpu().detach().numpy().item()

    def update_batched(self, edge_travel_times, labels):
        data = DataLoader(
            TensorDataset(
                torch.from_numpy(np.array(edge_travel_times, dtype=np.float32)).to(self.device),
                torch.from_numpy(np.array(labels, dtype=np.float32)[:, np.newaxis]).to(self.device)
            ), batch_size=self.config['classifier_config']['batch_size'], shuffle=False
        )
        for _ in range(self.config['classifier_config']['training_epochs']):
            losses = []

            for x, y_labels in data:
                loss = self.__update(x, y_labels)
                losses.append(loss)

            yield dict(
                mean=np.mean(losses),
                std=np.std(losses),
                min=np.min(losses),
                max=np.max(losses)
            )
