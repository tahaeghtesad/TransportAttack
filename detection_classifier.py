import numpy as np
import pandas as pd
import tikzplotlib
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.torch.classifier import Classifier

if __name__ == '__main__':
    device = torch.device('cpu')
    df = pd.read_csv('samples.csv')

    x = df.iloc[:, :-1].values
    y = df.iloc[:, [-1]].values
    train_count = int(x.shape[0] * 0.85)

    classifier = Classifier(dict(classifier_config=dict(lr=0.001, threshold=0.99)), device, (76, 1))

    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    x_train = x[:train_count, :, np.newaxis]
    y_train = y[:train_count, :]
    x_test = x[train_count:, :, np.newaxis]
    y_test = y[train_count:, :]

    data = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=128)

    try:

        for step in (pbar := tqdm(range(128))):
            for batch, (x_batch, y_batch) in enumerate(data):
                loss = classifier.update(x_batch, y_batch)

            y_pred = classifier.model.forward(x_train)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_train.cpu().detach().numpy().astype(np.int32)
            fpr, tpr, thres = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            pbar.set_description(f'loss: {loss:.4f} - auc: {roc_auc:.4f}')


    except KeyboardInterrupt:
        print('interrupted by user... evaluating model...')

    # evaluate the model on the last 10% of the data
    y_pred = classifier.model.forward(x_test)
    loss = classifier.criterion(y_pred, y_test)
    print(f'loss: {loss:.4f}')

    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_test.cpu().detach().numpy().astype(np.int32)
    fpr, tpr, thres = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Detector ROC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r-')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # tikzplotlib.save('roc.tikz')
    plt.show()