import numpy as np
import os
import pandas as pd


def import_od(matfile):
    f = open(matfile, 'r')
    all_rows = f.read()
    blocks = all_rows.split('Origin')[1:]
    matrix = {}
    for k in range(len(blocks)):
        orig = blocks[k].split('\n')
        dests = orig[1:]
        orig = int(orig[0])

        d = [eval('{'+a.replace(';',',').replace(' ','') +'}') for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[orig] = destinations

    od_data = []
    for orig in matrix:
        for dest in matrix[orig]:
            if matrix[orig][dest] > 0:
                od_data.append([orig, dest, int(matrix[orig][dest])])

    return od_data


def load_net(path):
    net = pd.read_csv(path, skiprows=8, sep='\t')

    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed

    # And drop the silly first andlast columns
    net.drop(['~', ';'], axis=1, inplace=True)
    # net.head()
    # net.columns

    # net_data = net.to_numpy()

    # link_data = np.array([net_data[:, 0], net_data[:, 1], net_data[:, 4],
    #                       np.ones_like(net_data[:, 4]), net_data[:, 2]]).T

    return net