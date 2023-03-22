import torch


class GraphConvolutionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, adj, **kwargs):
        super(GraphConvolutionLayer, self).__init__(**kwargs)
        self.register_buffer('adjacency_matrix', torch.tensor(adj))
        self.kernel = torch.nn.parameter.Parameter(torch.zeros((in_features, out_features)))
        self.bias = torch.nn.parameter.Parameter(torch.zeros(size=(out_features,)))
        self.beta = torch.nn.parameter.Parameter(torch.zeros(size=(adj.shape[0], )))

    def forward(self, x):
        diag = torch.diag(torch.reciprocal(torch.sum(self.adjacency_matrix, dim=1) + self.beta))
        adj = self.adjacency_matrix + torch.diag(self.beta)
        return diag @ adj @ x @ self.kernel + self.bias
