import torch

from models import CustomModule


class GraphConvolutionLayer(CustomModule):
    def __init__(self, in_features, out_features, adj):
        super(GraphConvolutionLayer, self).__init__('GraphConvolutionLayer')
        self.register_buffer('adjacency_matrix', torch.tensor(adj))
        self.kernel = torch.nn.parameter.Parameter(torch.zeros((in_features, out_features)), requires_grad=True)
        self.bias = torch.nn.parameter.Parameter(torch.zeros(size=(out_features,)), requires_grad=True)
        self.beta = torch.nn.parameter.Parameter(torch.zeros(size=(adj.shape[0], )), requires_grad=True)

        torch.nn.init.xavier_uniform_(self.kernel)
        torch.nn.init.uniform_(self.bias)
        torch.nn.init.uniform_(self.beta)

    def forward(self, x):
        diag = torch.diag(torch.reciprocal(torch.sum(self.adjacency_matrix, dim=1) + self.beta))
        adj = self.adjacency_matrix + torch.diag(self.beta)
        return diag @ adj @ x @ self.kernel + self.bias

    def extra_repr(self):
        return f'in_features={self.kernel.shape[0]}, out_features={self.kernel.shape[1]}'


class GraphConvolutionResidualBlock(CustomModule):
    def __init__(self, num_features, adj, activation, depth):
        super(GraphConvolutionResidualBlock, self).__init__('GraphConvolutionResidualBlock')
        assert depth > 1, 'Residual Block\'s depth must be greater than 1'

        self.layers = torch.nn.ModuleList([
            GraphConvolutionLayer(num_features, num_features, adj)
            for _ in range(depth)
        ])

        self.activation = getattr(torch.nn.functional, activation)

    def forward(self, x):
        input_ = x
        for l in self.layers:
            x = self.activation(l(x))
        return x + input_

    def extra_repr(self):
        return f'num_features={self.layers[0].kernel.shape[0]}, depth={len(self.layers)}'
