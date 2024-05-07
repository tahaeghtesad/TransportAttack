import torch.nn

from models.agents.rl_agents.attackers.component import ComponentInterface


class GreedyComponent(ComponentInterface):

    def __init__(self, edge_component_mapping):
        super().__init__('GreedyComponent')
        self.edge_component_mapping = edge_component_mapping
        self.n_components = len(edge_component_mapping)
        self.n_edges = sum([len(c) for c in edge_component_mapping])

    def forward(self, states, budgets, allocations, deterministic):
        action = torch.empty((states.shape[0], self.n_edges), device=self.device)
        for c in range(self.n_components):
            action[:, self.edge_component_mapping[c]] = torch.nn.functional.normalize(
                states[:, self.edge_component_mapping[c], 0], p=1, dim=1)
        return action

    # def get_state_dict(self):
    #     return dict(
    #         edge_component_mapping=self.edge_component_mapping,
    #     )
    #
    # @classmethod
    # def from_state_dict(cls, state_dict):
    #     return cls(
    #         edge_component_mapping=state_dict['edge_component_mapping'],
    #     )

    def update(self, states, actions, budgets, allocations, next_states, next_budgets, next_allocations, rewards,
               dones, truncateds):
        return dict()
