import torch

ddpg = torch.load('report_weights/30_budget_ddpg.tar')
high_level = torch.load('report_weights/30_budget_ddpg_allocator_greedy_component.tar')
low_level = torch.load('report_weights/30_budget_proportional_allocator_ddpg_component.tar')
hierarchical = torch.load('report_weights/30_budget_ddpg_allocator_maddpg_component.tar')

print('kir')