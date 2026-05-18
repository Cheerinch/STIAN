import torch

part_groups = {
    'head': [0, 1],
    'torso': [1, 2, 16],
    'left_arm': [2, 4, 6],
    'right_arm': [3, 5, 7],
    'left_leg': [8, 10, 12],
    'right_leg': [9, 11, 13]
}

part_names = list(part_groups.keys())

part_indices_list = [
    torch.tensor(part_groups[name], dtype=torch.long)
    for name in part_names
]

edges_17 = [
    (0,1), (1,2), (1,3), (2,4), (4,6),
    (3,5), (5,7), (1,16), (16,8), (16,9),
    (8,10), (10,12), (12,14),
    (9,11), (11,13), (13,15)
]

edge_index = torch.tensor(
    edges_17,
    dtype=torch.long
).t().contiguous()