import networkx as nx
import numpy as np
import random

vert_num = 100
edge_num = 500
max_weight = 20

np.random.seed(0)

#random adjacency generation
G_adj_matrix = np.eye(vert_num)
edge_list = [x for x in range(sum(range(vert_num)))]
edge_exist_indices = random.sample(edge_list, edge_num)
edge_binary_flat = []
for i in edge_list:
    if i in edge_exist_indices:
        edge_binary_flat.append(1)
    else:
        edge_binary_flat.append(0)

#building a symmetrical adjacency matrix
i = 0 #row index
k = 0 #index for existing edges
while i < vert_num:
    j = i + 1 #col index
    while j < vert_num:
        if edge_binary_flat[k] == 1:
            ans = np.random.randint(max_weight) + 1
            G_adj_matrix[i][j] = ans
            G_adj_matrix[j][i] = ans
        k += 1
        j += 1
    i += 1

if __name__ == "__main__":
    print(G_adj_matrix[0:3])
