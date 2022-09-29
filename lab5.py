import networkx as nx
import numpy as np
import random

vert_num = 100
edge_num = 200

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
            G_adj_matrix[i][j] = 1.0
            G_adj_matrix[j][i] = 1.0
        k += 1
        j += 1
    i += 1


def adj_matrix2list(adj_matrix, is_dict=False):
    """Function that turns adjacency matrix to adjacency list (list/dict of lists)."""
    sh = adj_matrix.shape
    assert sh[0] == sh[1], f"Rows ({sh[0]}) != cols ({sh[1]}), adj_matrix not square."
    if is_dict:
        adj_list = {}
    else:
        adj_list = []
    for i in range(sh[0]):
        neighbors = []
        for ind, j in enumerate(adj_matrix[i]):
            if float(j) != 0:
                neighbors.append(ind)
        if is_dict:
            adj_list.update({f"{i}": neighbors})
        else:
            adj_list.append(neighbors)
    return adj_list


if __name__ == "__main__":
    print(G_adj_matrix[0:3])

    a_l = adj_matrix2list(G_adj_matrix)
    print(a_l[0:3])
