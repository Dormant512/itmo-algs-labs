import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

vert_num = 100
edge_num = 200

#random adjacency generation
G_adj_matrix = np.zeros((vert_num, vert_num))
edge_list = [x for x in range(sum(range(vert_num)))]
edge_exist_indices = random.sample(edge_list, edge_num)
edge_binary_flat = []
for i in edge_list:
    if i in edge_exist_indices:
        edge_binary_flat.append(1)
    else:
        edge_binary_flat.append(0)

#building a graph in networkx
G = nx.Graph()
G.add_nodes_from(range(vert_num))

#building a symmetrical adjacency matrix
k = 0 #index for existing edges
for i in range(vert_num):
    for j in range(i+1, vert_num):
        if edge_binary_flat[k] == 1:
            G_adj_matrix[i][j] = 1.0
            G_adj_matrix[j][i] = 1.0
            G.add_edge(i, j)
        k += 1


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


def connected_components(graph):
    """Function that finds connected components"""
    done = set()
    for i in range(len(graph)):
        if i not in done:
            done.add(i)
            component = []
            que = deque([i])
            while que:
                node = que.popleft()
                component.append(node)
                for i in graph[node]:
                    if i not in done:
                        done.add(i)
                        que.append(i)
            yield component

def path_painter(path):
    """Function that is required to draw the shortest path"""
    edge_path = []
    for ind, item in enumerate(path[:-1]):
        edge_path.append((item, path[ind+1]))
    return edge_path

pos = nx.circular_layout(G)

if __name__ == "__main__":
    print("\nSeveral rows of the adjacency matrix:\n", G_adj_matrix[0:3])
    a_l = adj_matrix2list(G_adj_matrix)

    print("\nAdjacency list:\n",a_l[0:3])

    print("\nConnected components of the graph (standalone vertices have no connections):\n",list(connected_components(G)))

    #getting random points, calculating the shortest path and its length
        #it is necessary to add 1 to the calculated length due to ignoring the starting point in the calculation
    source,target = np.random.randint(vert_num), np.random.randint(vert_num)
    path=nx.shortest_path(G, source=source, target=target)
    path_len=nx.shortest_path_length(G, source=source, target=target)+1

    print("\nShortest path between random point",source, "and random point", target, "consists of", path_len, "points:", path,'\n')

    nx.draw_networkx(G, pos, font_weight='bold', font_size=8, node_size=200, node_color='skyblue', edge_color='steelblue')
    nx.draw_networkx_edges(G, pos, edgelist=path_painter(path), edge_color='firebrick', width=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[source, target], node_size=200, node_color='tomato')
    plt.show()