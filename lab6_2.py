import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import time

rows = 10
cols = 20
obstacle_num = 40

indices_flat = list(range(rows*cols))
grid_flat = np.zeros((rows*cols,))
obstacles = random.sample(indices_flat, obstacle_num)
for i in obstacles:
    grid_flat[i] = 1
grid = grid_flat.reshape(rows, cols)
print(grid)

#building a graph in networkx
G = nx.Graph()
for i in range(rows):
    for j in range(cols):
        if grid[i][j] != 1:
            G.add_node(f"({i},{j})", pos=(i,j))
            if (i + 1) < rows and grid[i+1][j] != 1:
                G.add_edge(f"({i+1},{j})", f"({i},{j})")
            if (j + 1) < cols and grid[i][j+1] != 1:
                G.add_edge(f"({i},{j+1})", f"({i},{j})")


def path_vert2edge(path):
    edge_path = []
    for ind, item in enumerate(path[:-1]):
        edge_path.append((item, path[ind+1]))
    return edge_path


if __name__ == "__main__":
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_shape='s', node_size=150, node_color="skyblue", with_labels=False)
    random_nodes = list(random.sample(list(G.nodes), 2))
    nx.draw_networkx_nodes(G, pos, node_shape='s', node_size=150, node_color='orange', nodelist=random_nodes)
    
    short_p_nodes = nx.shortest_path(G, source=random_nodes[0], target=random_nodes[1])
    short_p_edges = path_vert2edge(short_p_nodes)

    nx.draw_networkx_nodes(G, pos, node_shape='s', node_size=150, node_color='sandybrown', nodelist=short_p_nodes)
    nx.draw_networkx_edges(G, pos, edgelist=short_p_edges, edge_color='peru')

    plt.show()
