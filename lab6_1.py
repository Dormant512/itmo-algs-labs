import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import time

vert_num = 100
edge_num = 500
max_weight = 20

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
            ans = np.random.randint(max_weight) + 1
            G_adj_matrix[i][j] = ans
            G_adj_matrix[j][i] = ans
            G.add_edge(i, j, weight=ans)
        k += 1

def path_vert2edge(path):
    edge_path = []
    for ind, item in enumerate(path[:-1]):
        edge_path.append((item, path[ind+1]))
    return edge_path


if __name__ == "__main__":
    src, trg = np.random.randint(vert_num), np.random.randint(vert_num) #random start and end nodes
    other_verts = [i for i in list(G.nodes) if i != src]
    assert len(other_verts) == vert_num - 1, f"Length of other vertices {len(other_verts)} (wrong)!"

    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos, font_weight='bold', node_color='skyblue', edge_color='steelblue')
    nx.draw_networkx_nodes(G, pos, nodelist=[src, trg], node_color='tomato')
    labels = nx.get_edge_attributes(G, 'weight')

    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    dijkstra_times = []
    for i in range(10):
        start_time = time.time()
        for j in other_verts:
            dijkstra_p = nx.shortest_path(G, source=src, target=j, weight='weight', method='dijkstra')
        delta_time = time.time() - start_time
        dijkstra_times.append(delta_time)

    bellman_ford_times = []
    for i in range(10):
        start_time = time.time()
        for j in other_verts:
            bellman_ford_p = nx.shortest_path(G, source=src, target=j, weight='weight', method='bellman-ford')
        delta_time = time.time() - start_time
        bellman_ford_times.append(delta_time)
        
    print(f"Source:            {src}")
    print("Dijkstra time:     {:.4f}s".format(np.mean(dijkstra_times)))
    print("Bellman-Ford time: {:.4f}s".format(np.mean(bellman_ford_times)))

    #visualize one path
    dijkstra_vis = nx.shortest_path(G, source=src, target=trg, weight='weight', method='dijkstra')
    path_edges = path_vert2edge(dijkstra_vis)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='firebrick', width=1.5)

    costs = [int(G[e[0]][e[1]]['weight']) for e in path_edges]
    print(f"\nVisualized path\nPath: {dijkstra_vis}    Path costs: {costs}    Sum: {sum(costs)}")

    plt.show()
