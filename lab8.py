import networkx as nx
from networkx.algorithms import tree
import numpy as np
import matplotlib.pyplot as plt
import random
import time


def gen_graph(v=100, e=500, max_w=20):
    """Generates a weighted graph of V vertices and E edges of max weight max_w."""
    #random adjacency generation
    edge_list = [x for x in range(sum(range(v)))]
    edge_exist_indices = random.sample(edge_list, e)
    edge_binary_flat = []
    for i in edge_list:
        if i in edge_exist_indices:
            edge_binary_flat.append(1)
        else:
            edge_binary_flat.append(0)

    #building a graph in networkx
    G = nx.Graph()
    G.add_nodes_from(range(v))

    k = 0 #index for existing edges
    for i in range(v):
        for j in range(i+1, v):
            if edge_binary_flat[k] == 1:
                ans = np.random.randint(max_w) + 1
                G.add_edge(i, j, weight=ans)
            k += 1

    return G


if __name__ == "__main__":
    
    points = 20
    reps = 5
    start_V, end_V = 50, 350
    start_E = 5*start_V

    connected_max_edges = int(start_V*(start_V-1)/2)
    step_V = int((end_V - start_V) / points)
    step_E = int((connected_max_edges - start_E) / points)

    k = 1

    vert_list = list(range(start_V, end_V, step_V))
    edge_list = list(range(start_E, connected_max_edges, step_E))
    print(f"Maximums:    V = {vert_list[-1]}    E = {edge_list[-1]}")

    method = input("'k' for Kruskal, 'p' for prim: ").strip().lower()

    T_kruskal = np.zeros((len(vert_list), len(edge_list)))
    T_prim = np.zeros((len(vert_list), len(edge_list)))

    elapsed_start = time.time()
    for i, verts in enumerate(vert_list):
        for j, edges in enumerate(edge_list):
            delta_kruskal = []
            delta_prim = []
            for times in range(reps):
                graph = gen_graph(verts, edges)

                start = time.time()
                if method == 'k':
                    mst_kruskal = tree.minimum_spanning_edges(graph, algorithm="kruskal")
                    delta_kruskal.append(time.time() - start)
                else:
                    mst_prim = tree.minimum_spanning_edges(graph, algorithm="prim")
                    delta_prim.append(time.time() - start)

            if method == 'k':
                T_kruskal[i, j] = np.mean(delta_kruskal)*1000
            else:
                T_prim[i, j] = np.mean(delta_prim)*1000

            print(f"Point {k}/{len(vert_list) * len(edge_list)}. V={verts}, E={edges}.", end='\r')

            k += 1

    elapsed_end = time.time()
    print(f"Elapsed time: {elapsed_end - elapsed_start}")

    # https://numpy.org/doc/stable/reference/generated/numpy.save.html

    V, E = np.meshgrid(vert_list, edge_list)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(V.T, E.T, T_kruskal, 50, cmap='binary')
    ax.set_xlabel('V')
    ax.set_ylabel('E')
    ax.set_zlabel('T')
    plt.show()
