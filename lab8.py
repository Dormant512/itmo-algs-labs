from pathlib import Path
import networkx as nx
from networkx.algorithms import tree
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import random
import time


def time_complexity(v_e, a, b, c, d):
    """Time complexity for Kruskal's algorithm and Prim's algorithm (with binary heap, NetworkX)."""
    ans = a + v_e[1] * b * np.log2(v_e[0]*c + d)
    return ans


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
    

    def main():
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

        raw_kruskal = []
        raw_prim = []
        
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
                    T_kruskal[i,j] = np.mean(delta_kruskal)*1000
                    raw_kruskal.append((verts, edges, np.mean(delta_kruskal)*1000))
                else:
                    T_prim[i,j] = np.mean(delta_prim)*1000
                    raw_prim.append((verts, edges, np.mean(delta_prim)*1000))

                print(f"Point {k}/{len(vert_list) * len(edge_list)}. V={verts}, E={edges}.", end='\r')

                k += 1

        elapsed_end = time.time()
        print(f"Elapsed time: {elapsed_end - elapsed_start}")

        if len(raw_kruskal) > 0:
            raw_data, T = raw_kruskal, T_kruskal 
        else:
            raw_data, T = raw_prim, T_prim
        
        V, E = np.meshgrid(vert_list, edge_list)
        V, E = V.T, E.T

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(V, E, T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('V')
        ax.set_ylabel('E')
        ax.set_zlabel('T')
        plt.show()

        save = input("Save those values (y/n)? ").strip().lower()
        if save == 'y':
            np.savez(filename, V=V, E=E, T=T, raw_data=raw_data)


    method = input("'k' for Kruskal, 'p' for prim: ").strip().lower()

    if method == 'k': 
        filename = 'kruskal'
    else:
        filename = 'prim'

    try:
        npzfile = np.load(f"./{filename}.npz")
        V, E, T, raw_data = npzfile['V'], npzfile['E'], npzfile['T'], npzfile['raw_data']
        print(f"File {filename} exists, importing data from it.")

        # format V, E and T for curve fitting
        V_raw, E_raw, T_raw = [], [], []
        for row in raw_data:
            V_raw.append(row[0])
            E_raw.append(row[1])
            T_raw.append(row[2])

        params, covariance = curve_fit(time_complexity, [V_raw, E_raw], T_raw)
        T_fit = np.zeros(T.shape)
        
        for i in range(T_fit.shape[0]):
            for j in range(T_fit.shape[1]):
                T_fit[i,j] = time_complexity(np.array([V[i,j], E[i,j]]), *params)

        print(f"{params[0]:.4f} + E * {params[1]:.6f} * ln(V * {params[2]:.4f} + {params[3]:.4f})")

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        ax.plot_surface(V, E, T, cmap=cm.viridis, linewidth=0.5, antialiased=False, alpha=0.1)
        ax.scatter(V_raw, E_raw, T_raw, color='black')
        ax.plot_surface(V, E, T_fit, cmap=cm.Reds, linewidth=0.1, antialiased=False, alpha=0.5)

        ax.set_xlabel('V')
        ax.set_ylabel('E')
        ax.set_zlabel('T')
        plt.show()
    except:
        print(f"{filename}.npz does not exist, generating from scratch.")
        main()
