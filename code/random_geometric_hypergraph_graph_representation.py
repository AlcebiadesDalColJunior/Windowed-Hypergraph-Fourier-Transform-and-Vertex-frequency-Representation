import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import gen_hypergraphs
import tensor_operations


# V, E = gen_hypergraphs.path_hypergraph(7)
# name = "path_hypergraph7_gr"
# seed = 2

# V, E = gen_hypergraphs.path_hypergraph(7)
# name = "path_hypergraph7b_gr"
# seed = 2
# E.append([1,2])

# V, E = gen_hypergraphs.path_hypergraph(21)
# name = "path_hypergraph21_gr"
# seed = 4

# V, E = gen_hypergraphs.path_hypergraph(181)
# name = "path_hypergraph181_gr"
# seed = 0

# V, E = gen_hypergraphs.cyclic_hypergraph(24)
# name = "cycle_hypergraph24_gr"
# seed = 2

# V, E = gen_hypergraphs.squid_hypergraph(13) # or 40
# name = "squid_hypergraph13_gr"
# seed = 2

V, E, pos = gen_hypergraphs.random_geometric_hypergraph(64, 0.20, seed=4155)
name = "random_geometric_hypergraph_gr"
seed = 2

# V, E = gen_hypergraphs.path_hypergraph_order4(70)
# name = "path_hypergraph70_order4_gr"
# seed = 2

shifting = False
GFT = False
GFT_heatmap = False
iGFT = False
WGFT = True; plot_window = False

spectral_clustering = True; n_clusters = 3

translation = False; new_node = 4
modulation = False; new_module = 7

# signal = "delta"
# signal = "eigenvector"
signal = "eigenvectors"
# signal = "exponential"

if signal == "eigenvector":
    eigenvector = 2

if signal == "delta":
    center = 1

if name in ["path_hypergraph7_gr","path_hypergraph7b_gr","path_hypergraph21_gr",
            "cycle_hypergraph24_gr"]:
    tau = 0.5
    tau_x = 0.5

if name == "path_hypergraph181_gr":
    tau = 300
    tau_x = 1

if name == "squid_hypergraph_gr":
    tau = 2
    tau_x = 2

if name == "path_hypergraph70_order4_gr":
    tau = 10

if name == "random_geometric_hypergraph_gr":
    tau = 3

G = tensor_operations.graph_representation(V,E)

if name in ["path_hypergraph7_gr","path_hypergraph7b_gr","path_hypergraph21_gr",
            "cycle_hypergraph24_gr","squid_hypergraph13_gr"]:
    pos = nx.spring_layout(G, seed=seed)

if name in ["path_hypergraph181_gr","path_hypergraph70_order4_gr"]:
    pos = nx.spiral_layout(G)

start_time = time.process_time()

N = len(V)

L = nx.laplacian_matrix(G)
L = L.toarray()

eigenvalues, eigenvectors = np.linalg.eig(L)

order = np.argsort(eigenvalues)


#%% Graph Spectral Clustering

if name == "random_geometric_hypergraph_gr" and signal == "eigenvectors":
    spectral_clustering = True

if spectral_clustering:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(eigenvectors[:,order[:n_clusters]])
    
    kmeans_labels = kmeans.labels_
    
    if n_clusters == 3:
        set0 = list(np.where(np.array(kmeans_labels) == 0)[0])
        set1 = list(np.where(np.array(kmeans_labels) == 1)[0])
        set2 = list(np.where(np.array(kmeans_labels) == 2)[0])
        
        sets = [set0, set1, set2]
        
        sorted_sets = sorted(sets, key=len)
        
        set0, set1, set2 = sorted_sets
        
        for i in set0:
            kmeans_labels[i] = 0
        for i in set1:
            kmeans_labels[i] = 1
        for i in set2:
            kmeans_labels[i] = 2
    
    plt.figure()
    nx.draw_networkx(G,pos,node_color=kmeans_labels,node_size=100,vmin=0,vmax=8,
                      with_labels=False,cmap="Set1")
    plt.savefig("results/"+name+"_spectral_clustering"+str(n_clusters)+".pdf", bbox_inches='tight')
    plt.show()

#%% Signal

x = np.zeros((N,))

if signal == "delta":
    x[center-1] = 1

if signal == "eigenvector":
    x = eigenvectors[:,order[eigenvector-1]]

if signal == "eigenvectors":
    if name == "path_hypergraph181_gr":
        x[:60] = eigenvectors[:60,order[10]]
        x[60:120] = eigenvectors[60:120,order[60]]
        x[120:] = eigenvectors[120:,order[30]]
    
    if name == "path_hypergraph70_order4_gr":
        x[:20] = eigenvectors[:20,order[10]]
        x[20:40] = eigenvectors[20:40,order[27]]
        x[40:] = eigenvectors[40:,order[5]]
    
    if name == "random_geometric_hypergraph_gr":
        if n_clusters == 3:
            for i in set0:
                x[i] = eigenvectors[i,order[10]]
            for i in set1:
                x[i] = eigenvectors[i,order[27]]
            for i in set2:
                x[i] = eigenvectors[i,order[5]]

if signal == "exponential":
    x_hat = np.zeros((N,))
    for l in range(N):
        x_hat[l] = np.exp(-tau_x*eigenvalues[order[l]])
    
    x = np.zeros((N,))
    for n in range(N):
        for l in range(N):
            x[n] += x_hat[l] * eigenvectors[n,order[l]]

if signal == "delta":
    vmax = 1
    vmin = -1

if signal in ["eigenvector","eigenvectors"]:
    vmin = np.min(eigenvectors)
    vmax = np.max(eigenvectors)
    
    vmax = max(np.abs(vmin),vmax)
    vmin = - vmax

if signal == "exponential":
    vmin = np.min(x)
    vmax = np.max(x)
    
    vmax = max(np.abs(vmin),vmax)
    vmin = - vmax

cmap = "seismic"

plt.figure()
nx.draw_networkx(G,pos,node_color=x,node_size=100,vmin=vmin,vmax=vmax,with_labels=False, cmap=cmap)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
sm._A = []
plt.colorbar(sm, ax=plt.gca())
if name == "path_hypergraph181_gr":
    plt.axis("equal")
if signal == "eigenvector":
    plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+".pdf", bbox_inches='tight')
if signal == "eigenvectors":
    plt.savefig("results/"+name+"_eigenvectors.pdf", bbox_inches='tight')
if signal == "delta":
    plt.savefig("results/"+name+"_delta"+str(center)+".pdf", bbox_inches='tight')
if signal == "exponential":
    plt.savefig("results/"+name+"_exponential.pdf", bbox_inches='tight')
plt.show()


#%% Shifting Operator

if shifting:
    x_shifted = L @ x
    
    vmin_shifted = np.min(x_shifted)
    vmax_shifted = np.max(x_shifted)
    
    vmax_shifted = max(np.abs(vmin_shifted),vmax_shifted)
    vmin_shifted = - vmax_shifted
    
    plt.figure()
    nx.draw_networkx(G,pos,node_color=x_shifted,node_size=100,vmin=vmin_shifted,vmax=vmax_shifted,with_labels=False, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_shifted,vmax=vmax_shifted))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_shifted.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_shifted.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_shifted.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_shifted.pdf", bbox_inches='tight')
    plt.show()


#%% Graph Fourier transform

if translation or modulation:
    GFT = True

if GFT:
    x_hat = np.zeros((N,))
    
    for l in range(N):
        for n in range(N):
            x_hat[l] += x[n] * eigenvectors[n,order[l]]
    
    plt.figure()
    plt.scatter(eigenvalues[order], np.abs(x_hat))
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_GFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_GFT.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_GFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_GFT.pdf", bbox_inches='tight')
    plt.show()
    
    if GFT_heatmap:
        df = pd.DataFrame(np.abs(x_hat), columns=[1], index=range(1,N+1))
        
        plt.figure()
        ax = sns.heatmap(df, cmap="Blues")
        ax.invert_yaxis()
        plt.yticks(rotation=0)
        if signal == "eigenvector":
            plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_GFT_heatmap.pdf", bbox_inches='tight')
        if signal == "eigenvectors":
            plt.savefig("results/"+name+"_eigenvectors_GFT_heatmap.pdf", bbox_inches='tight')
        if signal == "delta":
            plt.savefig("results/"+name+"_delta"+str(center)+"_GFT_heatmap.pdf", bbox_inches='tight')
        if signal == "exponential":
            plt.savefig("results/"+name+"_exponential_GFT_heatmap.pdf", bbox_inches='tight') 
        plt.show()

if iGFT:
    x_rebuilt = np.zeros((N,))
    for n in range(N):
        for l in range(N):
            x_rebuilt[n] += x_hat[l] * eigenvectors[n,order[l]]
    
    plt.figure()
    nx.draw_networkx(G, pos, node_color=x_rebuilt, node_size=100, vmin=vmin, vmax=vmax,
                      with_labels=False, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_iGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_iGFT.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_iGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_iGFT.pdf", bbox_inches='tight')
    plt.show()


#%% Translation Operator

if translation:
    Tx = np.zeros((N,))
    
    for n in range(N):
        for l in range(N):
            Tx[n] += x_hat[l] * eigenvectors[new_node,order[l]] * eigenvectors[n,order[l]]
    
    Tx *= np.sqrt(N)
    
    vmin_Tx = np.min(Tx)
    vmax_Tx = np.max(Tx)
    
    vmax_Tx = max(np.abs(vmin_Tx),vmax_Tx)
    vmin_Tx = - vmax_Tx
    
    plt.figure()
    nx.draw_networkx(G, pos, node_color=Tx, node_size=100, vmin=vmin_Tx, vmax=vmax_Tx,
                      with_labels=False, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_Tx,vmax=vmax_Tx))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_translation"+str(new_node)+".pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_translation"+str(new_node)+".pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_translation"+str(new_node)+".pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_translation"+str(new_node)+".pdf", bbox_inches='tight')
    plt.show()


#%% Modulation Operator

if modulation:
    Mx = np.zeros((N,))
    
    for n in range(N):
        Mx[n] = x[n] * eigenvectors[n,order[new_module-1]]
    
    Mx *= np.sqrt(N)
    
    vmin_Mx = np.min(Mx)
    vmax_Mx = np.max(Mx)
    
    vmax_Mx = max(np.abs(vmin_Mx),vmax_Mx)
    vmin_Mx = - vmax_Mx
    
    plt.figure()
    nx.draw_networkx(G, pos, node_color=Mx, node_size=100, vmin=vmin_Mx, vmax=vmax_Mx,
                      with_labels=False, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_Mx,vmax=vmax_Mx))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_modulation"+str(new_module)+".pdf", bbox_inches='tight')
    plt.show()

    Mx_hat = np.zeros((N,))
    
    for l in range(N):
        for n in range(N):
            Mx_hat[l] += Mx[n] * eigenvectors[n,order[l]]
    
    plt.figure()
    plt.scatter(eigenvalues[order], np.abs(Mx_hat))
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_modulation"+str(new_module)+"_HGFT.pdf", bbox_inches='tight')
    plt.show()


#%% Windowed Hypergraph Fourier Transform (spectrogram)

if WGFT:
    g_hat = np.zeros((N,))
    for l in range(N):
        g_hat[l] = np.exp(-tau*eigenvalues[order[l]])
    
    if plot_window:
        plt.figure()
        plt.scatter(eigenvalues[order], g_hat)
        plt.savefig("results/"+name+"_window"+str(tau)+"_GFT.pdf", bbox_inches='tight')
        plt.show()
        
        g = np.zeros((N,))
        for n in range(N):
            for l in range(N):
                g[n] += g_hat[l] * eigenvectors[n,order[l]]
        
        vmin_g = np.min(g)
        vmax_g = np.max(g)
        
        vmax_g = max(np.abs(vmin_g),vmax_g)
        vmin_g = - vmax_g
        
        plt.figure()
        nx.draw_networkx(G, pos, node_color=g, node_size=100, vmin=vmin_g, vmax=vmax_g,
                          with_labels=False, cmap=cmap)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_g, vmax=vmax_g))
        sm._A = []
        plt.colorbar(sm, ax=plt.gca())
        plt.savefig("results/"+name+"_window"+str(tau)+".pdf", bbox_inches='tight')
        plt.show()
    
    spectrogram = np.zeros((N,N))
    
    for k in range(N):
        for i in range(N):
            gik = np.zeros((N,))
            for n in range(N):
                for l in range(N):
                    gik[n] += g_hat[l] * eigenvectors[i,order[l]] * eigenvectors[n,order[l]]
                
                gik[n] *= eigenvectors[n,order[k]]
            
            gik *= N
            
            for n in range(N):
                spectrogram[i,k] += x[n] * gik[n]
    
    spectrogram = spectrogram.T
    
    xticklabels = range(1,N+1)
    yticklabels = range(1,N+1)
    
    if name == "random_geometric_hypergraph_gr" and spectral_clustering == True:
        spectrogram = spectrogram[:,set0 + set1 + set2]
        
        xticklabels = ["r" for i in set0]
        xticklabels = xticklabels + ["b" for i in set1]
        xticklabels = xticklabels + ["g" for i in set2]
    
    df = pd.DataFrame(spectrogram*spectrogram, columns=xticklabels, index=yticklabels)
    
    plt.figure()
    ax = sns.heatmap(df,cmap="jet")
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_spectrogram.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_spectrogram.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_spectrogram.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_spectrogram.pdf", bbox_inches='tight')
    plt.show()
    
elapsed_time = time.process_time() - start_time
print("Elapsed time:",elapsed_time)











