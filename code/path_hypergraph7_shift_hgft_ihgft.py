import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

from scipy.fft import fft, fft2

from sklearn.cluster import KMeans

import gen_hypergraphs
import tensor_operations


V, E = gen_hypergraphs.path_hypergraph(7)
name = "path_hypergraph7"
seed = 2

# V, E = gen_hypergraphs.path_hypergraph(7)
# name = "path_hypergraph7b"
# seed = 2
# E.append([1,2])

# V, E = gen_hypergraphs.path_hypergraph(21)
# name = "path_hypergraph21"
# seed = 4

# V, E = gen_hypergraphs.path_hypergraph(181)
# name = "path_hypergraph181"
# seed = 0

# V, E = gen_hypergraphs.cyclic_hypergraph(24)
# name = "cycle_hypergraph24"
# seed = 2

# V, E = gen_hypergraphs.squid_hypergraph(13)
# name = "squid_hypergraph13"
# seed = 2

# V, E, pos = gen_hypergraphs.random_geometric_hypergraph(64, 0.20, seed=4155)
# name = "random_geometric_hypergraph"
# seed = 2

# V, E = gen_hypergraphs.path_hypergraph_order4(70)
# name = "path_hypergraph70_order4"
# seed = 2

graph_representation = False

shifting = True
HGFT = True
HGFT_heatmap = False
iHGFT = True
WHGFT = False; plot_window = False

plot_frontal_slice = False
plot_frontal_slices = False

spectral_clustering = False; n_clusters = 3

translation = False; new_node = 4
modulation = False; new_module = 7

t_HGFT = False
t_shifting = False

signal = "delta"
# signal = "eigenvector"
# signal = "eigenvectors"
# signal = "exponential"

if signal == "eigenvector":
    eigenvector = 2

if signal == "delta":
    center = 1

if name in ["path_hypergraph7","path_hypergraph7b","path_hypergraph21",
            "cycle_hypergraph24"]:
    tau = 1
    tau_x = 1

if name == "path_hypergraph181":
    tau = 300
    tau_x = 1

if name == "squid_hypergraph13":
    tau = 2
    tau_x = 2

if name == "path_hypergraph70_order4":
    tau = 50

if name == "random_geometric_hypergraph":
    tau = 3

G = tensor_operations.graph_representation(V,E)

if name in ["path_hypergraph7","path_hypergraph7b","path_hypergraph21",
            "cycle_hypergraph24","squid_hypergraph13"]:
    pos = nx.spring_layout(G, seed=seed)

if name in ["path_hypergraph181","path_hypergraph70_order4"]:
    pos = nx.spiral_layout(G)

start_time = time.process_time()

N = len(V)

if graph_representation:
    plt.figure()
    nx.draw_networkx(G, pos)
    if name == "path_hypergraph181":
        plt.axis("equal")
    plt.savefig("results/"+name+"_graph_representation.pdf", bbox_inches='tight')
    plt.show()

A = tensor_operations.adjacency_tensor(V,E)

D = tensor_operations.degree_tensor(V,E,A)

L = D - A

M = len(A.shape)

if M == 3:
    L_sym = tensor_operations.sym(L)
if M == 4:
    L_sym = tensor_operations.sym4(L)

if shifting:
    if M == 3:
        L_sym_hat = np.real(fft(L_sym))
    if M == 4:
        L_sym_hat = np.real(fft2(L_sym))

eigenvalues_hat, eigenvectors_hat = tensor_operations.t_eigendecomposition(L_sym)

eigenvalues_hat = np.real(eigenvalues_hat)
eigenvectors_hat = np.real(eigenvectors_hat)

# eigenvalues, eigenvectors = tensor_operations.t_decomposition(L_sym)

# if M == 3:
#     eigenvalues_hat = np.real(fft(eigenvalues, axis=-1))
#     eigenvectors_hat = np.real(fft(eigenvectors, axis=-1))
# if M == 4:
#     eigenvalues_hat = np.real(fft2(eigenvalues))
#     eigenvectors_hat = np.real(fft2(eigenvectors))

if M == 3:
    eigenvalues_vec = np.diag(eigenvalues_hat[:,:,0])
if M == 4:
    eigenvalues_vec = np.diag(eigenvalues_hat[:,:,0,0])

order = np.argsort(eigenvalues_vec)


#%% Hypergraph Spectral Clustering

if name == "random_geometric_hypergraph" and signal == "eigenvectors":
    spectral_clustering = True

if spectral_clustering:
    if M == 3:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(eigenvectors_hat[:,order[:n_clusters],0])
    
    if M == 4:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(eigenvectors_hat[:,order[:n_clusters],0,0])
    
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
    if name == "random_geometric_hypergraph":
        for edge in E:
            if len(edge) == 3:
                X = np.zeros([3,2])
                X[0,:] = pos[edge[0]]
                X[1,:] = pos[edge[1]]
                X[2,:] = pos[edge[2]]
                pol = plt.Polygon(X)
                plt.gca().add_patch(pol)
    plt.savefig("results/"+name+"_spectral_clustering"+str(n_clusters)+".pdf", bbox_inches='tight')
    plt.show()

#%% Signal

x = np.zeros((N,))

if signal == "delta":
    x[center-1] = 1

if signal == "eigenvector":
    if M == 3:
        x = eigenvectors_hat[:,order[eigenvector-1],0]
    if M == 4:
        x = eigenvectors_hat[:,order[eigenvector-1],0,0]

if signal == "eigenvectors":
    if name == "path_hypergraph181":
        x[:60] = eigenvectors_hat[:60,order[10],0]
        x[60:120] = eigenvectors_hat[60:120,order[60],0]
        x[120:] = eigenvectors_hat[120:,order[30],0]
    
    if name == "path_hypergraph70_order4":
        x[:20] = eigenvectors_hat[:20,order[10],0,0]
        x[20:40] = eigenvectors_hat[20:40,order[27],0,0]
        x[40:] = eigenvectors_hat[40:,order[5],0,0]
    
    if name == "random_geometric_hypergraph":
        if n_clusters == 3:
            for i in set0:
                x[i] = eigenvectors_hat[i,order[10],0]
            for i in set1:
                x[i] = eigenvectors_hat[i,order[27],0]
            for i in set2:
                x[i] = eigenvectors_hat[i,order[5],0]

if signal == "exponential":
    x_hat = np.zeros((N,))
    for l in range(N):
        if M == 3:
            x_hat[l] = np.exp(-tau_x * eigenvalues_hat[order[l],order[l],0])
        if M == 4:
            x_hat[l] = np.exp(-tau_x * eigenvalues_hat[order[l],order[l],0,0])
    
    x = np.zeros((N,))
    for n in range(N):
        for l in range(N):
            if M == 3:
                x[n] += x_hat[l] * eigenvectors_hat[n,order[l],0]
            if M == 4:
                x[n] += x_hat[l] * eigenvectors_hat[n,order[l],0,0]

if signal == "delta":
    vmax = 1
    vmin = -1

if signal in ["eigenvector","eigenvectors"]:
    if M == 3:
        vmin = np.min(eigenvectors_hat[:,:,0])
        vmax = np.max(eigenvectors_hat[:,:,0])
    if M == 4:
        vmin = np.min(eigenvectors_hat[:,:,0,0])
        vmax = np.max(eigenvectors_hat[:,:,0,0])
    
    vmax = max(np.abs(vmin),vmax)
    vmin = - vmax

if signal == "exponential":
    vmin = np.min(x)
    vmax = np.max(x)
    
    vmax = max(np.abs(vmin),vmax)
    vmin = - vmax

cmap = "seismic"

plt.figure()
nx.draw_networkx(G,pos,node_color=x,node_size=100,vmin=vmin,vmax=vmax,with_labels=False,cmap=cmap)
if name == "random_geometric_hypergraph":
    for edge in E:
        if len(edge) == 3:
            X = np.zeros([3,2])
            X[0,:] = pos[edge[0]]
            X[1,:] = pos[edge[1]]
            X[2,:] = pos[edge[2]]
            pol = plt.Polygon(X)
            plt.gca().add_patch(pol)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
sm._A = []
plt.colorbar(sm, ax=plt.gca())
if name == "path_hypergraph181":
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
    if M == 3:
        x_shifted = L_sym_hat[:,:,0] @ x
    if M == 4:
        x_shifted = L_sym_hat[:,:,0,0] @ x
    
    plt.figure()
    nx.draw_networkx(G,pos,node_color=x_shifted,node_size=100,vmin=vmin,vmax=vmax,with_labels=False,cmap=cmap)
    if name == "random_geometric_hypergraph":
        for edge in E:
            if len(edge) == 3:
                X = np.zeros([3,2])
                X[0,:] = pos[edge[0]]
                X[1,:] = pos[edge[1]]
                X[2,:] = pos[edge[2]]
                pol = plt.Polygon(X)
                plt.gca().add_patch(pol)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
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


#%% Hypergraph Fourier Transform

if translation or modulation:
    HGFT = True

if HGFT:
    x_hat = np.zeros((N,))

    for l in range(N):
        for n in range(N):
            if M == 3:
                x_hat[l] += x[n] * eigenvectors_hat[n,order[l],0]
            if M == 4:
                x_hat[l] += x[n] * eigenvectors_hat[n,order[l],0,0]

    plt.figure()
    plt.scatter(eigenvalues_vec[order], np.abs(x_hat))
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_HGFT.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_HGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_HGFT.pdf", bbox_inches='tight')     
    plt.show()
    
    if HGFT_heatmap:
        df = pd.DataFrame(np.abs(x_hat), columns=[1], index=range(1,N+1))
        
        plt.figure()
        ax = sns.heatmap(df, cmap="Blues")
        ax.invert_yaxis()
        plt.yticks(rotation=0)
        if signal == "eigenvector":
            plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_HGFT_heatmap.pdf", bbox_inches='tight')
        if signal == "eigenvectors":
            plt.savefig("results/"+name+"_eigenvectors_HGFT_heatmap.pdf", bbox_inches='tight')
        if signal == "delta":
            plt.savefig("results/"+name+"_delta"+str(center)+"_HGFT_heatmap.pdf", bbox_inches='tight')
        if signal == "exponential":
            plt.savefig("results/"+name+"_exponential_HGFT_heatmap.pdf", bbox_inches='tight') 
        plt.show()

if iHGFT:
    x_rebuilt = np.zeros((N,))
    for n in range(N):
        for l in range(N):
            if M == 3:
                x_rebuilt[n] += x_hat[l] * eigenvectors_hat[n,order[l],0]
            if M == 4:
                x_rebuilt[n] += x_hat[l] * eigenvectors_hat[n,order[l],0,0]
    
    plt.figure()
    nx.draw_networkx(G, pos, node_color=x_rebuilt, node_size=100, vmin=vmin, vmax=vmax,
                      with_labels=False, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    plt.colorbar(sm, ax=plt.gca())
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_iHGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_iHGFT.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_iHGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_iHGFT.pdf", bbox_inches='tight')
    plt.show()
    

#%% Translation Operator

if translation:
    Tx = np.zeros((N,))
    
    for n in range(N):
        for l in range(N):
            Tx[n] += x_hat[l] * eigenvectors_hat[new_node-1,order[l],0] * eigenvectors_hat[n,order[l],0]
    
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
        Mx[n] = x[n] * eigenvectors_hat[n,order[new_module-1],0]
    
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
            Mx_hat[l] += Mx[n] * eigenvectors_hat[n,order[l],0]
    
    plt.figure()
    plt.scatter(eigenvalues_vec[order], np.abs(Mx_hat))
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

if WHGFT:
    g_hat = np.zeros((N,))
    for l in range(N):
        if M == 3:
            g_hat[l] = np.exp(-tau*eigenvalues_hat[order[l],order[l],0])
        if M == 4:
            g_hat[l] = np.exp(-tau*eigenvalues_hat[order[l],order[l],0,0])
    
    if plot_window:
        plt.figure()
        plt.scatter(eigenvalues_vec[order], g_hat)
        plt.savefig("results/"+name+"_window"+str(tau)+"_HGFT.pdf", bbox_inches='tight')
        plt.show()
    
        g = np.zeros((N,))
        for n in range(N):
            for l in range(N):
                if M == 3:
                    g[n] += g_hat[l] * eigenvectors_hat[n,order[l],0]
                if M == 4:
                    g[n] += g_hat[l] * eigenvectors_hat[n,order[l],0,0]
        
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
                    if M == 3:
                        gik[n] += g_hat[l] * eigenvectors_hat[i,order[l],0] * eigenvectors_hat[n,order[l],0]
                    if M == 4:
                        gik[n] += g_hat[l] * eigenvectors_hat[i,order[l],0,0] * eigenvectors_hat[n,order[l],0,0]
                if M == 3:
                    gik[n] *= eigenvectors_hat[n,order[k],0]
                if M == 4:
                    gik[n] *= eigenvectors_hat[n,order[k],0,0]
            
            gik *= N
            
            for n in range(N):
                spectrogram[i,k] += x[n] * gik[n]
    
    spectrogram = spectrogram.T
    
    xticklabels = range(1,N+1)
    yticklabels = range(1,N+1)
    
    if name == "random_geometric_hypergraph" and spectral_clustering == True:
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


#%% Frontal Slices

if plot_frontal_slice or plot_frontal_slices:
    xticklabels = range(1,N+1)
    yticklabels = range(1,N+1)

if plot_frontal_slice:
    if M == 3:
        df = pd.DataFrame(eigenvectors_hat[:,order,0], columns=xticklabels, index=yticklabels)
    if M == 4:
        df = pd.DataFrame(eigenvectors_hat[:,order,0,0], columns=xticklabels, index=yticklabels)
        
    plt.figure()
    ax = sns.heatmap(df, cmap="Reds")
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    plt.savefig("results/"+name+"_first_frontal_slice.pdf", bbox_inches='tight')
    plt.show()

if plot_frontal_slices:
    for i in range(N):
        if M == 3:
            df = pd.DataFrame(eigenvectors_hat[:,order,i], columns=xticklabels, index=yticklabels)
        if M == 4:
            df = pd.DataFrame(eigenvectors_hat[:,order,i,i], columns=xticklabels, index=yticklabels)
            
        plt.figure(dpi=50)
        plt.title("Slice "+str(i+1))
        ax = sns.heatmap(df, cmap="Reds")
        ax.invert_yaxis()
        plt.savefig("results/"+name+"_"+str(i+1)+"_frontal_slice.pdf", bbox_inches='tight')
        plt.show()


#%% t-Hypergraph Fourier Transform

if t_shifting or t_HGFT:
    
    if M == 3:
        X = np.outer(x,x)
    if M == 4:
        X = tensor_operations.outer(np.outer(x,x),x)
    
    X_expand = tensor_operations.t_expand(X)
    
    X_expand_sym = tensor_operations.sym(X_expand)
    
    Ns = 2*N+1

    t_xticklabels = range(1,Ns+1)
    t_yticklabels = range(1,N+1)
    
if t_shifting:
    X_expand_sym_shift = tensor_operations.t_product(L_sym, X_expand_sym)
    X_expand_sym_shift = np.real(X_expand_sym_shift)
    
    df = pd.DataFrame(X_expand_sym_shift[:,0,:], columns=t_xticklabels, index=t_yticklabels)

    plt.figure()
    ax = sns.heatmap(df, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_t_shifted.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_t_shifted.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_t_shifted.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_t_shifted.pdf", bbox_inches='tight') 
    plt.show()
    
if t_HGFT:
    eigenvalues, eigenvectors = tensor_operations.t_decomposition(L_sym)
    
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    eigenvectors_transpose = tensor_operations.t_transpose(eigenvectors[:,order,:])
    
    X_expand_sym_Fourier = tensor_operations.t_product(eigenvectors_transpose, X_expand_sym)
    X_expand_sym_Fourier = np.real(X_expand_sym_Fourier)
    
    df = pd.DataFrame(X_expand_sym_Fourier[:,0,:], columns=t_xticklabels, index=t_yticklabels)
    
    plt.figure()
    ax = sns.heatmap(df, cmap="Blues")
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    if signal == "eigenvector":
        plt.savefig("results/"+name+"_eigenvector"+str(eigenvector)+"_tHGFT.pdf", bbox_inches='tight')
    if signal == "eigenvectors":
        plt.savefig("results/"+name+"_eigenvectors_tHGFT.pdf", bbox_inches='tight')
    if signal == "delta":
        plt.savefig("results/"+name+"_delta"+str(center)+"_tHGFT.pdf", bbox_inches='tight')
    if signal == "exponential":
        plt.savefig("results/"+name+"_exponential_tHGFT.pdf", bbox_inches='tight') 
    plt.show()

elapsed_time = time.process_time() - start_time
print("Elapsed time:",np.round(elapsed_time,4))




