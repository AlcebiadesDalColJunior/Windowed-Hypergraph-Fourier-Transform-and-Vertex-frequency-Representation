import math
import numpy as np
import networkx as nx

import itertools

from scipy.fft import fft, fft2, ifft, ifft2

    
def graph_representation(V,E):

    G = nx.empty_graph()
    G.add_nodes_from(V)
    
    G_edges = []
    for edge in E:
        for element in list(itertools.combinations(edge, 2)):
            G_edges.append(element)
    
    weights = dict()
    for edge in G_edges:
        weights[edge] = 0
    
    for edge in G_edges:
        weights[edge] += 1
    
    for edge in G_edges:
        G.add_edge(edge[0],edge[1],weight=weights[edge])
    
    return(G)

def adjacency_tensor(V,E):
    n = len(V)
    
    E_lengths = []
    for edge in E:
      E_lengths.append(len(edge))
    
    # Maximum cardinality of hyperedges (m.c.e)
    
    M = max(E_lengths)
    
    adjacency_tensor_size = [n for i in range(M)]
    
    A = np.zeros(adjacency_tensor_size)
    
    for edge in E:
        c = len(edge)
          
        subsets = list(itertools.combinations_with_replacement(edge, M-c))
        
        #print()
        #print(subsets)
        
        ps = []
        for subset in subsets:
            edge_subset = edge + list(subset)
            
            #print()
            #print(edge_subset)
        
            for p in list(set(itertools.permutations(edge_subset))):
                ps.append(p)
            
        #print(ps)
          
        #positive_integers = list(itertools.product(range(1,M), repeat=c))
        positive_integers = list(itertools.product(range(1,M-(c-1)+1), repeat=c))
        #print(positive_integers)
          
        summation = 0
        
        for ks in positive_integers:
            if np.sum(ks) == M:
                product = 1
                for k in ks:
                    product *= 1/math.factorial(k)
                  
                summation += product
          
        summation *= math.factorial(M)
        
        weight = c / summation
        
        for p in ps:
            indexes = []
            for node in p:
                indexes.append(V.index(node))
            
            indexes = tuple(indexes)
            
            A[indexes] = weight
    
    return(A)

def degree_tensor(V,E,A=None):
    if A is None:
        A = adjacency_tensor(V,E)
    
    n = len(V)
    
    E_lengths = []
    for edge in E:
      E_lengths.append(len(edge))
    
    # Maximum cardinality of hyperedges (m.c.e)
    
    M = max(E_lengths)
    
    adjacency_tensor_size = [n for i in range(M)]
    
    D = np.zeros(adjacency_tensor_size)
    
    counter = np.zeros(M, dtype=np.int_)
    
    entries = []
    while counter[-1] != n:
        entries.append(list(counter))
        
        counter[0] += 1
        
        for i in range(M):
            if counter[i] == n:
                if i+1 < M:
                    counter[i] = 0
                    counter[i+1] += 1
                else:
                    break
    
    for node in V:
        i = V.index(node)
        
        diagonal_position = tuple([i for j in range(M)])
        
        for entry in entries:
            if entry[0] == i:
                D[diagonal_position] += A[tuple(entry)]
    
    return(D)

def t_product(A,B):
    
    A_hat = fft(A, axis=2)
    B_hat = fft(B, axis=2)
    
    N1, N2, N3 = A.shape
    N2, N4, N3 = B.shape
    
    AB_hat = np.zeros((N1,N4,N3), dtype=complex)
    
    for k in range(N3):
        AB_hat[:,:,k] = A_hat[:,:,k] @ B_hat[:,:,k]
    
    AB = ifft(AB_hat, axis=2)
    
    return(AB)

def t_product4(A,B):
    
    A_hat = fft(A, axis=2)
    B_hat = fft(B, axis=2)
    
    N1, N2, N3, N4 = A.shape
    N2, L, N3, N4 = B.shape
    
    AB_hat = np.zeros((N1,L,N3,N4), dtype=complex)
    
    for k in range(N3):
        AB_hat[:,:,:,k] = t_product(A_hat[:,:,:,k], B_hat[:,:,:,k])
    
    AB = ifft(AB_hat, axis=2)
    
    return(AB)

def t_transpose(A):
    
    A_shape = A.shape
    A_transpose_shape = np.copy(A_shape)
    A_transpose_shape[0] = A_shape[1]
    A_transpose_shape[1] = A_shape[0]
    
    A_transpose = np.zeros(A_transpose_shape, dtype=complex)
    
    A_transpose[...,0] = A[...,0].T
    
    Np = A_shape[-1]
    
    for k in range(1,Np):
        A_transpose[...,k] = A[...,Np-k].T
    
    return(A_transpose)

def t_transpose4(A):
    
    A_shape = A.shape
    A_transpose_shape = np.copy(A_shape)
    A_transpose_shape[0] = A_shape[1]
    A_transpose_shape[1] = A_shape[0]
    
    A_transpose = np.zeros(A_transpose_shape, dtype=complex)
    
    A_transpose[...,0] = t_transpose(A[...,0])
    
    Np = A_shape[-1]
    
    for k in range(1,Np):
        A_transpose[...,k] = t_transpose(A[...,Np-k])
    
    return(A_transpose)

def t_norm(X_expand, efficient_computing=True):
    
    if efficient_computing:

        N1, N2, N3 = X_expand.shape
        
        X_expand_hat = fft(X_expand, axis=2)
        
        norm_X_expand_hat = np.zeros((N1, N2, N3), dtype=complex)
        
        for k in range(N3):
            norm_X_expand_hat[:,:,k] = X_expand_hat[:,:,k].T @ X_expand_hat[:,:,k]
        
        norm_X_expand_hat = np.sqrt(norm_X_expand_hat)
        
        norm_X_expand = ifft(norm_X_expand_hat, axis=2)
    
    else:
    
        norm_X_expand = np.sqrt(t_product(t_transpose(X_expand) , X_expand))
    
    return(norm_X_expand)


def t_decomposition(A):
    
    A_shape = A.shape
    
    M = len(A.shape)
    
    N3 = A_shape[-1]
    
    eigenvalues_hat = np.zeros(A_shape, dtype=complex)
    eigenvectors_hat = np.zeros(A_shape, dtype=complex)
    
    if M == 3:
        A_hat = fft(A)
    if M == 4:
        A_hat = fft2(A)
    
    for k in range(N3):
        if M == 3:
            eigenvalues_vec, eigenvectors_hat[:,:,k] = np.linalg.eig(A_hat[:,:,k])
            eigenvalues_hat[:,:,k] = np.diag(eigenvalues_vec)
        if M == 4:
            eigenvalues_vec, eigenvectors_hat[:,:,k,k] = np.linalg.eig(A_hat[:,:,k,k])
            eigenvalues_hat[:,:,k,k] = np.diag(eigenvalues_vec)
    
    if M == 3:
        eigenvalues = ifft(eigenvalues_hat)
        eigenvectors = ifft(eigenvectors_hat)
    if M == 4:
        eigenvalues = ifft2(eigenvalues_hat)
        eigenvectors = ifft2(eigenvectors_hat)
    
    return(eigenvalues,eigenvectors)

def t_eigendecomposition(A):
    
    A_shape = A.shape
    
    M = len(A.shape)
    
    N3 = A_shape[-1]
    
    eigenvalues_hat = np.zeros(A_shape, dtype=complex)
    eigenvectors_hat = np.zeros(A_shape, dtype=complex)
    
    if M == 3:
        A_hat = fft(A)
    if M == 4:
        A_hat = fft2(A)
    
    for k in range(N3):
        if M == 3:
            eigenvalues_vec, eigenvectors_hat[:,:,k] = np.linalg.eig(A_hat[:,:,k])
            eigenvalues_hat[:,:,k] = np.diag(eigenvalues_vec)
        if M == 4:
            eigenvalues_vec, eigenvectors_hat[:,:,k,k] = np.linalg.eig(A_hat[:,:,k,k])
            eigenvalues_hat[:,:,k,k] = np.diag(eigenvalues_vec)
    
    return(eigenvalues_hat,eigenvectors_hat)

def sym(A):
    
    N = A.shape[0]
    
    M = len(A.shape)

    Ns = 2*N+1
    
    A_sym_shape = list(A.shape)
    for i in range(2,M):
        A_sym_shape[i] = Ns
    
    A_sym = np.zeros(A_sym_shape)
    
    for i in range(N):
        A_sym[:,:,i+1] = A[:,:,i]
        A_sym[:,:,Ns-(i+1)] = A[:,:,i]
    
    A_sym *= 0.5
    
    return(A_sym)

def sym4(A):
    
    N = A.shape[0]
    
    M = len(A.shape)

    Ns = 2*N+1
    
    A_sym_shape = list(A.shape)
    for i in range(2,M):
        A_sym_shape[i] = Ns
    
    A_sym = np.zeros(A_sym_shape)
    
    for i in range(N):
        A_sym[...,i+1] = sym(A[...,i])
        A_sym[...,Ns-(i+1)] = sym(A[...,i])
    
    A_sym *= 0.5
    
    return(A_sym)

def outer(a,b):
    
    shape_a = a.shape
    shape_b = b.shape
    
    shape_outer = tuple(list(shape_a) + list(shape_b))
    
    outer_a_b = np.zeros((shape_outer))
    
    for k in range(shape_b[0]):
        if len(shape_a) == 1:
            for i in range(shape_a[0]):
                outer_a_b[i,k] = a[i] * b[k]
        if len(shape_a) == 2:
            for i in range(shape_a[0]):
                    for j in range(shape_a[1]):
                        outer_a_b[i,j,k] = a[i,j] * b[k]
    
    return(outer_a_b)

def t_expand(X):
    
    N, N = X.shape
    X_expand = np.zeros((N,1,N))
    
    X_expand[:,0,:] = X
    
    return(X_expand)









