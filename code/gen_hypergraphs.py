import math
import networkx as nx

def path_hypergraph(n=7):
    V = list(range(1,n+1))
    
    E = []
    for i in range(1,n-1,2):
        E.append([i,i+1,i+2])
    
    return(V,E)

def path_hypergraph_order4(n=7):
    V = list(range(1,n+1))
    
    E = []
    for i in range(1,n-1,3):
        E.append([i,i+1,i+2,i+3])
    
    return(V,E)

def cyclic_hypergraph(n):
    V = list(range(1,n+1))
    
    E = []
    for i in range(1,n-1,2):
        E.append([i,i+1,i+2])
    
    E.append([n-1,n,1])
    
    return(V,E)

def squid_hypergraph(n=40):
    V = list(range(1,n+1))
    
    remaining_nodes = list(range(2,n+1))
    centers = [1]
    
    E = []
    
    k = int(math.log(2*n+1,3)) - 1
    for n_iterations in range(k):
        n_centers = len(centers)
        for i in range(n_centers):
            E.append([centers[0],remaining_nodes[0],remaining_nodes[1],remaining_nodes[2]])
            
            centers.append(remaining_nodes[0])
            centers.append(remaining_nodes[1])
            centers.append(remaining_nodes[2])
            
            remaining_nodes.pop(0)
            remaining_nodes.pop(0)
            remaining_nodes.pop(0)
        
            centers.pop(0)
        
    #E.append([1,2,3,4])
    
    #E.append([2,5,6,7])
    #E.append([3,8,9,10])
    #E.append([4,11,12,13])
    
    return(V,E)

def random_geometric_hypergraph(n=64, radius=0.2, seed=4155):
    G = nx.random_geometric_graph(n, radius, seed=seed)
    
    mapping = dict()
    for i in range(n):
        mapping[i] = i+1
    G = nx.relabel_nodes(G, mapping)
    
    pos = nx.get_node_attributes(G, "pos")
    
    V = list(G.nodes)
    E = []
    
    for edge in G.edges:
        E.append([edge[0],edge[1]])
    
    E.append([1,46,63])
    E.append([4,7,61])
    E.append([28,53,55])
    
    return(V,E,pos)

# Hypergraph used in supplementary material
def hypergraph_H1():
    V = [1,2,3,4,5,6,7]
    
    E = [[1,2,5,7],[4,3,6],[3,5]]
    
    pos = dict()
    pos[1] = (0,2)
    pos[2] = (1,2)
    pos[3] = (1.5,1)
    pos[4] = (0.5,1)
    pos[5] = (2,2)
    pos[6] = (2.5,1)
    pos[7] = (3,2)
    
    return(V,E,pos)




