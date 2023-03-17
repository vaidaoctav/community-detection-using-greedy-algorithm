import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
def plotNetwork(G, communities=[1, 1, 1, 1, 1, 1]):
    # to freeze the graph's view (networks uses a random view)
    np.random.seed(123)
    pos = nx.spring_layout(G)  # compute graph layout
    plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
    nx.draw_networkx_nodes(G, pos, node_size=300,
                           cmap=plt.cm.RdYlBu, node_color=communities[:-1])
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show()
def greedyCommunitiesDetection(network):
    communities = greedy_modularity_communities(G)
    communities = [list(x) for x in communities]
    nodeCommunities = [0] * len(G.nodes())
    for i, community in enumerate(communities):
        for node in community:
            nodeCommunities[node] = i
    return nodeCommunities

def modularity(G, communities, resolution=1.0, weight='weight'):
    if not isinstance(communities, list):
        communities = list(communities)

    directed = G.is_directed()
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        m = sum(out_degree.values())
        norm = 1 / m**2
    else:
        out_degree = in_degree = dict(G.degree(weight=weight))
        deg_sum = sum(out_degree.values())
        m = deg_sum / 2
        norm = 1 / deg_sum**2

    def community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in G.edges(
            comm, data=weight, default=1) if v in comm)

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum = sum(in_degree[u]
                            for u in comm) if directed else out_degree_sum

        return L_c / m - resolution * out_degree_sum * in_degree_sum * norm

    return sum(map(community_contribution, communities))


def greedyCommunities(network):
    communities = list(frozenset([u]) for u in network.nodes())
    # Track merges
    merges = []
    # Greedily merge communities until no improvement is possible
    old_modularity = None
    new_modularity = modularity(network, communities)
    while old_modularity is None or new_modularity > old_modularity:
        # Save modularity for comparison
        old_modularity = new_modularity
        # Find best pair to merge
        trial_communities = list(communities)
        to_merge = None
        for i, u in enumerate(communities):
            for j, v in enumerate(communities):
                # Skip i==j and empty communities
                if j <= i or len(u) == 0 or len(v) == 0:
                    continue
                # Merge communities u and v
                trial_communities[j] = u | v
                trial_communities[i] = frozenset([])
                trial_modularity = modularity(network, trial_communities)
                if trial_modularity >= new_modularity:
                    # Check if strictly better or tie
                    if trial_modularity > new_modularity:
                        # Found new best, save modularity and group indexes
                        new_modularity = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                    elif to_merge and min(i, j) < min(to_merge[0], to_merge[1]):
                        # Break ties by choosing pair with lowest min id
                        new_modularity = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                # Un-merge
                trial_communities[i] = u
                trial_communities[j] = v
        if to_merge is not None:
            # If the best merge improves modularity, use it
            merges.append(to_merge)
            i, j, dq = to_merge
            u, v = communities[i], communities[j]
            communities[j] = u | v
            communities[i] = frozenset([])
    # Remove empty communities and sort
    communities = [list(x) for x in communities]
    nodeCommunities = [0] * (len(G.nodes()) + 1)
    for i, community in enumerate(communities):
        for node in community:
            nodeCommunities[node] = i
    return nodeCommunities

crtDir = os.getcwd()
filePath = os.path.join(crtDir, "data", "tinamatr", "tinamatr.gml")
G = nx.read_gml(filePath, label='id')

communities = greedyCommunities(G)
plotNetwork(G, communities)
print(communities, len(set(communities)))


