from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=3
N_WALKS=50
random.seed(42)

def run_random_walks(G, nodes, prob_matrix, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                if len(G.neighbors(curr_node))==0:
                    continue
                ## unweighted: next_node = random.choice(G.neighbors(curr_node))
                ## weighted
                probs = prob_matrix[curr_node]
                next_node = np.random.choice(len(probs), 1, p=probs)[0]
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs


if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    weighted_adjacency_matrix_file = sys.argv[2]
    out_file = sys.argv[3]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)

    weighted_adjacency_matrix = np.load(weighted_adjacency_matrix_file)
    # converting weighted adjacency matrix to probabaility matrix
    prob_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        prob_matrix[i] = weighted_adjacency_matrix[i]/np.sum(weighted_adjacency_matrix[i])

    pairs = run_random_walks(G, nodes, prob_matrix)
    seen = set()
    seen_add = seen.add
    seen_add = [x for x in pairs if not (x in seen or seen_add(x))]
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in seen_add]))

