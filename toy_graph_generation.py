import json
import numpy as np
import os

## creating nodes
nodes=[]
node={}
for i in range(0, 11):
	node['test']=False
	node['id']=i
	node['val']=False
	nodes.append(node)
	node={}

## creating edges
weighted_adjacency_matrix = np.zeros((11, 11))
edge_tuple = [(0, 1, 0.8), (0, 2, 0.9), (1, 2, 0.7), (2, 3, 0.3), (3, 4, 0.2), (4, 5, 0.3), (4, 8, 0.9), (5, 6, 0.7), (5, 7, 0.8), (8, 9, 0.8), (8, 10, 0.2), (9, 10, 0.5)]
edges=[]
edge={}
for item in edge_tuple:
    weighted_adjacency_matrix[item[0]][item[1]] = item[2]
    weighted_adjacency_matrix[item[1]][item[0]] = item[2]
    edge['source']=item[0]
    edge['target']=item[1]
    edges.append(edge)
    edge={}

## creating graph
G={}
G['directed']=False
G['graph']={}
G['nodes']=nodes
G['links']=edges
G['multigraph']=False

with open('toy-G.json', 'w') as fp:
    json.dump(G, fp)

## creating node_id to id integer
node_id_dict={}
for i in range(0, 11):
	node_id_dict[str(i)]=i

with open('toy-id_map.json', 'w') as fp:
    json.dump(node_id_dict, fp)

## creating node_id to classes
node_class={}
class_list=np.random.randint(0,3,11)
for i in range(len(class_list)):
	class_label=np.zeros(3,np.int8)
	class_label[class_list[i]]=1
	class_label=class_label.tolist()
	node_class[str(i)]=class_label

with open('toy-class_map.json', 'w') as fp:
    json.dump(node_class, fp)
	
## creating node_features
arr_of_emb = [[1, 7], [2, 9], [3, 5], [5, 8], [7, 11], [8, 7], [6, 5], [9, 4.5], [10, 9], [11.5, 7], [11, 2]]
arr_of_emb = np.array(arr_of_emb, dtype='float32')
np.save('toy-feats.npy',arr_of_emb)

## saving weighted adjacency matrix
np.save('toy-weighted_adjacency_matrix.npy',weighted_adjacency_matrix)




