import networkx as nx
import random
import torch
from torch.autograd import Variable

class Graph(object):

    def __init__(self, graph):
        self.graph = graph
        self.n = nx.number_of_nodes(graph)
        self._node2id = dict(enumerate(graph.nodes()))
        self._id2node = {v:k for v,k in self._node2id.items()}
        self.all_nodes = set(graph.nodes())
        self.cached_negs = dict()

    def __len__(self):
        return self.n

    def get_negative_samples(self, node, num_negative=10):
        neighbors = set(self.graph[node].keys())
        neighbors.add(node)
        not_neighbors = self.all_nodes - set(neighbors)
        samples = random.sample(not_neighbors, num_negative)
        tensor_contents = [self._node2id[s] for s in samples]
        tensor = torch.LongTensor(tensor_contents)
        var = Variable(tensor)
        return var

    def get_examples(self, num_negative=10):
        for u,v in self.graph.edges_iter():
            u_var = Variable(torch.LongTensor([self._node2id[u]]))
            v_var = Variable(torch.LongTensor([self._node2id[v]]))
            # if (u, v) not in self.cached_negs:
            #     self.cached_negs[u,v] = self.get_negative_samples(u, num_negative)
            # negatives = self.cached_negs[u,v]
            negatives = self.get_negative_samples(u, num_negative)
            yield u_var, v_var, negatives

