

import networkx as nx
import model
import Graph

graph = Graph.Graph(nx.karate_club_graph().to_directed())
embedding_model = model.Model(len(graph), 5)
embedding_model.fit(graph, alpha=0.10, iter=20, negative_samples=4)