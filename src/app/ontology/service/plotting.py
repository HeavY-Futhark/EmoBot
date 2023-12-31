import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph


def plot_ontology(g: Graph) -> None:
    G = rdflib_to_networkx_multidigraph(g)

    # Plot Networkx instance of RDF Graph
    pos = nx.spring_layout(G, scale=2)
    edge_labels = nx.get_edge_attributes(G, "r")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos, with_labels=True)
    
    plt.show()
