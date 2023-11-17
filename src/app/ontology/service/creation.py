from rdflib import Graph, Namespace


def create_emonto() -> Graph:
    g = Graph()
    EMO = Namespace("http://www.emonto.org/ (accessed on 18 December 2020)") 
    # g.add()
    return g
