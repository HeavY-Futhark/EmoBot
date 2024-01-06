from pprint import pp
from typing import List
from pydantic import BaseModel
from pydantic.fields import SHAPE_LIST, SHAPE_SINGLETON
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD, FOAF

from app.ontology.models import EMO, Annotator, Category, Modality
from app.ontology.models import Object, Person, Event, Emotion
from typing import Type

from app.ontology.models.base import EmoModel


def add_subclasses(g: Graph, cls: Type[BaseModel]) -> Graph:
    for subclass in cls.__subclasses__():
        g.add(
            (
                EMO[subclass.__name__],
                RDFS.subClassOf,
                EMO[cls.__name__],
            )
        )
    return g


def add_instance(g: Graph, obj: EmoModel) -> Graph:
    """
    Adds an instance of an EmoModel to the graph
    Adds recursively all the nested EmoModels
    """
    if (EMO[obj.id], RDF.type, EMO[obj.__class__.__name__]) in g:
        raise Exception(
            f"Instance {obj.id} of type {obj.__class__.__name__} already exists"
        )
    g.add((EMO[obj.id], RDF.type, EMO[obj.__class__.__name__]))

    for k, v in obj.__fields__.items():
        if k == "id":
            continue
        if v.shape == SHAPE_LIST:
            fields = getattr(obj, k)
        elif v.shape == SHAPE_SINGLETON:
            fields = [getattr(obj, k)]
        else:
            raise Exception(f"Unsupported data shape {v.shape}")
        
        for field in fields:
            if issubclass(v.type_, EmoModel):
                add_instance(g, field)
                g.add((EMO[obj.id], EMO[k], EMO[field.id]))
            else:
                g.add((EMO[obj.id], EMO[k], Literal(field)))
    return g


def create_emonto(g: Graph | None = None) -> Graph:
    if g is None:
        g = Graph()

    g.bind("emo", EMO)
    g.bind("foaf", FOAF)
    add_subclasses(g, Modality)
    add_subclasses(g, Annotator)
    add_subclasses(g, Category)

    return g


def add_event(
    g: Graph,
    e: Event,
) -> Graph:
    add_instance(g, e)

    return g
