from enum import Enum
from typing import List
from uuid import uuid1, uuid4
from matplotlib import category
from pydantic import BaseModel, Field
from rdflib import Graph, Namespace
from .modality import Modality as Modality
from .base import EmoModel as EmoModel
from .category import (
    Category as Category,
    Archetypal as Archetypal,
    DouglasCowie as DouglasCowie,
    RobertPlutchik as RobertPlutchik,
)

EMO = Namespace("http://www.emonto.org/")


class Object(EmoModel):
    id: str = Field(default_factory=lambda: f"O:{uuid1()}")


class Person(EmoModel):
    id: str = Field(default_factory=lambda: f"P:{uuid1()}")


class Annotator(EmoModel):
    id: str = Field(default_factory=lambda: f"A:{uuid1()}")


class AutomaticAnnotator(Annotator):
    pass


class HumanAnnotator(Annotator):
    pass


class Emotion(EmoModel):
    id: str = Field(default_factory=lambda: f"EM:{uuid1()}")
    hasCategory: Category
    isAnnotatedBy: List[Annotator] = []
    hasModality: List[Modality] = []


class Event(EmoModel):
    id: str = Field(default_factory=lambda: f"E:{uuid1()}")
    isProducedBy: List[Person] = []
    isCausedBy: List[Object] = []
    produces: List[Emotion] = []
