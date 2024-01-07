from uuid import uuid1
from pydantic import BaseModel, Field

from app.ontology.models.base import EmoModel


class Modality(EmoModel):
    """
    Modality class
    Possible subclasses :  Context, Text, Voice, Posture, Face, Gesture
    """
    id: str = Field(default_factory=lambda: f"M:{uuid1()}")

class Context(Modality):
    pass
    

class Text(Modality):
    pass

class Voice(Modality):
    pass

class Posture(Modality):
    pass

class Face(Modality):
    pass

class Gesture(Modality):
    pass