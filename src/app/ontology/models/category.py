from uuid import uuid1
from pydantic import BaseModel, Field

from app.ontology.models.base import EmoModel


class Category(EmoModel):
    id: str = Field(default_factory=lambda: f"C:{uuid1()}")
    hasIntensity: float
    emotionValue: str


class Archetypal(Category):
    pass


class DouglasCowie(Category):
    pass


class RobertPlutchik(Category):
    pass
