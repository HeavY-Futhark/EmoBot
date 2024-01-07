from ipaddress import ip_address
import json
from pprint import pp
from fastapi import APIRouter
from matplotlib.patches import Arc
from app.ontology.models import Emotion
from app.ontology.models.base import EmoModel
from app.ontology.models import Event
from app.ontology.models.category import Archetypal, Category
from app.ontology.service import create_emonto, add_event
from app.ontology.service.plotting import plot_ontology

router = APIRouter()


@router.post("/test/create")
def test_create(
    e: Event = Event(produces=[Emotion(hasCategory=Archetypal(hasIntensity=0.5, emotionValue="test"))])
                                       ):
    g = create_emonto()
    add_event(g, e)    
    print(isinstance(e.produces[0].hasCategory, Archetypal))  # returns False

    plot_ontology(g)
    return json.loads(g.serialize(format="json-ld"))
