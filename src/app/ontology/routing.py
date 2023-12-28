from pprint import pp
from fastapi import APIRouter
from ontology.service import create_emonto

router = APIRouter()

@router.get("/test/create")
def test_create():
    g = create_emonto()
    pp(g)
    return g


