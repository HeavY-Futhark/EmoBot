from fastapi import APIRouter
from app.ontology.service import create_emonto

router = APIRouter()

@router.get("/test/create")
def test_create():
    g = create_emonto()
    return g


