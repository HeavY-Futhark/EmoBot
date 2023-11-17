
from fastapi import FastAPI
from app.ontology.routing import router as ontology_router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

app.include_router(ontology_router, prefix="/ontology")


