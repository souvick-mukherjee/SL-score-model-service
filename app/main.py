from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.v1.routes import router

app = FastAPI()

print("🎯 FastAPI is starting up")


@app.get("/ping")
def ping():
    return {"pong": True}


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}


app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(router, prefix="/api/v1")
