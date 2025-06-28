from fastapi import FastAPI

app = FastAPI()


@app.get("/ping")
def ping():
    return {"pong": True}


@app.get("/")
def root():
    return {"ok": True}
