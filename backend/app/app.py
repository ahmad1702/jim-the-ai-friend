from fastapi import FastAPI
from .ml.chat import chat_with_jim

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello ML Class"}

@app.get('/chat')
async def chat():
    return chat_with_jim()