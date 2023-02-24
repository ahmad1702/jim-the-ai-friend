from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from .ml.chat import chat as ml_chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    'http://127.0.0.1:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Hello ML Class"}

@app.get('/chat')
async def chat():
    return {"message": "Hi bud"}

@app.post('/chat')
async def input_chat(message: Message):
    msg = message.message
    if not msg:
        raise HTTPException(status_code=400, detail="Invalid Message")
    return await ml_chat(msg)