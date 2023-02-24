from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from .ml.chat import chat as ml_chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="chat-ai-ml-2023",
    version="1.0.0",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url=None,
    root_path="/api", # Set the root path to /api
)

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
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
    return ml_chat(msg)