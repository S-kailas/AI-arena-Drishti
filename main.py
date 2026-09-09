from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

app = FastAPI()

# CHANGE THIS TO LAPTOP 1'S IP
OLLAMA_URL = "http://192.168.1.105:11434"

MODEL = "qwen2.5:7b-instruct-q4_K_M"


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": request.message
            }
        ],
        "stream": False
    }

    async with httpx.AsyncClient(timeout=300) as client:

        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload
        )

        response.raise_for_status()

        data = response.json()

    return {
        "response": data["message"]["content"]
    }


# Serve website
app.mount("/", StaticFiles(directory="static", html=True), name="static")
