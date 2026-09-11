import json
import httpx

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


app = FastAPI()

# ==============================
# CONFIG
# ==============================

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b-instruct-q4_K_M"


# ==============================
# DEBATER PROMPTS
# ==============================

DEBATER_A = """
You are Debater A.

You are arguing IN FAVOR of the topic.

Rules:
- Defend your position strongly.
- Give logical and convincing arguments.
- Directly respond to the opponent.
- Point out weaknesses in their reasoning.
- Do not agree with the opponent unless absolutely necessary.
- Stay focused on the debate topic.
- Do not mention that you are an AI.
- Keep your response reasonably concise and short about 2-3 sentence
"""

DEBATER_B = """
You are Debater B.

You are arguing AGAINST the topic.

Rules:
- Challenge the opponent's position strongly.
- Give logical counterarguments.
- Directly respond to the opponent.
- Point out weaknesses in their reasoning.
- Do not agree with the opponent unless absolutely necessary.
- Stay focused on the debate topic.
- Do not mention that you are an AI.
- Keep your response reasonably concise and short about 2-3 sentence
"""


# ==============================
# REQUEST MODEL
# ==============================

class DebateRequest(BaseModel):
    topic: str


# ==============================
# ASK OLLAMA — STREAMING
# ==============================

async def stream_ollama(system_prompt, prompt):

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": True
    }

    async with httpx.AsyncClient(timeout=None) as client:

        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/chat",
            json=payload
        ) as response:

            response.raise_for_status()

            async for line in response.aiter_lines():

                if not line:
                    continue

                data = json.loads(line)

                if "message" in data:

                    token = data["message"].get("content", "")

                    if token:
                        yield token


# ==============================
# DEBATE STREAM
# ==============================

async def debate_stream(topic):

    previous_argument = ""

    # --------------------------------
    # ROUND 1 — A
    # --------------------------------

    yield json.dumps({
        "type": "speaker",
        "speaker": "A",
        "round": 1
    }) + "\n"

    prompt = f"""
The debate topic is:

"{topic}"

You are opening the debate.

Present your strongest argument IN FAVOR of this position.
"""

    argument_a = ""

    async for token in stream_ollama(DEBATER_A, prompt):

        argument_a += token

        yield json.dumps({
            "type": "token",
            "speaker": "A",
            "token": token
        }) + "\n"


    # --------------------------------
    # ROUND 2 — B
    # --------------------------------

    yield json.dumps({
        "type": "speaker",
        "speaker": "B",
        "round": 2
    }) + "\n"

    prompt = f"""
The debate topic is:

"{topic}"

Debater A said:

"{argument_a}"

Now respond as Debater B.

Challenge Debater A's argument and present your strongest counterargument.
"""

    argument_b = ""

    async for token in stream_ollama(DEBATER_B, prompt):

        argument_b += token

        yield json.dumps({
            "type": "token",
            "speaker": "B",
            "token": token
        }) + "\n"


    # --------------------------------
    # ROUND 3 — A
    # --------------------------------

    yield json.dumps({
        "type": "speaker",
        "speaker": "A",
        "round": 3
    }) + "\n"

    prompt = f"""
The debate topic is:

"{topic}"

Debater B said:

"{argument_b}"

Your previous argument was:

"{argument_a}"

Now give a strong rebuttal.

Directly address Debater B's argument and defend your position.
"""

    argument_a2 = ""

    async for token in stream_ollama(DEBATER_A, prompt):

        argument_a2 += token

        yield json.dumps({
            "type": "token",
            "speaker": "A",
            "token": token
        }) + "\n"


    # --------------------------------
    # ROUND 4 — B
    # --------------------------------

    yield json.dumps({
        "type": "speaker",
        "speaker": "B",
        "round": 4
    }) + "\n"

    prompt = f"""
The debate topic is:

"{topic}"

Debater A's latest argument:

"{argument_a2}"

Debater B's previous argument:

"{argument_b}"

Now give your final rebuttal.

Directly challenge Debater A and defend your position.
"""

    async for token in stream_ollama(DEBATER_B, prompt):

        yield json.dumps({
            "type": "token",
            "speaker": "B",
            "token": token
        }) + "\n"


    # --------------------------------
    # FINISHED
    # --------------------------------

    yield json.dumps({
        "type": "done"
    }) + "\n"


# ==============================
# API
# ==============================

@app.post("/debate")
async def start_debate(request: DebateRequest):

    return StreamingResponse(
        debate_stream(request.topic),
        media_type="application/x-ndjson"
    )


# ==============================
# WEBSITE
# ==============================

app.mount(
    "/",
    StaticFiles(
        directory="static",
        html=True
    ),
    name="static"
)
