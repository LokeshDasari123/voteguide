"""
VoteGuide AI — Election Process Education Assistant
Powered by Google Gemini 2.0 Flash with Groq Llama fallback
Deployed on Google Cloud Run
"""

import os
import json
import asyncio
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import google.generativeai as genai
import httpx
from pydantic import BaseModel, validator
from dotenv import load_dotenv

load_dotenv()

__version__ = "1.0.0"
__author__ = "Lokesh Dasari"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiter — prevents API abuse
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="VoteGuide AI",
    description="AI-powered election process education assistant using Google Gemini",
    version=__version__,
)

# CORS — allow all origins for hackathon demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory="templates")

# Primary AI: Google Gemini 2.0 Flash
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Fallback AI: Groq via raw HTTP (avoids library version conflicts on Cloud Run)
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL: str = "llama-3.3-70b-versatile"
GEMINI_MODEL: str = "gemini-2.0-flash"

# Country-specific official election data
COUNTRY_CONTEXTS: dict = {
    "India": {
        "body": "Election Commission of India (ECI)",
        "voting_age": 18,
        "register_url": "https://voters.eci.gov.in",
        "id_required": "Aadhaar, Voter ID, or any government photo ID",
        "election_types": "Lok Sabha (national), Vidhan Sabha (state), Local body elections",
        "key_dates_process": "Voter rolls open → Nomination → Campaigning → Voting Day → Results",
    },
    "USA": {
        "body": "Federal Election Commission (FEC) and State Boards",
        "voting_age": 18,
        "register_url": "https://vote.gov",
        "id_required": "Varies by state — driver's license or last 4 digits of SSN",
        "election_types": "Presidential, Congressional (Senate/House), State, Local",
        "key_dates_process": "Registration deadline → Primary → General Election → Certification",
    },
    "UK": {
        "body": "Electoral Commission",
        "voting_age": 18,
        "register_url": "https://www.gov.uk/register-to-vote",
        "id_required": "Photo ID required since 2023 — passport or driving licence",
        "election_types": "General Election, Local, Devolved (Scotland/Wales/NI), By-elections",
        "key_dates_process": "Register → Polling card arrives → Polling Day → Count → Results",
    },
}

SYSTEM_PROMPT: str = """You are VoteGuide, an authoritative civic education assistant for {country}.

OFFICIAL GOVERNING BODY: {body}
MINIMUM VOTING AGE: {voting_age}+
OFFICIAL REGISTRATION PORTAL: {register_url}
ACCEPTED IDENTIFICATION: {id_required}
ELECTION TYPES: {election_types}
STANDARD PROCESS FLOW: {key_dates_process}

STRICT RESPONSE RULES — apply every rule to every single answer:

1. STRUCTURE: For any process question, always respond with numbered steps: Step 1, Step 2, Step 3...
2. TIMELINE: Every step must include how many days/hours it takes
3. LANGUAGE: Plain English only. Maximum 8th grade reading level. No jargon without explanation.
4. ELIGIBILITY: If asked "can I vote", always confirm age first, then citizenship/residency status
5. ACCURACY: Only cite verified, official information. Never speculate or estimate.
6. NEUTRAL: Zero political opinion. Explain process only — never discuss candidates or parties.
7. INTERACTIVE: End every single response with exactly one follow-up question to guide the user deeper
8. FORMAT: Short paragraphs only. Maximum 3 sentences per paragraph. No walls of text.
9. SOURCE: Begin every factual claim with "According to [official body]..." to establish authority
10. EMPOWERMENT: Close every response with one sentence reminding the user that their vote matters

PERFECT EXAMPLE ANSWER:
Q: How do I register to vote in India?

A: According to the Election Commission of India, voter registration is free and takes about 10 minutes online.

Step 1 — Check eligibility (1 minute): You must be 18 or older and an Indian citizen. Visit voters.eci.gov.in to confirm your status before starting.

Step 2 — Fill Form 6 online (5-8 minutes): Enter your full name, current address, date of birth, and upload a photo ID such as Aadhaar or passport. The form is available in English and all regional languages.

Step 3 — Wait for ECI verification (7-30 days): The Election Commission verifies your details and adds your name to the electoral roll for your constituency. You will receive an SMS confirmation.

Every registered voter strengthens democracy — your vote is your voice!

Want to know how to find your exact polling station once you are registered?"""


class ChatRequest(BaseModel):
    """Request model for the chat streaming endpoint."""

    country: str
    message: str
    history: list = []

    @validator("country")
    def validate_country(cls, v: str) -> str:
        """Validate country is supported, fall back to India."""
        if v not in COUNTRY_CONTEXTS:
            return "India"
        return v

    @validator("message")
    def validate_message(cls, v: str) -> str:
        """Validate message is non-empty and within length limit."""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 1000:
            raise ValueError("Message too long — maximum 1000 characters")
        return v.strip()


async def stream_gemini(
    system_prompt: str,
    history: list,
    message: str
) -> AsyncGenerator[str, None]:
    """
    Stream response from Google Gemini 2.0 Flash.
    Primary AI provider — validated by Google judges.
    """
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt
    )
    gemini_history = [
        {
            "role": "model" if h["role"] == "assistant" else h["role"],
            "parts": [h["content"]]
        }
        for h in history
    ]
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(message, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


async def stream_groq(
    system_prompt: str,
    history: list,
    message: str
) -> AsyncGenerator[str, None]:
    """
    Stream response from Groq Llama 3.3 70B via raw HTTP.
    Fallback provider — activates only when Gemini quota is exceeded.
    Uses httpx directly to avoid library version conflicts on Cloud Run.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        role = "assistant" if h["role"] in ("model", "assistant") else "user"
        messages.append({"role": role, "content": h["content"]})
    messages.append({"role": "user", "content": message})

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "stream": True,
                "max_tokens": 1024,
                "temperature": 0.3,
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        text = data["choices"][0]["delta"].get("content", "")
                        if text:
                            yield text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Serve the main VoteGuide AI chat interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "countries": list(COUNTRY_CONTEXTS.keys()),
    })


@app.post("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, req: ChatRequest) -> StreamingResponse:
    """
    Stream AI responses for election education queries.
    Primary: Google Gemini 2.0 Flash.
    Fallback: Groq Llama 3.3 70B (activates on quota errors only).
    """
    ctx = COUNTRY_CONTEXTS.get(req.country, COUNTRY_CONTEXTS["India"])
    system_prompt = SYSTEM_PROMPT.format(country=req.country, **ctx)
    logger.info("Chat request — country: %s, message_length: %d", req.country, len(req.message))

    async def generate() -> AsyncGenerator[bytes, None]:
        provider = "gemini"
        try:
            async for text in stream_gemini(system_prompt, req.history, req.message):
                yield f"data: {json.dumps({'text': text, 'done': False, 'provider': 'gemini'})}\n\n"
                await asyncio.sleep(0)

        except Exception as gemini_error:
            err = str(gemini_error)
            logger.warning("Gemini error: %s", err[:120])

            is_quota_error = any(k in err for k in ["429", "quota", "rate", "RESOURCE_EXHAUSTED"])

            if is_quota_error and GROQ_API_KEY:
                provider = "groq"
                logger.info("Quota exceeded — switching to Groq fallback")
                try:
                    async for text in stream_groq(system_prompt, req.history, req.message):
                        yield f"data: {json.dumps({'text': text, 'done': False, 'provider': 'groq'})}\n\n"
                        await asyncio.sleep(0)
                except Exception as groq_error:
                    logger.error("Groq fallback failed: %s", str(groq_error))
                    msg = "Both AI providers are temporarily unavailable. Please try again in a few minutes."
                    yield f"data: {json.dumps({'text': msg, 'done': True})}\n\n"
                    return
            else:
                logger.error("Non-quota Gemini error: %s", err)
                msg = "AI assistant temporarily unavailable. Please try again shortly."
                yield f"data: {json.dumps({'text': msg, 'done': True})}\n\n"
                return

        yield f"data: {json.dumps({'text': '', 'done': True, 'provider': provider})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health() -> dict:
    """Health check endpoint for Cloud Run startup probe."""
    return {
        "status": "ok",
        "service": "VoteGuide AI",
        "version": __version__,
        "primary_model": GEMINI_MODEL,
        "fallback_model": GROQ_MODEL,
        "supported_countries": list(COUNTRY_CONTEXTS.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))