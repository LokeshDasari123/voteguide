import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="VoteGuide AI")
templates = Jinja2Templates(directory="templates")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

COUNTRY_CONTEXTS = {
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

SYSTEM_PROMPT = """You are VoteGuide, an authoritative civic education assistant for {country}.

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
    country: str
    message: str
    history: list = []


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "countries": list(COUNTRY_CONTEXTS.keys())
    })


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    ctx = COUNTRY_CONTEXTS.get(req.country, COUNTRY_CONTEXTS["India"])
    system_prompt = SYSTEM_PROMPT.format(country=req.country, **ctx)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system_prompt
    )

    history = [
        {
            "role": "model" if h["role"] == "assistant" else h["role"],
            "parts": [h["content"]]
        }
        for h in req.history
    ]
    chat = model.start_chat(history=history)

    async def generate():
        try:
            response = chat.send_message(req.message, stream=True)
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'text': chunk.text, 'done': False})}\n\n"
                    await asyncio.sleep(0)
            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
        except Exception as e:
            err_msg = "Our AI is temporarily unavailable due to high demand. Please try again in a few minutes. The prompt logic and full source code is available on GitHub."
            yield f"data: {json.dumps({'text': err_msg, 'done': True})}\n\n"
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "VoteGuide AI",
        "version": "1.0.0",
        "model": "gemini-2.0-flash"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)