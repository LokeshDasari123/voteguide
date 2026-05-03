# VoteGuide AI 🗳️
### Election Process Education Assistant — PromptWars Virtual Challenge 2

**Live Demo:** [Your Cloud Run URL here]  
**Built with:** Gemini 2.0 Flash · FastAPI · Google Cloud Run

---

## What It Does

VoteGuide AI is a conversational civic education assistant that helps citizens understand the democratic process in plain, simple language. It covers:

- Voter registration step-by-step
- Eligibility checks
- Polling day procedures
- Types of elections
- Official ID requirements
- Finding your polling station

Supports **India**, **USA**, and **UK** with country-specific official information.

---

## Prompt Engineering — Core Design

This is the heart of the application and the primary scoring criterion.

### System Prompt Architecture

The system prompt is structured in **three layers**:

#### Layer 1 — Identity & Context Injection
```
You are VoteGuide, an authoritative civic education assistant for {country}.
GOVERNING BODY: {body}
VOTING AGE: {voting_age}+
OFFICIAL REGISTRATION: {register_url}
```
Country context is dynamically injected at runtime, making the same base prompt work for India, USA, and UK without duplication.

#### Layer 2 — Strict Behavioral Rules (10 rules)
```
1. STRUCTURE: Always use numbered steps for any process question
2. TIMELINE: Always include how many days each step takes
3. LANGUAGE: Plain English only — 8th grade reading level
4. ELIGIBILITY: Confirm age first, then citizenship
5. ACCURACY: Only verified, official information
6. NEUTRAL: Zero political opinion
7. INTERACTIVE: End every answer with one follow-up question
8. FORMAT: Short paragraphs, max 3 sentences each
9. SOURCES: Always cite the official source
10. EMPOWERMENT: End with an encouraging line
```

#### Layer 3 — Few-Shot Example
A worked example of a perfect answer trains the model on the exact output format expected, dramatically improving consistency.

### Why This Prompt Wins

| Aspect | Our Approach | Generic Chatbot |
|--------|-------------|-----------------|
| Structure | Numbered steps with timelines | Paragraphs of text |
| Tone | 8th grade, official source citations | Varies, often too technical |
| Engagement | Follow-up question every time | Dead ends |
| Accuracy | Bound to official body + URL | General knowledge |
| Country | Dynamic context injection | Static or absent |

---

## Technical Architecture

```
User Browser
    │ HTTPS
    ▼
Google Cloud Run (asia-south1)
    │
    ├── FastAPI (main.py)
    │     ├── GET  /          → HTML frontend (Jinja2)
    │     ├── POST /chat/stream → Server-Sent Events (streaming)
    │     └── GET  /health    → Health check
    │
    └── Gemini 2.0 Flash API
          └── System prompt + conversation history
```

### Streaming Architecture
Responses stream token-by-token using **Server-Sent Events (SSE)**, making the AI feel instant and alive rather than waiting for a full response.

---

## Setup & Deployment

### Prerequisites
- Google Cloud account
- Project: `voteguide-495017`
- Google AI Studio API key

### Deploy to Cloud Run (5 commands)

```bash
# 1. Open Google Cloud Shell at console.cloud.google.com

# 2. Clone / upload the project
git clone https://github.com/YOUR_USERNAME/voteguide.git
cd voteguide

# 3. Set your project
gcloud config set project voteguide-495017

# 4. Deploy (one command does everything)
gcloud run deploy vote-guide \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=YOUR_KEY_HERE \
  --port 8080

# 5. Copy the URL from the output → paste in Hack2skill dashboard
```

### Local Development

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
uvicorn main:app --reload --port 8080
# Open http://localhost:8080
```

---

## Project Structure

```
voteguide/
├── main.py              # FastAPI app + Gemini integration + prompts
├── requirements.txt     # Dependencies
├── Dockerfile           # Container config for Cloud Run
└── templates/
    └── index.html       # Full frontend (HTML/CSS/JS, streaming chat)
```

---

## Judging Criteria Alignment

| Criterion | Implementation |
|-----------|---------------|
| **Prompt Quality** | 10-rule system prompt + dynamic country context + few-shot example |
| **Live App** | Deployed on Cloud Run, streaming responses, mobile responsive |
| **GitHub Docs** | This README with full prompt logic explained |
| **LinkedIn** | Posted with #PromptWars tag |

---

*Built solo for PromptWars Virtual — Challenge 2: Election Process Education*