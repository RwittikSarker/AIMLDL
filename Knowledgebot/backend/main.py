import json
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types as genai_types

KB_FILE           = Path(__file__).parent / "knowledge_base.json"
FRONTEND_DIR      = Path(__file__).parent.parent / "frontend"
MATCH_THRESHOLD   = 0.15   # similarity score below this → "out of scope"
TOP_K             = 4      # how many KB chunks to send to Gemini
GEMINI_MODEL      = "gemini-flash-latest"


app = FastAPI(title="KnowledgeBot API")

# Allow the frontend (running as a local file or different port) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


kb_entries   = []   # list of {"title": str, "content": str}
vectorizer   = None # TfidfVectorizer fitted on KB
tfidf_matrix = None # sparse matrix  (n_entries × n_features)
api_key      = None # Google AI Studio key supplied via /api/configure


def load_kb_from_disk():
    global kb_entries
    if KB_FILE.exists():
        kb_entries = json.loads(KB_FILE.read_text(encoding="utf-8"))
    else:
        kb_entries = []
    rebuild_index()

def save_kb_to_disk():
    KB_FILE.write_text(json.dumps(kb_entries, indent=2, ensure_ascii=False), encoding="utf-8")

def rebuild_index():
    """Re-fit TF-IDF on the current kb_entries list."""
    global vectorizer, tfidf_matrix
    if not kb_entries:
        vectorizer   = None
        tfidf_matrix = None
        return
    corpus = [f"{e['title']} {e['content']}" for e in kb_entries]
    vectorizer   = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

def search(query: str):
    """Return top-K KB entries whose similarity score ≥ MATCH_THRESHOLD."""
    if vectorizer is None:
        return []
    q_vec  = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    ranked = np.argsort(scores)[::-1]
    hits   = []
    for idx in ranked[:TOP_K]:
        if scores[idx] >= MATCH_THRESHOLD:
            hits.append({
                "title":   kb_entries[idx]["title"],
                "content": kb_entries[idx]["content"],
                "score":   round(float(scores[idx]), 3),
            })
    return hits


class ConfigureRequest(BaseModel):
    api_key: str

class ChatRequest(BaseModel):
    message: str
    history: list = []   # list of {"role": "user"|"model", "content": "..."}

class KBEntryModel(BaseModel):
    title:   str
    content: str

@app.get("/")
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Backend is running. Open frontend/index.html in your browser."}

@app.post("/api/configure")
def configure(req: ConfigureRequest):
    """Save the Google AI Studio API key and verify it with a test ping."""
    global api_key
    key = req.api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")
    # Quick verification
    try:
        client = genai.Client(api_key=key)
        client.models.generate_content(model=GEMINI_MODEL, contents="hi")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Key rejected by Google: {e}")
    api_key = key
    return {"ok": True, "message": "API key saved and verified."}

@app.get("/api/configure/status")
def configure_status():
    return {"configured": api_key is not None}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not api_key:
        raise HTTPException(status_code=400, detail="No API key set. Call POST /api/configure first.")

    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message is empty.")

    # 1. Search KB
    hits     = search(query)
    in_scope = len(hits) > 0

    # 2. Build system prompt
    if in_scope:
        context = "\n\n".join(f"[{h['title']}]\n{h['content']}" for h in hits)
        system  = (
            "You are a helpful assistant. "
            "Answer the user's question using ONLY the knowledge base excerpts below. "
            "If the excerpts don't fully answer the question, say so — never make things up.\n\n"
            "KNOWLEDGE BASE:\n" + context
        )
    else:
        system = (
            "You are a helpful assistant with a custom knowledge base. "
            "The user's question does not match anything in the knowledge base. "
            "Politely tell them you couldn't find relevant information, "
            "and suggest they either rephrase the question or add the topic to the knowledge base. "
            "Do NOT answer from your general knowledge."
        )

    # 3. Build conversation history for Gemini
    contents = []
    for turn in req.history:
        contents.append(
            genai_types.Content(
                role=turn["role"],
                parts=[genai_types.Part(text=turn["content"])],
            )
        )
    contents.append(
        genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
    )

    # 4. Call Gemini
    try:
        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=1024,
            ),
        )
        answer = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

    return {
        "answer":   answer,
        "in_scope": in_scope,
        "sources":  [{"title": h["title"], "score": h["score"]} for h in hits],
    }


@app.get("/api/kb")
def get_kb():
    return {"entries": kb_entries, "count": len(kb_entries)}

@app.post("/api/kb")
def add_entry(entry: KBEntryModel):
    title   = entry.title.strip()
    content = entry.content.strip()
    if not title or not content:
        raise HTTPException(status_code=400, detail="Title and content are required.")
    kb_entries.append({"title": title, "content": content})
    save_kb_to_disk()
    rebuild_index()
    return {"ok": True, "total": len(kb_entries)}

@app.put("/api/kb/{index}")
def update_entry(index: int, entry: KBEntryModel):
    if not (0 <= index < len(kb_entries)):
        raise HTTPException(status_code=404, detail="Entry not found.")
    kb_entries[index] = {"title": entry.title.strip(), "content": entry.content.strip()}
    save_kb_to_disk()
    rebuild_index()
    return {"ok": True}

@app.delete("/api/kb/{index}")
def delete_entry(index: int):
    if not (0 <= index < len(kb_entries)):
        raise HTTPException(status_code=404, detail="Entry not found.")
    kb_entries.pop(index)
    save_kb_to_disk()
    rebuild_index()
    return {"ok": True, "total": len(kb_entries)}

@app.post("/api/kb/bulk")
def bulk_import(payload: dict):
    """
    Expects: { "entries": [ {"title": "...", "content": "..."}, ... ] }
    """
    added = 0
    for e in payload.get("entries", []):
        t = str(e.get("title",   "")).strip()
        c = str(e.get("content", "")).strip()
        if t and c:
            kb_entries.append({"title": t, "content": c})
            added += 1
    save_kb_to_disk()
    rebuild_index()
    return {"ok": True, "added": added, "total": len(kb_entries)}

@app.delete("/api/kb")
def clear_kb():
    kb_entries.clear()
    save_kb_to_disk()
    rebuild_index()
    return {"ok": True}

load_kb_from_disk()

if __name__ == "__main__":
    print("\n  KnowledgeBot backend starting...")
    print("  API docs → http://localhost:8000/docs")
    print("  Frontend → open frontend/index.html in your browser\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
