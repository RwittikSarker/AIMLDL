# KnowledgeBot

AI chatbot that answers ONLY from your custom knowledge base, powered by Google Gemini.

---

## Quickstart (3 steps)

### 1. Install dependencies
```
cd backend
pip install -r requirements.txt
```

### 2. Start the backend
```
python main.py
```
You should see:
```
KnowledgeBot backend starting...
API docs → http://localhost:8000/docs
```

### 3. Open the frontend
Open `frontend/index.html` in your browser.
(Just double-click it — no server needed for the frontend.)

---

## First-time setup in the UI

1. Click **⚙ Settings** in the top-right
2. Paste your Google AI Studio API key (`AIza...`)
3. Click **Save & Test** — wait for the green "Connected" badge
4. Add knowledge entries in the right panel
5. Start chatting!

Get a free API key: https://aistudio.google.com/app/apikey

---

## Project structure

```
kb-chatbot/
├── backend/
│   ├── main.py               ← FastAPI server (single file, heavily commented)
│   ├── requirements.txt
│   └── knowledge_base.json   ← auto-created, edit directly if you want
└── frontend/
    └── index.html            ← complete UI in one file
```

---

## API endpoints (for debugging)

Open http://localhost:8000/docs for the interactive docs.

| Method | Endpoint              | What it does                        |
|--------|-----------------------|-------------------------------------|
| POST   | /api/configure        | Save + verify API key               |
| GET    | /api/configure/status | Check if key is set                 |
| POST   | /api/chat             | Send a message, get an answer       |
| GET    | /api/kb               | List all knowledge base entries     |
| POST   | /api/kb               | Add one entry                       |
| PUT    | /api/kb/{index}       | Edit an entry                       |
| DELETE | /api/kb/{index}       | Delete one entry                    |
| POST   | /api/kb/bulk          | Import many entries from JSON       |
| DELETE | /api/kb               | Clear the entire knowledge base     |

---

## How it works (simple explanation)

1. You add text entries to the knowledge base (title + content).
2. When you ask a question, the backend searches those entries using **TF-IDF cosine similarity**.
3. The top matching entries are passed to **Gemini** as context.
4. Gemini answers using ONLY that context — no hallucination from general knowledge.
5. If nothing matches (score below threshold), a polite fallback is returned instead.

---

## Bulk import format

Paste JSON like this in the Bulk Import box:

```json
[
  {
    "title": "Refund Policy",
    "content": "Customers may request a full refund within 30 days of purchase by contacting support@example.com."
  },
  {
    "title": "Shipping Times",
    "content": "Standard shipping is 5–7 business days. Express is 1–2 business days."
  }
]
```
