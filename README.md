# Exam Assistant

An exam preparation assistant built with **LangChain** and **LangGraph**. It answers questions from your materials (PDF, Word, PNG), generates practice tests, grades your answers, and uses web search as a fallback when documents are insufficient.

---

## How to run and use the agent

### 1. Prerequisites

- **Python 3.10+**
- **OpenAI** or **Azure OpenAI** (see below)

**Option A – OpenAI**

Set your key in `.env` (project root or `app/`):

```env
OPENAI_API_KEY=sk-your-key-here
```

**Option B – Azure OpenAI**

If `AZURE_OPENAI_ENDPOINT` or `AZURE_OPENAI_API_KEY` is set, the app uses Azure. In `.env`:

```env
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

Use your Azure deployment names for the chat and embedding models. The app chooses Azure automatically when these variables are present.

### 2. Install dependencies

From the project root (`exam-assitant`):

```powershell
pip install -r requirements.txt
```

### 3. Add your study materials

Put your exam resources in **`data/raw/`**:

- **PDF** (`.pdf`)
- **Word** (`.docx`)
- **Images** (`.png`) – text is extracted via OCR

Example:

```
data/raw/
  my-notes.pdf
  chapter-1.docx
  diagram.png
```

### 4. Ingest documents (build the vector store)

One-time (or re-run when you add/change files):

```powershell
python -m app.main ingest
```

This loads all PDF/DOCX/PNG under `data/raw/`, chunks them, embeds with OpenAI, and saves a Chroma index to **`data/vectorstore/`**. You must run this before asking questions or generating practice tests.

### 5. Ask the agent a question (exam graph)

Uses the **LangGraph** flow: retrieve context → answer → evaluate confidence → retry with more context or **DuckDuckGo web search** if needed.

```powershell
python -m app.main ask "What is dimensional modeling?"
```

You get an answer plus a confidence score. Low-confidence answers trigger a retry with more docs, then web search as fallback.

### 6. Generate a practice test

Creates **5 exam-style questions** from your documents (no answers shown). If you have weak topics recorded, it prioritizes those.

```powershell
python -m app.main practice
```

Optional: focus on a topic:

```powershell
python -m app.main practice --topic "Microsoft Fabric"
```

Questions are printed; the correct answers are stored for grading.

### 7. Get your answers graded (evaluation mode)

After running **practice**, submit your answers so the agent **grades and explains**:

**Option A – command line:**

```powershell
python -m app.main evaluate --answers "First answer" "Second answer" "Third" "Fourth" "Fifth"
```

**Option B – stdin (one answer per line):**

```powershell
python -m app.main evaluate
# Then type or paste 5 lines, one per question; Ctrl+Z then Enter (Windows) to finish
```

You get a **score (0–100%)** and **feedback** (per-question grade and explanation).

---

## Command summary

| Command | What it does |
|--------|-------------------------------|
| `python -m app.main ingest` | Load `data/raw/` into the vector store (required before ask/practice). |
| `python -m app.main ask "Your question?"` | Run the exam graph: answer using your docs + optional web fallback. |
| `python -m app.main practice [--topic TOPIC]` | Generate 5 practice questions (optionally focused on a topic). |
| `python -m app.main evaluate [--answers A1 A2 A3 A4 A5]` | Grade the last practice test; show score and feedback. |

---

## Project structure

- **`app/`** – main code: ingest, agents, graph, tools, prompts.
- **`data/raw/`** – your PDF/DOCX/PNG files (gitignored).
- **`data/processed/`** – last practice Q&A and weak topics (for revision).
- **`data/vectorstore/`** – Chroma index (created by `ingest`).
- **`requirements.txt`** – Python dependencies.
