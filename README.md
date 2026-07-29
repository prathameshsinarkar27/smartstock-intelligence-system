# SmartStock Intelligence Platform

**AI-powered stock market analytics, research, and decision intelligence system.**

SmartStock combines data engineering, financial analytics, machine learning, NLP, explainable AI, generative AI, and retrieval-augmented generation (RAG) into one end-to-end platform. It helps users analyze stocks, understand market sentiment, evaluate risk, receive AI-generated research summaries, and interact with a company's actual annual report through natural language.

---

## Features

- **Market Overview dashboard** — KPIs, top gainers/losers, sector performance, live-updating price ticker, latest market news
- **Company Detail pages** — 3-panel interactive technical charts (SMA/EMA/Bollinger Bands, RSI, MACD), news & sentiment, ML predictions with SHAP explanations, AI-generated research summaries
- **Sentiment analysis** of company news (VADER)
- **ML trend & risk predictions** — ensembled Random Forest + XGBoost, chronologically validated
- **Explainable AI** — SHAP feature-contribution breakdowns for every prediction
- **AI Research Assistant** — Gemini-powered outlook, summary, and key considerations per company
- **RAG annual report chatbot** — ask natural-language questions grounded in a company's ingested 10-K/annual report PDF
- **Portfolio Analyzer** — real positions with shares/cost basis, live P&L, sector concentration
- **JSON API** — every feature above also available as a documented REST API (`/docs`)
- **Automatic scheduler** — data, sentiment, and predictions refresh themselves on a schedule, no manual runs required
- **Dockerized** — one command to run the full stack anywhere

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Database | PostgreSQL |
| Data processing | Pandas, NumPy |
| Backend | FastAPI |
| Frontend | Jinja2, Bootstrap 5, Plotly.js |
| Machine Learning | scikit-learn, XGBoost |
| NLP | vaderSentiment |
| Explainability | SHAP |
| Generative AI | Google Gemini (`google-genai`) |
| RAG | ChromaDB, pypdf, LangChain (text splitters) |
| Scheduling | APScheduler (Docker) / Windows Task Scheduler (local) |
| Containerization | Docker, Docker Compose |
| Deployment | AWS EC2, Render |

---

## Architecture

```
Stock Market APIs (Twelve Data, Finnhub, NewsAPI)
            │
            ▼
   Data Ingestion + ETL Pipelines
            │
            ▼
        PostgreSQL
            │
   ┌────────┼─────────────┬──────────────┐
   ▼        ▼              ▼              │
Analytics  ML Engine    NLP Engine        │
Engine    (+ SHAP)     (Sentiment)        │
   │        │              │              │
   └───┬────┴──────┬───────┘              │
       ▼           ▼                      │
  AI Research   RAG System  ◄──────────────
  Assistant     (ChromaDB)
       │
       ▼
  FastAPI (HTML dashboard + JSON API)
       │
       ▼
  Docker Compose (db + app + scheduler)
       │
       ▼
  AWS EC2 → Render
```

Everything reads from and writes to one shared PostgreSQL database. Each `src/` package has a single responsibility and talks to the database, not directly to other packages — no message queue, no microservices, just a well-organized monolith.

---

## Project Structure

```
smartstock-intelligence-platform/
├── config/                 Tracked symbol list
├── data/                   raw/, processed/, external/, reports/ (RAG PDFs)
├── notebooks/               EDA notebooks
├── database/                Schema (tables.sql, views.sql) + migrations/
├── models/                   Trained ML models
├── vector_db/                ChromaDB persistent index
├── src/
│   ├── ingestion/             Raw data fetchers (prices, company data, news)
│   ├── etl/                   Clean, transform, load into Postgres
│   ├── pipeline/               Single-command pipeline orchestrator
│   ├── analytics/               KPIs, technical indicators, portfolio metrics
│   ├── sentiment/                News sentiment scoring
│   ├── ml/                       Feature engineering, training, prediction
│   ├── explainability/            SHAP explanations
│   ├── genai/                     AI Research Assistant
│   ├── rag/                        Annual report ingestion + Q&A
│   ├── scheduler/                   Automatic recurring jobs
│   ├── api/                          FastAPI app (routes, services, templates, schemas)
│   └── utils/                        Config, database, logging
├── scripts/windows/          Windows Task Scheduler wrapper scripts
├── tests/                     Automated tests
├── docs/                       Setup/Testing/Commit guide per phase
├── Dockerfile, docker-compose.yml, .dockerignore, requirements-docker.txt
├── requirements.txt
└── .env.example
```

---

## Getting Started

### Option A — Docker (recommended)

```bash
git clone <this-repo-url>
cd smartstock-intelligence-platform
cp .env.example .env
# fill in your API keys + a non-empty POSTGRES_PASSWORD

docker compose up -d --build
```

Load your first data:
```bash
docker compose exec app python -m src.pipeline.run_pipeline
docker compose exec app python -m src.sentiment.sentiment_pipeline
docker compose exec app python -m src.ml.train_model
docker compose exec app python -m src.ml.predict
```

Open **http://localhost:8000**.

### Option B — Local (no Docker)

```bash
git clone <this-repo-url>
cd smartstock-intelligence-platform
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cp .env.example .env
# fill in your API keys + PostgreSQL connection details

psql -U postgres -d smartstock -f database/tables.sql
psql -U postgres -d smartstock -f database/views.sql

python -m src.pipeline.run_pipeline
python -m src.sentiment.sentiment_pipeline
python -m src.ml.train_model
python -m src.ml.predict

uvicorn src.api.main:app --reload
```

Open **http://127.0.0.1:8000**.

---

## API Keys Required

| Service | Used for | Get one at |
|---|---|---|
| Twelve Data | Historical stock prices | https://twelvedata.com/pricing (free plan) |
| Finnhub | Company profile/fundamentals | https://finnhub.io/register |
| NewsAPI.org | News articles | https://newsapi.org/register |
| Google AI Studio (Gemini) | AI Research Assistant + RAG | https://aistudio.google.com/app/apikey |

Only one Gemini key is needed — it powers both the AI Research Assistant and RAG.

---

## Configuration

Every setting is an environment variable — see `.env.example` for the full, documented list, including:
- Database connection (`POSTGRES_*`)
- API keys (`FINNHUB_API_KEY`, `TWELVEDATA_API_KEY`, `NEWSAPI_API_KEY`, `GEMINI_API_KEY`)
- RAG embedding rate limits (`RAG_EMBED_*`)
- Automatic scheduler timing (`SCHEDULER_*`)
- Docker host ports (`POSTGRES_HOST_PORT`, `APP_HOST_PORT`)

---

## Automatic Scheduler

Data, sentiment, and predictions refresh themselves — no manual script runs required day to day.

- **Daily job**: data ingestion + ETL, sentiment scoring, ML predictions (current model)
- **Weekly job**: full model retraining, evaluation, fresh predictions

**Docker:** runs automatically via the `scheduler` service — nothing extra to set up.

**Windows (no Docker):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\register_scheduled_tasks.ps1
```

---

## JSON API

Every dashboard feature is also available as JSON under `/api/...`:

```
GET  /api/stocks                        GET  /api/sentiment/{symbol}
GET  /api/stocks/{symbol}                GET  /api/predict/{symbol}
GET  /api/stocks/{symbol}/prices          GET  /api/assistant/{symbol}
GET  /api/news                            POST /api/assistant/{symbol}/ask
GET  /api/company/{symbol}                GET/POST/DELETE /api/portfolio...
```

Interactive docs (try every endpoint from the browser): **http://localhost:8000/docs**

---

## Testing

```bash
python -m pytest tests/ -v
```

Every phase's tests run against mocked database/API boundaries but exercise real code paths (FastAPI's `TestClient`, not just isolated function calls).

---

## Deployment

Documented for two targets in `docs/PHASE_14_SETUP_GUIDE.md`:

1. **AWS EC2** — a single instance running `docker-compose.yml` directly. Full walkthrough: instance sizing, Docker install, deploy, and recommended hardening (reverse proxy + HTTPS, closing the database port, backups).
2. **Render** — managed Postgres + a Web Service (`app`) + a Background Worker (`scheduler`), with the full `docker-compose.yml` → Render service mapping since Render doesn't run Compose files directly.

---

