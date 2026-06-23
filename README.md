# SmartStock Intelligence Platform

AI-Powered Stock Market Analytics, Research & Decision Intelligence System.

## Overview

SmartStock Intelligence Platform is an end-to-end stock market analytics and
AI-powered research platform combining data engineering, financial analytics,
machine learning, NLP, explainable AI, generative AI, and retrieval-augmented
generation (RAG). It helps users analyze stocks, understand market sentiment,
evaluate risk, receive AI-generated insights, and interact with company reports
through natural language.

## Project Status

**Current Phase:** Planning & Architecture


## Folder Structure

```
smartstock-intelligence-platform/
├── data/            # raw, processed, external, reports
├── notebooks/        # eda, sentiment_analysis, ml_experiments, feature_engineering
├── database/         # schema.sql, tables.sql, views.sql
├── src/               # ingestion, etl, analytics, sentiment, ml, explainability, genai, rag, api, utils
├── dashboard/         # Streamlit multi-page app
├── models/            # serialized trained models
├── vector_db/         # ChromaDB persistence
├── tests/             # unit/integration tests
├── docs/              # planning documents (Phase 0)
├── requirements.txt
├── README.md
└── docker-compose.yml
```

