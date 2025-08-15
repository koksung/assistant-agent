# Master Orchestrator - Academic Paper Summarizer & Research Agent

This system summarizes academic papers using a LangChain-based orchestrator with belief-driven reasoning, tool invocation, and adaptive strategies based on user goals.

## Features
- PDF summarization and deep-dive analysis
- Equation parsing and image captioning
- Belief vector system with Bayesian updates
- Tool registry (local & remote/MCP)
- FastAPI-based API service

## Running the App

```bash
uvicorn app.main:app --reload
