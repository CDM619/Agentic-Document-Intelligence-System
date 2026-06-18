# Agentic Document Intelligence System

## Overview

The Agentic Document Intelligence System is an AI-powered document question-answering platform that combines Retrieval-Augmented Generation (RAG), hybrid search, reranking, and AI agents to provide accurate answers from uploaded documents.

Unlike traditional RAG systems, this project integrates tool-calling agents and LangGraph workflows, enabling intelligent decision-making, web search capabilities, and extensible multi-step reasoning.

---

## Features

### Document Intelligence

* PDF document ingestion
* Intelligent document chunking
* Semantic retrieval using embeddings
* Vector database storage using ChromaDB

### Hybrid Search

* Dense retrieval using embeddings
* Sparse retrieval using BM25
* Hybrid retrieval pipeline
* Max Marginal Relevance (MMR) search

### Reranking

* CrossEncoder reranking
* Improved retrieval precision
* Top-k relevant chunk selection

### AI Agent

* Tool-calling support
* Document retrieval tool
* Calculator tool
* Web search tool
* ReAct reasoning workflow

### LangGraph Workflow

* State-based workflow execution
* Node and edge architecture
* Extensible routing system
* Foundation for memory and multi-agent workflows

### User Interface

* Streamlit web application
* Interactive chat interface
* Document upload support

---

## Architecture

User Query

↓

Hybrid Retrieval

├── Embedding Search (ChromaDB)

└── BM25 Search

↓

Hybrid Merge

↓

CrossEncoder Reranking

↓

Relevant Context

↓

Agent / LangGraph Workflow

↓

LLM (Mistral via Ollama)

↓

Final Answer

---

## Technology Stack

### LLM

* Mistral
* Ollama

### Frameworks

* LangChain
* LangGraph

### Retrieval

* ChromaDB
* BM25
* Sentence Transformers

### Embeddings

* all-MiniLM-L6-v2

### Reranking

* cross-encoder/ms-marco-MiniLM-L-6-v2

### Frontend

* Streamlit

---

## Project Structure

```text
Agentic-Document-Intelligence-System/

├── app.py
├── agent.py
├── tools.py
├── requirements.txt
├── chroma_db/
├── data/
├── README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Agentic-Document-Intelligence-System.git

cd Agentic-Document-Intelligence-System
```

### Create Virtual Environment

```bash
python -m venv rag_env
```

### Activate Environment

Windows:

```bash
rag_env\Scripts\activate
```

Linux / Mac:

```bash
source rag_env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pull Mistral Model

```bash
ollama pull mistral
```

### Run Application

```bash
streamlit run app.py
```

---

## Current Capabilities

* Retrieval-Augmented Generation (RAG)
* Hybrid Search (Vector + BM25)
* CrossEncoder Reranking
* Agent Tool Calling
* Web Search Integration
* Calculator Tool
* LangGraph Workflows

---

## Roadmap

### Version 2

* LangGraph Conditional Routing
* Agent Memory
* Conversation Persistence

### Version 3

* Multi-Agent Architecture
* Query Planning Agent
* Research Agent

### Version 4

* FastAPI Backend
* Docker Deployment
* Cloud Deployment

---

## Learning Objectives

This project was built to explore:

* Retrieval-Augmented Generation (RAG)
* Hybrid Retrieval Systems
* CrossEncoder Reranking
* Tool Calling Agents
* LangGraph Workflows
* Agent Memory
* Multi-Agent Systems
* Production GenAI Architecture

---

## Author

Chetan Misquith

Aspiring AI Engineer focused on Generative AI, RAG Systems, Agents, and Applied Machine Learning.
