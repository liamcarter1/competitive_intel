---
title: Competitive Intelligence Monitor
emoji: 🔍
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
license: mit
suggested_hardware: cpu-upgrade
suggested_storage: small
---

# Danfoss Power Solutions — Competitive Intelligence Monitor

A LangGraph-powered competitive intelligence platform that generates executive-ready briefings by analyzing competitors across news, financials, and strategic moves.

## ⚠️ Hardware Requirements

**Required**: CPU Upgrade (2 vCPU, 16GB RAM) - **Free tier will crash with OOM error**

This app requires more resources than the free tier due to:
- Multiple LLM API calls with large context windows
- Parallel competitor scanning (fan-out architecture)
- Heavy dependencies (LangChain, LangGraph, multiple AI SDKs)
- 112 search results per competitor (14 searches × 8 results each)

**To upgrade**: Space Settings → Hardware → Select "CPU Upgrade"

## Features

- **Dual-Pipeline Analysis**:
  - Competitive Briefing: Recent news, product launches, pricing changes
  - Annual Report Deep Dive: Financial health, market share, M&A, patents, hiring

- **Dual-Endpoint Search Strategy**:
  - 9 news searches (Serper /news, past month filter)
  - 5 web searches (Serper /search, broader context)
  - ~112 search results per competitor analyzed

- **Real-Time Progress Streaming**: See live updates as each pipeline node completes

- **Quality Gate System**: Automatic evaluation and retry of analysis and recommendations

- **Interactive Q&A**: Chat with reports or run deep-dive web research

## Setup

Set these **Secrets** in Space Settings → Repository secrets:

- `OPENAI_API_KEY` - For scan nodes and quick chat
- `ANTHROPIC_API_KEY` - For analysis, recommendations, and evaluation
- `SERPER_API_KEY` - For web search

## Usage

1. Enter your company name, industry, and competitors
2. Choose a pipeline:
   - **Generate Briefing**: Weekly intelligence briefing
   - **Annual Report Analysis**: Deep financial and strategic analysis
3. Download results as PDF
4. Ask questions via Quick Chat or Research This (with live web search)

## Rate Limits

- Quick Chat: 10 requests/hour per IP
- Deep Dive Research: 3 requests/hour per IP

## Tech Stack

- **LangGraph** - Pipeline orchestration with fan-out/fan-in
- **Gradio** - Web UI
- **OpenAI GPT-4o** - Scan and formatting nodes
- **Anthropic Claude Sonnet** - Analysis, recommendations, evaluation
- **Serper API** - Dual-endpoint web/news search
