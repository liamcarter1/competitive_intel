---
title: Competitive Intelligence Monitor
emoji: 📊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
---

# Competitive Intelligence Monitor

A LangGraph-powered competitive intelligence platform with Gradio UI.

## Features

- **Briefing Pipeline**: Generate executive competitive intelligence briefings
- **Annual Report Analysis**: Deep-dive competitor analysis with 16-section reports
- **Chat Interface**: Quick Q&A against generated briefings
- **Deep-dive Research**: Live web search with AI synthesis
- **PDF Export**: Download reports as formatted PDFs

## Environment Variables Required

Set these in your Hugging Face Space settings:

- `OPENAI_API_KEY` - For scan nodes and quick chat
- `ANTHROPIC_API_KEY` - For analysis, recommendations, and evaluation
- `SERPER_API_KEY` - For web search

## Tech Stack

- **LangGraph** - Pipeline orchestration with fan-out/fan-in
- **Gradio** - Web UI
- **OpenAI GPT-4o** - Scan and formatting nodes
- **Anthropic Claude Sonnet** - Analysis, recommendations, evaluation
- **Serper API** - Web search
