# Deploying to Hugging Face Spaces

## Quick Deploy

1. **Create a new Space** at https://huggingface.co/new-space
   - Choose "Gradio" as the SDK
   - Set to "Public" or "Private" as needed

2. **Upload all files** from this `huggingface/` directory to your Space:
   - `app.py`
   - `graph.py`
   - `requirements.txt`
   - `README.md`
   - `config/` folder (agents.yaml, tasks.yaml)
   - `tools/` folder (__init__.py)
   - Logo images (optional)

3. **Set Environment Variables** in Space Settings → Variables:
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   SERPER_API_KEY=...
   ```

4. **Build will auto-trigger** when you commit files. Check the logs for any errors.

## File Structure

```
your-space/
├── app.py                    # Gradio UI (entry point)
├── graph.py                  # LangGraph pipeline
├── requirements.txt          # Dependencies
├── README.md                 # Space description (with YAML header)
├── config/
│   ├── agents.yaml          # Agent system prompts
│   └── tasks.yaml           # Task descriptions
└── tools/
    └── __init__.py          # search_serper() function
```

## Important Notes

- **Do NOT commit `.env`** - use HF Space environment variables instead
- The app creates an `output/` directory at runtime for generated reports
- PDF generation requires `reportlab` (included in requirements.txt)
- Default models: GPT-4o for scan/write, Claude Sonnet for analysis/eval

## Troubleshooting

- **Build fails**: Check requirements.txt versions match your local env
- **Runtime errors**: Verify all 3 API keys are set in Space settings
- **Import errors**: Ensure `config/` and `tools/` folders uploaded correctly
- **Theme warning**: Gradio 6.0 deprecation warning is harmless (line 283 in app.py)

## Local Testing

Before deploying:
```bash
cd huggingface
pip install -r requirements.txt
python app.py
```
