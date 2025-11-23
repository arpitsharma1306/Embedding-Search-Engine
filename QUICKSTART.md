# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare Data

```bash
python main.py --prepare-data
```

This downloads the 20 Newsgroups dataset and saves it to `data/docs/`.

## 3. Start the API

```bash
uvicorn src.api:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## 4. Test Search

### Using curl:
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "computer graphics", "top_k": 3}'
```

### Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "machine learning", "top_k": 5}
)
print(response.json())
```

## 5. Use Streamlit UI (Optional)

```bash
streamlit run src/ui.py
```

## Alternative: Use CLI

```bash
python main.py --query "your search query" --top-k 5
```

## Troubleshooting

- **No documents found**: Run `python main.py --prepare-data` first
- **Import errors**: Make sure you're in the project root directory
- **Model download slow**: First run downloads ~80MB model, be patient
- **Cache issues**: Delete `embeddings_cache.db` to clear cache

