# AI Virtual Agent

Simple Streamlit app that provides a chat interface to an LLM (OpenAI).

## Quick start (local)

1. Create a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file from the example and fill in your keys:

```powershell
copy .env.example .env
```

Then edit `.env` and set:

```text
OPENAI_API_KEY=sk-...
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

4. Run the app:

```powershell
streamlit run app.py
```

Open http://localhost:8501

## Deploy to Streamlit Community Cloud

- Push this repo to GitHub.
- On share.streamlit.io create a new app, point to the repo, branch `main`, main file `app.py`.
- Add `OPENAI_API_KEY` in Advanced Settings → Secrets.

## AWS Nova integration

The app can start an AWS Nova workflow using the `nova-act` service.

- Ensure AWS credentials are available as environment variables or Streamlit secrets:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - optionally `AWS_SESSION_TOKEN`
- In the app sidebar, enter the Nova workflow name (default `premkumar`) and AWS region.
- Click `Start Nova workflow run` to invoke `CreateWorkflowRun`.

## Next steps / Enhancements

- PDF upload and Q&A
- CSV analysis
- Add conversation persistence (database)
- Add user authentication

## Retrieval (RAG) instructions

1. Create a FAISS index from a dataset or local text file:

```powershell
# set your OpenAI key locally
setx OPENAI_API_KEY "sk-..."
python rag_ingest.py --dataset openwebtext --limit 1000 --out_dir rag_index
```

2. In the app sidebar enable "Enable retrieval-augmented generation (RAG)" and set the index path to `rag_index/index.faiss` and metadata to `rag_index/meta.jsonl`.

3. Ask questions in the chat. The app will retrieve top-k chunks and include them in the prompt to the LLM.

Notes:
- Start with a small `--limit` (e.g., 500-2000 documents) for development.
- The ingestion script uses OpenAI embeddings; embedding costs apply.

