# Ask the Doc — Retrieval-Augmented Q&A over Uploaded Documents

A Streamlit app that lets a user upload a text document, ask a natural-language question about it, and get an answer grounded in that document rather than in the model's training data.

This is the **first** of two RAG projects in my portfolio. It uses a hosted LLM (OpenAI) and a managed retrieval chain. The follow-up, [`agentic-ai-hitl`](https://github.com/SaintClair33/agentic-ai-hitl), rebuilds the same idea with **no API keys** using local Hugging Face models, and adds agent planning with human-in-the-loop approval and persistent memory. Read together they show the progression from *calling a hosted API* to *operating a model myself*.

## What problem it solves

An LLM on its own cannot answer questions about a document it has never seen, and it will often invent an answer instead of admitting that. Retrieval-augmented generation fixes this by fetching the relevant passages from the user's own document and giving them to the model as context, so the answer is traceable to the source material.

## How it works

The full pipeline runs in `app.py`:

1. **Ingest** — the uploaded `.txt` file is read into memory.
2. **Chunk** — `CharacterTextSplitter` splits the text into 1,000-character chunks. Chunking is required because embedding models have a fixed context window, and because retrieval is more precise over small passages than over a whole document.
3. **Embed** — each chunk is converted to a vector with `OpenAIEmbeddings`, so that semantic similarity becomes a distance calculation.
4. **Index** — the vectors are loaded into a **Chroma** vector store.
5. **Retrieve** — the store is exposed as a retriever, which returns the chunks most similar to the user's question.
6. **Generate** — a `RetrievalQA` chain (`chain_type="stuff"`) inserts the retrieved chunks into the prompt and asks the LLM to answer from them.

`chain_type="stuff"` means every retrieved chunk is stuffed into a single prompt. It is the simplest and cheapest strategy and it is the right default here, but it breaks down once the retrieved context exceeds the model's context window — see *Limitations*.

## Design notes

- **The API key is never stored.** It is entered in a password-masked field, held only for the duration of the request, and explicitly deleted from scope (`del openai_api_key`) once the call completes. Nothing is written to disk and nothing is committed.
- **The UI is progressively gated.** The question field stays disabled until a file is uploaded, and the key field and submit button stay disabled until both a file and a question exist. This makes invalid states unreachable instead of validating them after the fact.
- **The index is per-session and in-memory.** Each upload builds a fresh vector store, so there is no cross-user data leakage between sessions.

## Stack

Python · Streamlit · LangChain · OpenAI (LLM + embeddings) · Chroma vector store · tiktoken

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

You will need your own OpenAI API key. Paste it into the app at run time — there is no `.env` to populate and no key in this repo.

## Limitations, and what I would change

Being explicit about these, because knowing where a design stops working is the useful part:

- **`.txt` only.** Real documents arrive as PDF and DOCX. Adding a loader per file type is the first change I would make.
- **No index persistence.** The vector store is rebuilt on every upload, so identical documents are re-embedded and re-paid for. Persisting Chroma to disk with a content hash as the key would remove that cost.
- **`stuff` does not scale.** On a long document with many relevant passages, the assembled prompt will exceed the context window and the call fails. `map_reduce` or `refine` trades latency for the ability to handle large inputs.
- **No source attribution.** The app returns an answer but not the passages it came from. Returning the retrieved chunks alongside the answer is what makes RAG auditable, and it is cheap to add via `return_source_documents=True`.
- **Fixed-size chunking splits mid-sentence.** A recursive or semantic splitter respects paragraph boundaries and measurably improves retrieval quality.
- **Hard dependency on a paid API.** Every question costs money and sends user documents to a third party. That constraint is exactly what motivated the local-model rebuild in `agentic-ai-hitl`.

## Provenance

Built as a hands-on implementation of the canonical LangChain + Streamlit RAG pattern while studying applied AI. The value of the exercise for me was in the retrieval mechanics — chunking, embedding, and the retriever/chain boundary — which carry over directly to the local, keyless version.
