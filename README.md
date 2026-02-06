# ObsidianRAG

A Retrieval-Augmented Generation (RAG) system for querying your personal knowledge base using local AI models.

## Overview

ObsidianRAG transforms your Obsidian notes into an intelligent, searchable knowledge base. It uses vector embeddings and AI-powered reranking to retrieve the most relevant information and generate accurate answers to your questions.

## Features

- ✨ **Local Embeddings**: Uses Sentence Transformers (`all-mpnet-base-v2`) for privacy-focused, cost-free embeddings
- 🔍 **Semantic Search**: Vector-based similarity search using ChromaDB
- 🎯 **AI-Powered Reranking**: LLM-based reranking for improved relevance
- 💬 **Conversational Interface**: Query your notes naturally with conversation history
- 📊 **Visualization**: 2D t-SNE visualization of your knowledge base
- 🏷️ **Multi-Category Support**: Organizes notes by topic (CMM, RNN, RAG, etc.)

## Architecture

```
┌─────────────────┐
│ Obsidian Notes  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Extraction │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sentence-       │
│ Transformers    │
│ Embeddings      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ChromaDB        │
│ Vector Store    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Engine    │
│ + Reranking     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Response    │
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install sentence-transformers chromadb litellm plotly scikit-learn numpy
```

Optional (for Jupyter notebooks):
```bash
pip install notebook nbformat
```

## Configuration

1. **Set up your notes directory**: Update the path to your Obsidian vault or notes folder
2. **Configure the LLM model**: Set your preferred model in the `MODEL` variable
3. **Adjust retrieval parameters**: Modify `RETRIEVAL_K` for the number of chunks to retrieve

## Usage

### 1. Create Embeddings from Your Notes

```python
from sentence_transformers import SentenceTransformer

# Load your notes/chunks
chunks = load_your_notes()

# Create embeddings
create_embeddings(chunks)
```

### 2. Query Your Knowledge Base

```python
# Simple query
question = "What is attention mechanism in transformers?"
chunks = fetch_context_unranked(question)

# With reranking
ranked_chunks = rerank(question, chunks)

# Generate answer
answer = generate_answer(question, ranked_chunks)
```

### 3. Visualize Your Vector Store

```python
# Create 2D visualization
visualize_vectors(vectors, doc_types, documents)
```

## Project Structure

```
.
├── README.md
├── embeddings/
│   └── create_embeddings.py    # Embedding generation
├── retrieval/
│   ├── fetch_context.py        # Retrieval functions
│   └── rerank.py               # Reranking logic
├── generation/
│   └── generate_answer.py      # LLM response generation
├── utils/
│   └── visualize.py            # Visualization tools
└── data/
    └── chroma_db/              # ChromaDB storage
```

## Key Components

### Embedding Model
- **Model**: `all-mpnet-base-v2`
- **Dimensions**: 768
- **Type**: Sentence Transformers (local, no API required)

### Vector Store
- **Database**: ChromaDB
- **Storage**: Persistent local storage
- **Metadata**: Stores chunk_id, headline, summary, tags, source, type

### LLM Integration
- **Framework**: LiteLLM
- **Supported Providers**: Ollama, Hugging Face, OpenAI, Anthropic
- **Default**: Configurable via `MODEL` variable

## Supported Note Categories

The system currently supports organizing notes into categories:
- **CMM**: Computational Models
- **RNN**: Recurrent Neural Networks
- **RAG**: Retrieval-Augmented Generation
- **LLM Engineering**: General LLM topics

Add more categories by updating the color mapping in the visualization code.

## Performance Tips

1. **Batch Processing**: Process large note collections in batches
2. **Model Caching**: Load the embedding model once and reuse
3. **Retrieval Tuning**: Adjust `RETRIEVAL_K` based on your use case (default: 10)
4. **Reranking**: Use reranking for better accuracy (adds latency but improves results)

## Troubleshooting

### ChromaDB Metadata Errors
If you get metadata validation errors, ensure lists are converted to JSON strings:
```python
meta[k] = json.dumps(v) if isinstance(v, list) else v
```

### Plotly Rendering Issues
Install nbformat:
```bash
pip install nbformat
```

### LLM Provider Errors
Ensure your MODEL variable uses the correct format:
- Ollama: `ollama/model-name`
- Hugging Face: `huggingface/model-name`
- OpenAI: `gpt-4o-mini`

## Roadmap

- [ ] Add support for more document formats (PDF, DOCX)
- [ ] Implement incremental updates (add new notes without rebuilding)
- [ ] Add conversation memory persistence
- [ ] Web UI for easier interaction
- [ ] Multi-language support
- [ ] Query analytics and insights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this project for personal or commercial purposes.

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for local embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [LiteLLM](https://docs.litellm.ai/) for unified LLM interface
- [Obsidian](https://obsidian.md/) for note-taking inspiration

---

**Note**: This project is designed for personal knowledge management. Ensure you have the rights to process any documents you add to the system.
