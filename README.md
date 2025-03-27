# Stark PDF Assistant

Stark is an intelligent Telegram bot that leverages LLaMA models and hierarchical RAG (Retrieval-Augmented Generation) to answer questions about PDF documents with precise references.

## Features

- **PDF Document Management**: Upload, list, and manage PDF documents
- **Hierarchical RAG**: Advanced retrieval system that organizes document content into pages, sections, and chunks
- **Context-Aware Answers**: Responses include document name, page number, and section references
- **Admin Controls**: Special commands for administrators to manage the knowledge base
- **Fallback Mechanisms**: Robust error handling with multiple query processing methods

## Architecture

The system consists of two main components:

1. **LlamaBot (Stark.py)**: Core RAG engine that handles:
   - PDF document loading and hierarchical chunking
   - Vector embedding and retrieval using FAISS
   - Ollama integration for embedding and query processing
   - Fallback mechanisms (keyword search when embeddings fail)
   - Document hierarchy management
   
2. **Telegram Bot (main.py)**: User interface that provides:
   - Command handling for document management
   - Conversation flows for PDF uploads
   - User session management
   - Admin privilege controls

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) with LLaMA models
- Telegram Bot Token
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hung369/LLM-Engineering.git
   cd LLM-Engineering/src
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration:
   ```
   BOT_TOKEN=your_telegram_bot_token
   ```

4. Make sure Ollama is installed and running with the required models:
   ```bash
   ollama pull llama3.1:8b
   ```

5. Create the necessary directories:
   ```bash
   mkdir -p data vector_db metadata
   ```

## Usage

### Starting the Bot

```bash
python src/main.py
```

### Bot Commands

- `/start` - Start the bot
- `/help` - Show help message
- `/reset` - Reset your chat history
- `/documents` - List available PDF documents

### Admin Commands

- `/addpdf` - Upload a new PDF document - **coming soon**
- `/structure <doc_id>` - View the hierarchical structure of a document - **coming soon**

### Typical Workflow

1. Admin uploads PDF documents using the `/addpdf` command
2. Users ask questions about the documents
3. The bot retrieves relevant sections and answers with precise references
4. Users can reset their chat history with `/reset`

## How It Works

### Document Processing

1. PDF documents are loaded and split into hierarchical chunks:
   - Pages → Sections → Chunks
2. Each chunk is embedded using the LLaMA model via Ollama
3. Embeddings are stored in a FAISS vector database

### Query Processing

1. User's query is enhanced with historical context
2. System retrieves relevant document chunks using hierarchical search
3. Chunks are reranked based on relevance and hierarchy level
4. Contextually relevant sections are assembled into a prompt
5. LLaMA model generates a response with document references

### Fallback Mechanisms

If vector embeddings fail, the system falls back to:
1. Keyword-based search
2. Simple BM25 reranking
3. Regular chat without document context

## Customization

### System Prompt

You can customize the system prompt in `main.py`:

```python
SYSTEM_PROMPT = """You are Stark, an AI assistant specialized in answering questions about PDF documents.
You respond only in English with clear, concise replies and always cite the specific parts of documents 
when providing information.

The following context comes from PDF documents:

{context}

When answering, mention the document name, page number, and section when relevant."""
```

### Model Selection

You can change the LLaMA model in `Stark.py`:

```python
def __init__(
    self, 
    model_name: str = "llama3.1:8b", 
    embedding_model: str = "llama3.1:8b",
    # other parameters...
):
```

## Directory Structure

- `src/` - Source code
  - `Stark.py` - Core RAG engine
  - `main.py` - Telegram bot interface
- `data/` - PDF documents storage
- `vector_db/` - FAISS vector database
- `metadata/` - Document hierarchy information

## Troubleshooting

### Common Issues

- **Embedding Errors**: Check that Ollama is running and the specified model is available
- **PDF Processing Fails**: Ensure the PDF is not password-protected and is text-based (not scanned images)
- **Out of Memory**: Try using a smaller model or reducing chunk sizes in `Stark.py`

### Logs

Check `bot.log` for detailed information about any errors.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain) for document processing utilities
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Ollama](https://github.com/jmorganca/ollama) for local LLM inference
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) for Telegram integration
