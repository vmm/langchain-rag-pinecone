# RAG Implementation with LangChain and Pinecone

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and Pinecone. It allows you to ingest text and PDF documents, and then query the knowledge base using natural language.

## Features

- Document ingestion (supports both .txt and .pdf files)
- Customizable text chunking parameters
- Interactive chat interface with model selection (OpenAI GPT or Anthropic Claude)
- Built with LangChain and Pinecone for efficient vector storage and retrieval
- Supports multiple language models:
  - OpenAI's GPT models
  - Anthropic's Claude models
- Flexible model switching at runtime

## Prerequisites

- Docker and VS Code with Dev Containers extension installed
- OpenAI API key (if using OpenAI models)
- Anthropic API key (if using Claude models)
- Pinecone account and API key

## Getting Started with Dev Container

1. Clone the repository
2. Open the project in VS Code
3. When prompted, click "Reopen in Container" or run the "Dev Containers: Reopen in Container" command from the command palette
4. The container will be built and started automatically, installing all required dependencies

## Environment Setup

1. Create a `.env` file in the project root with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
```

Note: You only need to provide the API keys for the models you plan to use. For example, if you only want to use Claude, you only need the ANTHROPIC_API_KEY.

## Project Structure

```
.
├── chat.py              # Interactive chat interface with model selection
├── ingestion.py         # Core document ingestion functionality
├── ingest.py           # Unified document ingestion script
├── documents/          # Directory for source documents
└── tests/             # Test directory
```

## Usage

### Ingesting Documents

The project provides a unified ingestion script that handles both text and PDF files:

```bash
python ingest.py
```

The script will interactively prompt you for:
1. File path (supports .txt and .pdf files)
2. Chunk size (defaults to 1000)
3. Chunk overlap (defaults to 0)

Example session:
```
=== Document Ingestion Tool ===
This tool will help you ingest text and PDF files into the vector database.
Supported file types: .txt, .pdf

Enter the path to your file: ./documents/example.txt
Enter chunk size (default 1000): 1500
Enter chunk overlap (default 0): 200

Starting ingestion process...
```

### Using the Chat Interface

Start the chat interface:
```bash
python chat.py
```

The chat interface will:
1. Prompt you to select a model (OpenAI GPT or Anthropic Claude)
2. Initialize the selected model for chat
3. Allow you to start asking questions

Commands:
- Type your questions and press Enter
- Type 'quit' or 'exit' to end the conversation
- Press Ctrl+C to exit

## Development

This project uses Poetry for dependency management. The dev container automatically sets up the Python environment with all required dependencies.

### Adding Dependencies

```bash
poetry add package_name
```

### Running Tests

```bash
poetry run pytest
```

## Code Quality

The project uses:
- Black for code formatting
- Pylint for code linting
- Type hints for better code clarity

## Error Handling

The application includes robust error handling for:
- Missing environment variables
- API connection issues
- Document processing errors
- File validation (existence and type checking)
- User input validation
- Model-specific error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
