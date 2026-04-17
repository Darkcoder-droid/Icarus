# Icarus: Voice AI Code Companion

![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)
![Vapi](https://img.shields.io/badge/Vapi-Voice%20AI-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

Icarus is an AI-powered repository context engine designed to bridge the gap between human conversation and complex codebases. By leveraging advanced vector search and real-time voice synthesis, Icarus allows developers to "talk" to their code. It automatically indexes public GitHub repositories, analyzes the project structure, and provides a semantic understanding of the source code to a voice-enabled AI assistant.

## The Icarus Experience

Interacting with Icarus is simple and intuitive:

1. **Repository Indexing**: Users provide a URL to any public GitHub repository.
2. **Analysis**: Icarus works in the background to download the source, map the folder architecture, and chunk the code into searchable context.
3. **Voice Interaction**: Once indexed, a floating voice widget becomes active. Users can click the button and start speaking naturally to Icarus.
4. **Context-Aware Dialog**: You can ask questions like:

   * *"Can you explain the main logic in the backend?"*
   * *"Where is the database connection handled?"*
   * *"Give me a high-level overview of the project structure."*
5. **Real-Time Retrieval**: Icarus retrieves the most relevant snippets from the repository in milliseconds, giving the voice assistant the "memory" it needs to provide accurate technical answers.

## Technical Details

Icarus is built using a modern AI-native stack:

### Frontend

* **Vanilla HTML5/CSS3**: A premium, dark-themed interface with high attention to aesthetics and typography.
* **Vapi Client SDK**: Integrated via the Vapi widget for high-performance, low-latency voice AI interaction.
* **Real-time Status Tracking**: Feedback loops for repo cloning and indexing progress.

### Backend

* **FastAPI**: A high-performance Python web framework for handling ingestion and context retrieval.
* **Qdrant Cloud**: A managed vector database used to store and query code embeddings with high efficiency.
* **Sentence Transformers**: Specifically the `all-MiniLM-L6-v2` model, used to generate 384-dimensional embeddings for code chunks.
* **Repo Ingestion Engine**: A custom pipeline that clones public repos, filters non-essential files (like `node_modules` or `.git`), and processes allowed extensions (`.py`, `.js`, `.ts`, etc.).
* **Master Architecture Mapping**: A specialized indexing step that captures the entire project tree for structural queries.

---

## Environment Setup

This project requires API keys for **Vapi** and **Qdrant**. These keys are stored securely in a `.env` file.

After cloning the repository, you **must create a `.env` file in the root directory** and add the following:

```
VAPI_API_KEY=your_vapi_api_key
QDRANT_API_KEY=your_qdrant_api_key
```

Without these environment variables, the application will not be able to connect to required services and will fail to function correctly.

---

## Agent Tools

The Icarus agent is equipped with two core tools:

* **`search_codebase`**
  A retrieval tool that queries the vector database (Qdrant) and architecture maps to find relevant code snippets, structures, or debugging context based on user queries and errors.

* **`function_tool`**
  A tool connected to an **ngrok server**, enabling the agent to call external or remote functions as part of its execution workflow.

---

## Requirements & Setup

* **Python 3.8+**
* **Qdrant Cloud Account** (URL and API Key)
* **Vapi AI Account** (Public Key and Assistant ID)
* **FastAPI Dependencies**: `fastapi`, `uvicorn`, `qdrant-client`, `sentence-transformers`, `requests`.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as per the license terms.
