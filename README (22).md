# WorkingWithLLMs

A collection of hands-on Jupyter notebooks for running and experimenting with **Large Language Models (LLMs) locally** using [Ollama](https://ollama.com/), [LangChain](https://python.langchain.com/), and multimodal vision models — all without any cloud API dependency.

---

## Overview

This repository demonstrates three core LLM workflows:

| Notebook | Model(s) Used | Key Capability |
|---|---|---|
| `ExtractTextFromImage_Llava.ipynb` | LLaVA (via Ollama) | Multimodal OCR — extract and interpret text from images |
| `OllamaLangChain_Llama3.ipynb` | Llama 3 (via Ollama + LangChain) | Prompt chaining and LangChain integration |
| `OllamaLlama3_StreamOutput.ipynb` | Llama 3 (via Ollama) | Streaming token-by-token LLM output |

All notebooks run **fully locally** — no OpenAI key, no cloud costs.

---

## Repository Structure

```
WorkingWithLLMs/
├── ExtractTextFromImage_Llava.ipynb   # Vision model OCR with LLaVA
├── OllamaLangChain_Llama3.ipynb       # LangChain + Llama 3 integration
├── OllamaLlama3_StreamOutput.ipynb    # Streaming output with Ollama
├── TestOllama                          # Basic Ollama connectivity test script
├── img/                                # Sample images used in notebooks
└── README.md
```

---

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com).

Pull the required models:

```bash
ollama pull llama3.2
ollama pull llava
```

Verify Ollama is running:

```bash
ollama list
```

### 2. Python Environment

Recommended: Python 3.10+ with a virtual environment or conda.

Install dependencies:

```bash
pip install ollama langchain langchain-community jupyter
```

### 3. Running in Docker (optional, e.g. corporate/restricted environments)

If Ollama is running inside Docker, the `OLLAMA_URL` in the notebooks may need to point to the container host rather than `localhost`:

```python
# Use this if your script runs outside the container
OLLAMA_URL = "http://host.docker.internal:11434"

# Or find the container IP with:
# docker inspect <container_name> | grep IPAddress
```

---

## Notebooks

### `ExtractTextFromImage_Llava.ipynb`

**Observation:** Images often contain text (receipts, forms, screenshots, printed documents) that needs to be programmatically extracted.

**Hypothesis:** A locally-run vision-language model (LLaVA) can perform OCR-style text extraction without sending data to any external API.

**Experiment:** Load an image from disk, encode it to base64, and send it to LLaVA via the Ollama Python client with a structured prompt asking for text extraction.

**What you'll learn:**
- How to pass images to multimodal LLMs via Ollama
- How to structure vision prompts for OCR tasks
- How to handle base64 image encoding in Python

**Key libraries:** `ollama`, `base64`, `PIL` (optional)

---

### `OllamaLangChain_Llama3.ipynb`

**Observation:** Raw Ollama calls are straightforward but lack composability for chains, templates, and memory.

**Hypothesis:** LangChain's abstraction layer over Ollama enables more structured, reusable prompt workflows.

**Experiment:** Wire up a `ChatOllama` LangChain component with Llama 3, define prompt templates, and run a basic chain.

**What you'll learn:**
- How to use `langchain-community`'s `ChatOllama` wrapper
- How to build prompt templates and chains
- How LangChain abstracts model interaction for composability

**Key libraries:** `langchain`, `langchain-community`

---

### `OllamaLlama3_StreamOutput.ipynb`

**Observation:** Waiting for a full LLM response before displaying output creates a poor user experience for long generations.

**Hypothesis:** Ollama's streaming API allows token-by-token output, enabling real-time display in a notebook.

**Experiment:** Use Ollama's `stream=True` parameter and iterate over the response to print tokens as they arrive.

**What you'll learn:**
- How to enable and handle streaming responses from Ollama
- How to build a simple CLI-style streaming display in a Jupyter cell
- The difference between blocking and streaming LLM calls

**Key libraries:** `ollama`

---

### `TestOllama`

A minimal script to verify that the Ollama server is reachable and responding. Run this first to confirm your local setup is working before executing the notebooks.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/alketcecaj12/WorkingWithLLMs.git
cd WorkingWithLLMs

# 2. Ensure Ollama is running
ollama serve  # (skip if already running as a service)

# 3. Pull required models
ollama pull llama3.2
ollama pull llava

# 4. Launch Jupyter
jupyter notebook

# 5. Open any notebook and run all cells
```

---

## Use Cases

This repository is useful for:

- **Data Scientists and ML Engineers** exploring local LLM tooling without cloud API costs
- **Enterprise / restricted-network environments** where data cannot leave the machine
- **Developers building AI pipelines** with LangChain who want a local model backend
- **Students and practitioners** learning the Ollama + LangChain ecosystem

---

## Tech Stack

| Component | Tool |
|---|---|
| Local LLM runtime | [Ollama](https://ollama.com/) |
| Language model | [Meta Llama 3.2](https://ollama.com/library/llama3.2) |
| Vision-language model | [LLaVA](https://ollama.com/library/llava) |
| LLM framework | [LangChain](https://python.langchain.com/) |
| Notebook environment | [Jupyter](https://jupyter.org/) |
| Optional containerisation | [Docker](https://www.docker.com/) |

---

## Author

**Alket Cecaj** — Data Scientist & Quantitative Risk Analyst  
[GitHub](https://github.com/alketcecaj12)

---

## License

This project is open source. Feel free to fork, adapt, and build on top of it.
