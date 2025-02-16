# LLM-Engineering

## Table of Contents

1. [Overview](#overview)
2. [RAG Work Flow](#RAG-Work-Flow)
3. [Fine-tuning LLM](#Fine-tuning-LLM)

---

## Overview

This repository is dedicated to advanced LLM engineering with a focus on Retrieval-Augmented Generation (RAG) and fine-tuning techniques, in order to enhance language model performance by leveraging external knowledge sources and custom training strategies, enabling more robust, context-aware, and domain-adaptable applications.

---

## RAG Work Flow

**RAG Flow** is designed to address one of the primary challenges of standalone LLMs: factual inaccuracy and context limitations. By augmenting generation with relevant, retrieved information, RAG Flow can:

- **Improve Accuracy:** Incorporate up-to-date external data during inference.
- **Enhance Contextual Relevance:** Use retrieved documents to ground responses in a broader context.
- **Enable Domain Adaptability:** Easily switch or update the knowledge base for specific applications.

**RAG Flow** is composed of two main modules:

1. **Retrieval Module**
   - **Input:** User query.
   - **Process:** Searches an external knowledge base (e.g., a vector database using FAISS or Elasticsearch) for documents relevant to the query.
   - **Output:** A set of context documents or passages.

2. **Generation Module**
   - **Input:** Combines the original query with the retrieved context.
   - **Process:** Feeds the enriched input into a language model (e.g., an LLM like LLAMA or GPT) to generate a coherent, contextually informed response.
   - **Output:** The final generated answer.

To run the inference code of RAG, just open ```RAG_Reranking_LLAMA.ipynb``` in Google Colab Pro/Pro+ then run every cell followed by the instruction

---

## Fine-tuning LLM

Coming soon

---
