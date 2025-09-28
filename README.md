# AI & NLP Experiments — LoRA & RAG

This repository contains **Jupyter/Colab notebooks** exploring **Low-Rank Adaptation (LoRA)** for model fine-tuning and **Retrieval-Augmented Generation (RAG)** for question answering and similarity search. Each notebook is self-contained and demonstrates a distinct workflow, ranging from SST-2 classification and causal LM coding tasks to PDF-based Q&A with Gemini and basic sentence-level retrieval.

---

## 📂 Notebooks

### 1. Low-Rank Adaptation — NLP Classifier

**File:** `Low_Rank_Adaptation_NLP_Classifier.ipynb`

* **Focus:** Fine-tuning a text classifier on sentiment-style datasets (GLUE SST-2 train/validation parquet shards).
* **Goal:** Efficiently adapt a supervised classification model without updating all model weights.
* **Key Steps:** Data loading, preprocessing, LoRA-based fine-tuning, evaluation on validation splits.

---

### 2. Low-Rank Adaptation — Causal LM (Coding)

**File:** `Low_Rank_Adaptation_CasualLM_Coding.ipynb`

* **Focus:** Adapting a causal language model for code generation tasks using lightweight parameter updates.
* **Goal:** Improve code completion and generation performance while keeping the base model mostly frozen.
* **Key Steps:** Tokenizer/model setup, LoRA adapter training, code generation evaluation.

---

### 3. RAG — Retrieval with Basic Sentences

**File:** `Rag_Retrival_Basic_Sentences.ipynb`

* **Focus:** Embedding short text sentences and performing similarity search.
* **Goal:** Demonstrate a compact retrieval pipeline using **SentenceTransformers** for vectorization.
* **Key Steps:** Embedding creation, FAISS or vector store indexing, nearest-neighbor search for sentence retrieval.

---

### 4. RAG — Collaborative LLM Q&A (PDF)

**File:** `RAG_Collab_LLMQA.ipynb`

* **Focus:** PDF ingestion, chunking, embedding with **all-MiniLM-L6-v2**, FAISS-based retrieval, and answer generation using **Google Generative AI Gemini models**.
* **Goal:** Build an interactive “chat with your document” workflow.
* **Key Steps:**

  1. Configure `GOOGLE_API_KEY` via Colab secrets.
  2. Upload PDFs, extract text with PyMuPDF.
  3. Chunk content using `RecursiveCharacterTextSplitter`.
  4. Embed text chunks and create a FAISS index.
  5. Query top chunks and generate answers via Gemini.

---

## 🚀 Quickstart

1. Open any notebook in **Google Colab** or **Jupyter** with Python 3.
2. Install required dependencies in the first cells (if not already available).
3. Run cells top-to-bottom; each notebook is independent and can be executed separately.

---

## ⚙️ Requirements

* **Python 3** (as indicated in notebooks’ kernelspecs)
* **Libraries (for RAG & LoRA workflows):**

  * `sentence-transformers`
  * `faiss-cpu`
  * `pymupdf`
  * `langchain`
  * `google-generativeai`
  * `ipywidgets`
* **Datasets:** GLUE SST-2 (used in NLP classifier)

---

## 📝 Notes

* The **RAG PDF Q&A notebook** keeps credentials secure via Colab secrets.
* PDF ingestion relies on **PyMuPDF** for text extraction and **FAISS IndexFlatL2** for vector search.
* The **basic sentence retrieval notebook** demonstrates minimal pipelines for nearest-neighbor searches using SentenceTransformers.
* LoRA notebooks show **efficient model adaptation** without full weight updates, ideal for resource-constrained environments.

---

This README now clearly separates notebooks, their purposes, dependencies, and execution instructions while being visually easy to read.

---

