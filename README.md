# Low-Rank-Adaptation_Collab_Notebooks | AI & NLP Experiments

This repository contains multiple Jupyter/Colab notebooks exploring **Low-Rank Adaptation (LoRA)** techniques and **Retrieval-Augmented Generation (RAG)** systems.  
Each notebook is independent and focuses on a specific concept or implementation.

---

## 📂 Notebooks

### 1. Low-Rank Adaptation – NLP Classifier
**File:** `Low-Rank_Adaptation_NLP_Classifier.ipynb`  
- Implements LoRA fine-tuning for a text classification task.  
- Demonstrates efficiency gains by training only low-rank matrices instead of full model weights.  
- Example dataset: sentiment analysis / text categorization.

---

### 2. Low-Rank Adaptation – Causal LM (Coding)
**File:** `Low-Rank_Adaptation_CausalLM_Coding.ipynb`  
- Applies LoRA to a **Causal Language Model (LM)** for code generation.  
- Fine-tunes a pretrained coding model (e.g., CodeLLaMA, GPT-NeoX) on small code datasets.  
- Focus: improving task-specific performance with minimal GPU memory.

---

### 3. RAG – Retrieval with Basic Sentences
**File:** `RAG_Retrieval_Basic_Sentences.ipynb`  
- Simple RAG pipeline using FAISS or similar vector store.  
- Embeds short text sentences and retrieves relevant information.  
- Serves as an introduction to retrieval-based augmentation.

---

### 4. RAG – Collaborative LLM Q&A
**File:** `RAG_Collab_LLMQA.ipynb`  
- Extends RAG for question answering with **collaborative LLM workflows**.  
- Demonstrates indexing, retrieval, and LLM-based answer generation.  
- Can be adapted for multi-user Q&A or knowledge base querying.

---

## 🚀 Usage
1. Open any notebook in **Google Colab** or Jupyter.  
2. Install required dependencies (`transformers`, `datasets`, `faiss`, etc.).  
3. Run cells step by step.  
4. Modify datasets or hyperparameters as needed.

---

## 📌 Requirements
- Python 3.9+  
- Hugging Face `transformers`  
- FAISS / ChromaDB (for RAG notebooks)  
- PyTorch (GPU recommended for LoRA fine-tuning)  

---

## 📝 License
This repository is provided for **educational purposes**.  
Feel free to adapt and extend the code for your own experiments.
