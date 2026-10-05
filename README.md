# LLM Projects — Experimentation

A collection of hands-on LLM experiments — **fine-tuning models, building RAG pipelines, and occasionally discovering why the model confidently said something completely wrong.**

These projects are small, practical experiments focused on understanding what actually happens inside LLM workflows rather than hiding everything behind a framework.

---
> **📌 A quick note about the notebooks**
>
> GitHub's notebook preview displays the code from all cells together, so the projects can look much larger than they actually are.
>
> The notebooks are structured as **separate cells with explanations, code, and outputs**. If you want to see the experiments as they were meant to be followed, open the `.ipynb` file in **Google Colab or Jupyter Notebook** rather than judging it by the GitHub preview.
>
> In other words: **it looks scarier on GitHub than it is. 😄**

---
## 🧪 Projects

### 01 — Customer Support Bot

**Can a tiny LLM learn to sound like a customer support assistant?**

I fine-tuned TinyLlama 1.1B using QLoRA and HuggingFace PEFT on a small customer-support dataset.

**Exploring:** supervised fine-tuning · LoRA · 4-bit quantization · training behaviour · response evaluation

**Stack:** TinyLlama · PEFT · QLoRA · SFTTrainer · BitsAndBytes

---
### 02 — HR Policy Bot

**What happens when the same fine-tuning approach meets a different domain?**

I fine-tuned TinyLlama on a small HR policy instruction-response dataset and changed the LoRA configuration compared with Project 1.

The interesting part wasn't just getting the model to produce HR-style answers — it was seeing where those answers stopped being reliable.

**Exploring:** domain-specific fine-tuning · LoRA configuration · training behaviour · hallucination and response evaluation

**Stack:** TinyLlama · PEFT · QLoRA · SFTTrainer · BitsAndBytes

---
### 03 — RAG Pipeline

**Before using a framework, what is actually happening underneath?**

This project builds a basic RAG pipeline from scratch — no LangChain doing the heavy lifting.

A document is split into chunks, converted into embeddings, stored in FAISS, retrieved for a question, and passed to TinyLlama as context.

And then comes the interesting part: **the model can still make things up even when retrieval did its job.**

**Exploring:** chunking · embeddings · vector search · retrieval · context grounding · generation

**Stack:** Sentence Transformers · FAISS · TinyLlama · Python

---
## 🔍 What These Experiments Cover

| Experiment   | What I'm exploring                           |
| ------------ | -------------------------------------------- |
| Fine-tuning  | How a model adapts to examples               |
| LoRA / QLoRA | Parameter-efficient fine-tuning              |
| Quantization | Running models with reduced precision        |
| RAG          | Retrieval + generation as separate steps     |
| Embeddings   | Representing text as vectors                 |
| FAISS        | Searching those vectors for relevant context |
| Evaluation   | Looking beyond “it ran successfully”         |

---
## ▶️ Running the Projects

The notebooks can be opened in Jupyter or Google Colab. Projects involving model fine-tuning require a compatible GPU runtime.

Open a notebook, run the cells, and see what happens.

**Preferably with enough GPU memory. And perhaps a little patience.**
