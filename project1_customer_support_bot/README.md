# Project 1 — Customer Support Bot

### Fine-tuning TinyLlama 1.1B with QLoRA

A small experiment to see what happens when a compact LLM is fine-tuned on a handful of customer-support examples.

---

## 🧪 The Experiment

I used **10 customer-support instruction-response pairs** covering:

* Order tracking and delivery issues
* Returns and refunds
* Password resets and account access
* Billing and payment problems
* Subscription cancellation

The goal was to understand the complete fine-tuning workflow — from preparing the dataset and configuring LoRA to training the model and testing its responses.

---

## ⚙️ What I Used

| Component    | Choice              |
| ------------ | ------------------- |
| Base model   | TinyLlama 1.1B Chat |
| Fine-tuning  | QLoRA               |
| LoRA rank    | 8                   |
| LoRA alpha   | 16                  |
| Quantization | 4-bit NF4           |
| Training     | SFTTrainer          |
| Hardware     | Google Colab T4 GPU |

The small dataset makes this a learning experiment rather than a serious customer-support system. That also makes the model's limitations fairly easy to spot.

---

## 🔍 Things I Tested

* Formatting instruction-response examples for training
* 4-bit model loading and quantization
* LoRA adapters instead of updating the full model
* Supervised fine-tuning with `SFTTrainer`
* Changing generation temperature during inference
* Comparing the model's responses after fine-tuning
* Saving and reloading the LoRA adapter

---

## 🧠 What I Observed

The fine-tuned model picked up the general **customer-support style** from the examples, but the small training set also limited what it could reliably learn.

This was useful because the experiment wasn't just about getting training to run. It showed the difference between:

**“the model learned the style”**
and
**“the model knows the facts.”**

A small fine-tuning dataset can influence how a model responds without giving it enough information to reliably answer every possible customer question.

---

## 📓 Notebook

`Shruti_LoRA_Finetuning_Notebook.ipynb`

The notebook contains the dataset preparation, model setup, LoRA configuration, training, and inference experiments.

---

## ▶️ Running It

The notebook can be opened in Google Colab with a compatible GPU runtime.

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Shruti_LoRA_Finetuning_Notebook.ipynb`
3. Select a GPU runtime
4. Run the cells from top to bottom

Training time will depend on the runtime available.

---

## 🛠️ Stack

```text
Python · PyTorch · HuggingFace Transformers
PEFT · TRL · BitsAndBytes
TinyLlama 1.1B · QLoRA · SFTTrainer
```
