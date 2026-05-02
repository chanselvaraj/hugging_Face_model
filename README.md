# Hugging Face Model Demos

A hands-on YouTube series exploring 9 real Hugging Face models — from text embeddings and sentiment analysis to image generation and pose-guided art. Every script is self-contained and ready to run after a one-time setup.

---

## Getting Started

### 1. Open the project in your IDE

Clone or download this repo, then open the folder in VS Code or PyCharm.

```bash
git clone https://github.com/your-username/hugging_Face_model.git
cd hugging_Face_model
```

---

### 2. Create a virtual environment

**VS Code / terminal**
```bash
python -m venv venv
```

**Activate it**

| Platform | Command |
|----------|---------|
| Windows (cmd) | `venv\Scripts\activate.bat` |
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |
| macOS / Linux | `source venv/bin/activate` |

You should see `(venv)` appear at the start of your terminal prompt.

---

### 3. Install PyTorch

PyTorch must be installed before everything else because the right version depends on your hardware.

**GPU (recommended for image generation)**
Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), pick your CUDA version, and run the generated command. Example for CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (works for all NLP scripts, slow for image generation)**
```bash
pip install torch torchvision
```

---

### 4. Install all other dependencies

```bash
pip install -r requirements.txt
```

> Models are downloaded automatically from Hugging Face the first time you run each script. They are cached locally so subsequent runs are instant. Cache location: `~/.cache/huggingface/hub/`

---

### 5. Run the scripts

Each script is standalone — run them in any order.

```bash
python 01_bge_large_en_embeddings.py
python 02_deepseek_r1_reasoning.py
python 03_distilbert_squad_qa.py
python 04_distilbert_sst2_sentiment.py
python 05_gpt2_text_generation.py
python 06_sentence_transformers_similarity.py
python 07_stable_diffusion_v15_image_gen.py
python 08_sdxl_turbo_fast_image_gen.py
python 09_controlnet_openpose_image_gen.py
```

Generated images from scripts 07–09 are saved to the `outputs/` folder.

---

## The Models

### 01 — BAAI/bge-large-en · Semantic Search
**Script:** `01_bge_large_en_embeddings.py`
**HF link:** https://huggingface.co/BAAI/bge-large-en

BGE (BAAI General Embedding) converts text into dense numeric vectors called embeddings. Sentences with similar meaning end up close together in vector space — even if they use completely different words. This demo builds a 10-entry AI knowledge base, embeds a user question, and ranks the passages by cosine similarity to find the most relevant answer.

---

### 02 — DeepSeek-R1-Distill-Qwen-1.5B · Chain-of-Thought Reasoning
**Script:** `02_deepseek_r1_reasoning.py`
**HF link:** https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

DeepSeek-R1 is a reasoning model trained to "think out loud" before giving an answer, similar to OpenAI o1. This distilled 1.5B version is compact enough to run on a laptop GPU. The demo gives it a multi-step train journey problem and shows the full step-by-step reasoning trace the model produces before reaching the final answer.

---

### 03 — distilbert-base-cased-distilled-squad · Extractive Q&A
**Script:** `03_distilbert_squad_qa.py`
**HF link:** https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad

DistilBERT is a smaller, faster version of BERT — 40% fewer parameters, 60% faster, 97% of BERT's performance. Fine-tuned on SQuAD, it reads a paragraph and extracts the exact span of text that answers a question. The demo feeds it a passage about how the Internet works and asks four natural questions — confidence scores are shown alongside each answer.

---

### 04 — distilbert-base-uncased-finetuned-sst-2-english · Sentiment Analysis
**Script:** `04_distilbert_sst2_sentiment.py`
**HF link:** https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english

The same DistilBERT architecture fine-tuned on SST-2 (Stanford Sentiment Treebank) to classify text as POSITIVE or NEGATIVE. This demo runs 10 product reviews through the classifier and prints a colour-coded report — green for positive, red for negative — with a confidence score for each prediction.

---

### 05 — GPT-2 · Text Generation
**Script:** `05_gpt2_text_generation.py`
**HF link:** https://huggingface.co/openai-community/gpt2

GPT-2 was OpenAI's 2019 language model, famous for being "too dangerous to release" at the time. Today it is fully open. It predicts the next token in a sequence, one token at a time, to generate flowing text. The demo starts with a sci-fi story hook about humanity's first alien contact and generates 3 different creative continuations — showing how temperature and sampling affect the output.

---

### 06 — all-MiniLM-L6-v2 · Sentence Similarity
**Script:** `06_sentence_transformers_similarity.py`
**HF link:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

A lightweight sentence embedding model optimised for speed. It maps short texts to 384-dimensional vectors in milliseconds, making it ideal for real-time search. The demo builds a small FAQ knowledge base, embeds every question, then matches four new user queries to the closest FAQ entry — the foundation of every chatbot, support bot, and semantic search product.

---

### 07 — stable-diffusion-v1-5 · Text-to-Image
**Script:** `07_stable_diffusion_v15_image_gen.py`
**HF link:** https://huggingface.co/runwayml/stable-diffusion-v1-5

Stable Diffusion v1.5 is the workhorse of open-source image generation. It starts from pure noise and iteratively denoises it over ~30 steps, guided by your text prompt, until a full image emerges. The demo generates a cinematic cyberpunk city scene and shows the effect of a negative prompt — a list of things you don't want in the image — to keep results clean and sharp.

> Requires ~4 GB VRAM on GPU. CPU generation takes ~10–20 minutes per image.

---

### 08 — sdxl-turbo · Ultra-Fast Image Generation
**Script:** `08_sdxl_turbo_fast_image_gen.py`
**HF link:** https://huggingface.co/stabilityai/sdxl-turbo

SDXL-Turbo uses Adversarial Diffusion Distillation (ADD) to produce high-quality images in just 1–4 steps instead of the usual 30–50. The demo generates the same dragon prompt at 1, 2, and 4 steps and records the time for each — giving you a live speed-vs-quality comparison on screen. Note: guidance scale must be set to 0 because it was trained without classifier-free guidance.

> Requires ~8 GB VRAM for best results.

---

### 09 — control_v11p_sd15_openpose · Pose-Guided Image Generation
**Script:** `09_controlnet_openpose_image_gen.py`
**HF link:** https://huggingface.co/lllyasviel/control_v11p_sd15_openpose

ControlNet adds spatial conditioning on top of Stable Diffusion — you can control the layout, edges, depth, or pose of the generated image. This demo uses the OpenPose variant: it downloads a reference photo, runs an OpenPose detector to extract a stick-figure skeleton, then generates a brand-new fantasy warrior character locked to that exact pose. Two models work together here — `lllyasviel/Annotators` for pose detection and `control_v11p_sd15_openpose` as the ControlNet weights.

> Requires ~6 GB VRAM. Both the pose skeleton and the final image are saved to `outputs/`.

---

## Hardware Guide

| Scripts | CPU | GPU (4 GB) | GPU (8 GB+) |
|---------|-----|------------|-------------|
| 01–06 (NLP) | fast | fast | fast |
| 07 stable-diffusion-v1-5 | slow (~15 min) | good | great |
| 08 sdxl-turbo | very slow | limited | great |
| 09 controlnet | slow (~20 min) | good | great |
