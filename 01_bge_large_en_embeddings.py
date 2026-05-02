# model  : BAAI/bge-large-en
# task   : semantic search over a knowledge base using dense embeddings

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

MODEL_NAME = "BAAI/bge-large-en"

KNOWLEDGE_BASE = [
    "Transformers are deep learning models that use self-attention mechanisms to process sequential data.",
    "Gradient descent is an optimization algorithm used to minimize the loss function during model training.",
    "Overfitting occurs when a model learns the training data too well and performs poorly on unseen data.",
    "Transfer learning allows a pre-trained model to be fine-tuned on a new, often smaller, dataset.",
    "Convolutional Neural Networks (CNNs) are primarily used for image recognition and computer vision tasks.",
    "Reinforcement learning trains agents by rewarding desired behaviors and punishing undesired ones.",
    "BERT is a bidirectional transformer pre-trained on masked language modeling and next sentence prediction.",
    "Embeddings are dense vector representations that capture the semantic meaning of text.",
    "Hugging Face is an open-source platform that hosts thousands of pre-trained machine learning models.",
    "Large Language Models like GPT-4 are trained on massive text corpora and can generate human-like text.",
]

QUERY = "What is Hugging Face and why is it important for AI developers?"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def get_embeddings(texts, tokenizer, model):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    return F.normalize(mean_pooling(output, encoded["attention_mask"]), p=2, dim=1)


def main():
    print("loading:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    kb_embeddings = get_embeddings(KNOWLEDGE_BASE, tokenizer, model)
    query_embedding = get_embeddings([QUERY], tokenizer, model)
    scores = (query_embedding @ kb_embeddings.T).squeeze(0)
    ranked = scores.argsort(descending=True)

    print(f"\nquery: {QUERY}\n")
    for rank, idx in enumerate(ranked[:3].tolist(), start=1):
        print(f"  #{rank}  {scores[idx]:.4f}  {KNOWLEDGE_BASE[idx]}")


if __name__ == "__main__":
    main()
