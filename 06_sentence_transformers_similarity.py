# model  : sentence-transformers/all-MiniLM-L6-v2
# task   : FAQ search engine using sentence similarity

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FAQ = {
    "How do I reset my password?": "Go to Settings → Security → Reset Password. You'll receive an email with a reset link.",
    "What payment methods do you accept?": "We accept Visa, Mastercard, PayPal, and Apple Pay.",
    "How long does shipping take?": "Standard shipping takes 5–7 business days. Express shipping takes 1–2 business days.",
    "Can I return a product after 30 days?": "Returns are accepted within 30 days. After that, only exchanges are allowed.",
    "How do I contact customer support?": "You can reach us 24/7 via live chat or email at support@example.com.",
    "Is my personal data safe?": "We use AES-256 encryption and never sell your data to third parties.",
    "How do I cancel my subscription?": "Go to Account → Subscription → Cancel Plan. Access continues until end of billing period.",
    "Do you offer a free trial?": "Yes — 14 days, no credit card required.",
}

USER_QUERIES = [
    "I forgot my password, how can I get back in?",
    "Do you take credit cards?",
    "When will my order arrive?",
    "I want to stop my monthly plan.",
]


def mean_pooling(model_output, attention_mask):
    tokens = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
    return (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode(texts, tokenizer, model):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**encoded)
    return F.normalize(mean_pooling(out, encoded["attention_mask"]), p=2, dim=1)


def main():
    print("loading:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    faq_questions = list(FAQ.keys())
    faq_answers = list(FAQ.values())
    faq_embeddings = encode(faq_questions, tokenizer, model)

    print()
    for query in USER_QUERIES:
        scores = (encode([query], tokenizer, model) @ faq_embeddings.T).squeeze(0)
        idx = scores.argmax().item()
        print(f"query  : {query}")
        print(f"matched: {faq_questions[idx]}  ({scores[idx]:.4f})")
        print(f"answer : {faq_answers[idx]}\n")


if __name__ == "__main__":
    main()
