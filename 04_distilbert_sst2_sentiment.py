# model  : distilbert/distilbert-base-uncased-finetuned-sst-2-english
# task   : sentiment analysis on product reviews

from transformers import pipeline

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

REVIEWS = [
    "This laptop is absolutely incredible — the battery lasts all day and the display is stunning!",
    "I've never tasted a better pizza in my life. Perfectly crispy crust and fresh ingredients.",
    "The customer support team was so helpful and resolved my issue within minutes. Highly recommended!",
    "Best sci-fi movie of the decade. The special effects blew my mind and the plot was gripping.",
    "Total waste of money. The product broke after two days and customer service never replied.",
    "The hotel room smelled awful, the Wi-Fi didn't work, and the breakfast was inedible.",
    "I sat through two hours of this film waiting for something interesting to happen. Deeply disappointing.",
    "The app crashes every time I try to open it. Absolutely unusable.",
    "It's okay I guess. Does what it says, nothing more, nothing less.",
    "Not bad, but I expected more for this price point.",
]

COLORS = {"POSITIVE": "\033[92m", "NEGATIVE": "\033[91m"}
RESET = "\033[0m"


def main():
    print("loading:", MODEL_NAME)
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME)
    results = classifier(REVIEWS)

    positive = 0
    for review, r in zip(REVIEWS, results):
        c = COLORS.get(r["label"], "")
        print(f"{c}{r['label']:<10}{RESET}  {r['score']:.2%}  |  {review[:65]}{'...' if len(review) > 65 else ''}")
        if r["label"] == "POSITIVE":
            positive += 1

    print(f"\n{positive} positive / {len(REVIEWS) - positive} negative out of {len(REVIEWS)} reviews")


if __name__ == "__main__":
    main()
