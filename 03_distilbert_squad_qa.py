# model  : distilbert/distilbert-base-cased-distilled-squad
# task   : extractive question answering

from transformers import pipeline

MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"

CONTEXT = """
The Internet is a global system of interconnected computer networks that use the
Internet protocol suite (TCP/IP) to communicate between networks and devices.

The World Wide Web (WWW) is an information system where documents are identified
by URLs and accessible over the Internet. The Web was invented by British scientist
Tim Berners-Lee in 1989 while working at CERN. The first website was published on
August 6, 1991.

Data transmitted over the Internet is broken into small packets that travel
independently and are reassembled at the destination. This packet-switching
architecture means that if one path is blocked, packets can take an alternate route.

The Domain Name System (DNS) acts as the Internet's phonebook, translating
human-friendly domain names like www.example.com into numerical IP addresses.
"""

QUESTIONS = [
    "Who invented the World Wide Web?",
    "When was the first website published?",
    "What does DNS stand for and what does it do?",
    "How does packet-switching make the Internet resilient?",
]


def main():
    print("loading:", MODEL_NAME)
    qa = pipeline("question-answering", model=MODEL_NAME)

    for question in QUESTIONS:
        result = qa(question=question, context=CONTEXT)
        print(f"Q: {question}")
        print(f"A: {result['answer']}  ({result['score']:.2%})\n")


if __name__ == "__main__":
    main()
