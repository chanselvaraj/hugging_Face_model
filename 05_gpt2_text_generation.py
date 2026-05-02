# model  : gpt2
# task   : open-ended creative text generation

from transformers import pipeline, set_seed

MODEL_NAME = "gpt2"

PROMPT = (
    "In the year 2157, humanity received its first confirmed signal from an "
    "alien civilisation. The message was only three words long, but those three "
    "words changed everything:"
)

NUM_SEQUENCES = 3
MAX_NEW_TOKENS = 150


def main():
    print("loading:", MODEL_NAME)
    generator = pipeline("text-generation", model=MODEL_NAME)
    set_seed(42)

    print(f"\nprompt:\n{PROMPT}\n")

    outputs = generator(
        PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=NUM_SEQUENCES,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    for i, output in enumerate(outputs, start=1):
        continuation = output["generated_text"][len(PROMPT):]
        print(f"--- continuation #{i} ---")
        print(PROMPT + "\033[93m" + continuation + "\033[0m\n")


if __name__ == "__main__":
    main()
