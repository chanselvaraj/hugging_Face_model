# model  : deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# task   : chain-of-thought math reasoning

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

PROBLEM = (
    "A train travels from City A to City B, a distance of 360 kilometres. "
    "It departs at 8:00 AM and travels at 90 km/h for the first half of the "
    "journey, then speeds up to 120 km/h for the second half. "
    "At what time does the train arrive at City B? Show your full reasoning."
)


def build_prompt(problem):
    # <think> scaffold nudges the model into step-by-step reasoning mode
    return (
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def main():
    print("loading:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    inputs = tokenizer(build_prompt(PROBLEM), return_tensors="pt").to(model.device)
    print(f"\nproblem:\n{PROBLEM}\n")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
