from transformers import pipeline, set_seed

MODEL_NAME = "openai-community/gpt2"
INPUT = "The movie was full of"
MAX_NEW_TOKENS = 64

def main():
    set_seed(42)
    generator = pipeline("text-generation", model=MODEL_NAME)

    temps = [0.2, 0.4, 0.6, 0.8]

    for temp in temps:
        outputs = generator(INPUT, max_new_tokens=MAX_NEW_TOKENS, temperature=temp)
        print(f"temp {temp}: {outputs[0]['generated_text']}")

if __name__=="__main__":
    main()