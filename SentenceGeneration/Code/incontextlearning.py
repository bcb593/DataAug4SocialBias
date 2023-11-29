from transformers import AutoTokenizer
import transformers
import torch

def is_relevant(sentence, examples, model="text-davinci-003", api_key="your-api-key"):
    openai.api_key = api_key

    prompt = "\n\n".join(examples) + f"\n\nSentence: \"{sentence}\"\nLabel:"
    
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=1,
            n=1,
            stop=["\n"]
        )

        label = response.choices[0].text.strip().lower()
        return label in ["relevant", "yes"]
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
example_sentences = [
    "Sentence: \"This policy affects people of all genders equally.\"\nLabel: Relevant",
    "Sentence: \"The tree in the backyard is very old.\"\nLabel: Not Relevant",
]

corpus = [
    "This policy affects people of all genders equally.",
    "The tree in the backyard is very old.",
    # Add more sentences from your corpus
]

# Process each sentence in the corpus
relevant_sentences = [sentence for sentence in corpus if is_relevant(sentence, example_sentences)]

print("Relevant Sentences:")
print("\n".join(relevant_sentences))
