import os
os.environ['TRANSFORMERS_CACHE']='/scratch0/bashyalb/pretrained_models/'
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import pipeline, AutoTokenizer, logging
import transformers
import torch
import re
import argparse
from tqdm import tqdm

# Disable unnecessary transformers logging
logging.set_verbosity_error()

def main(args):
    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
    
    llama_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    def cot(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt

    def get_llama_response(prompt: str) -> str:
        full_text = llama_pipeline(prompt, max_length=900)[0]['generated_text']
        return full_text

    def return_relevant_sentences(sentences_path, prompt_path):
        prompt = cot(prompt_path)
        relevant_sentences = []

        with open(sentences_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file]

        for i, sentence in enumerate(tqdm(sentences[:1], desc="Processing sentences")):
            print(sentence)
            full_prompt = f"{prompt}#Original Sentence#:{sentence}\"Output:"
            aug_sent = get_llama_response(full_prompt)
            print(aug_sent)
            relevant_sentences.append(aug_sent)

        return relevant_sentences

    relevant_sentences = return_relevant_sentences(args.sentences_path, args.prompt_path)

    def extract_groups(relevant_sentences):
        pattern = r"For the same social group:(.*?)(?:\n|$)For the opposite social group:(.*?)(?:\n|$)"
        same_social_group_sentences = []
        opposite_social_group_sentences = []

        for element in relevant_sentences:
            matches = re.finditer(pattern, element, re.DOTALL)
            for match in matches:
                same_social_group_sentences.append(match.group(1).strip())
                opposite_social_group_sentences.append(match.group(2).strip())

        return same_social_group_sentences, opposite_social_group_sentences

    same_group, opposite_group = extract_groups(relevant_sentences)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir, 'w', encoding='utf-8') as f:
        for sentence in relevant_sentences:
            f.write(sentence + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for sentence generation and extraction")
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-chat-hf", help="The model name or path")
    parser.add_argument("--prompt_path", default="/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/prompts/samediversegen_unique.txt", help="Path to the prompt file")
    parser.add_argument("--sentences_path", required=True, help="Path to the sentences file")
    parser.add_argument("--output_dir", required=True, help="Output directory for saving results")
    args = parser.parse_args()
    main(args)
