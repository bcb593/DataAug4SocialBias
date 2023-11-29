import os
os.environ['TRANSFORMERS_CACHE']='/scratch0/bashyalb/pretrained_models/'
from transformers import pipeline
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf" # meta-llama/Llama-2-70b-hf
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
prompt_path='/scratch0/bashyalb/ACL2024/SentenceGeneration/Data/prompt_gender.txt'
sentences_path = '/scratch0/bashyalb/ACL2024/SentenceGeneration/Data/gender_output_100.txt'


llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
def cot(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    return prompt
prompt = cot(prompt_path)
print(prompt)
def get_llama_response(prompt: str) -> str:
    full_text = llama_pipeline(prompt, max_length=1024)[0]['generated_text']
    response_keyword = "Response:"
    response_start = full_text.find(response_keyword)

    if response_start != -1:
        # Extract everything after "Response:"
        response_text = full_text[response_start + len(response_keyword):].strip()
        return response_text
    else:
        # If "Response:" is not found, return an empty string or a relevant message
        return "No response found in the text."

def return_relevant_sentences(sentences_path, prompt_path):
    prompt = cot(prompt_path)

    relevant_sentences = []
    with open(sentences_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file]
        i=0
    for sentence in sentences:
        if i%5==0:
          print('Done with',i,'sentences')
        i+=1
        full_prompt = f"{prompt}\"{sentence}\"Response:For the same social group:"
        aug_sent = get_llama_response(full_prompt)
        relevant_sentences.append(aug_sent)

    return relevant_sentences

relevant_sentences = return_relevant_sentences(sentences_path, prompt_path)