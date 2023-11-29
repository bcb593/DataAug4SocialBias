from transformers import pipeline
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf" # meta-llama/Llama-2-70b-hf
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
prompt_path='/scratch0/bashyalb/ACL2024/SentenceGeneration/Data/chain_of_thoughts.txt'
sentences_path = '/scratch0/bashyalb/ACL2024/SentenceGeneration/Data/longer_text10k.txt'

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
def get_llama_response(prompt: str) -> str:
    response = llama_pipeline(prompt, max_length=256)[0]['generated_text']

    return response
def return_relevant_sentences(sentences_path, prompt_path):
    prompt = cot(prompt_path)

    relevant_sentences = []
    with open(sentences_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file]

    for sentence in sentences[:3]:
        #print('Processing..')
        aug_sent = get_llama_response(prompt)
        relevant_sentences.append(aug_sent)
        print('Done with one sentence.')

    return relevant_sentences
relevant_sentences = return_relevant_sentences(sentences_path, prompt_path)
print(relevant_sentences)
