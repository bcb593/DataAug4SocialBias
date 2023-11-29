import os
os.environ['TRANSFORMERS_CACHE']='/scratch0/bashyalb/pretrained_models/'
from transformers import pipeline
from transformers import AutoTokenizer
import transformers
import torch
import re

model = "meta-llama/Llama-2-7b-chat-hf" # meta-llama/Llama-2-70b-hf
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
prompt_path='/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/prompts/prompt_gender.txt'
sentences_path = '/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/longer_text10k_gender.txt'


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
    for sentence in sentences[:5]:
        if i%100==0:
          print('Done with',i,'sentences')
        i+=1
        full_prompt = f"{prompt}\"{sentence}\"Response:For the same social group:"
        aug_sent = get_llama_response(full_prompt)
        relevant_sentences.append(aug_sent)

    return relevant_sentences

relevant_sentences = return_relevant_sentences(sentences_path, prompt_path)

## Extract the sentences from the generation: 
def extract_groups(relevant_sentences):
    pattern = r"(?i)for (the )?same social group:\s*(.*?)(?:\n|$)+for (the )?opposite social group:\s*(.*?)(?:\n|$)"

    same_social_group_sentences = []
    opposite_social_group_sentences = []

    for element in relevant_sentences:
        match = re.search(pattern, element, re.DOTALL)
        if match:
            same_social_group_sentences.append(match.group(2).strip())
            opposite_social_group_sentences.append(match.group(4).strip())
    
    return same_social_group_sentences, opposite_social_group_sentences

same_group,opposite_group=extract_groups(relevant_sentences)
with open('/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/SocialGroups/gender/Generated/Same_Social_Group.txt', 'w',encoding='utf-8') as f:
    for sentence in same_group:
        f.write(sentence + '\n')

with open('/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/SocialGroups/gender/Generated/Opposite_Social_Group.txt', 'w',encoding='utf-8') as f:
    for sentence in opposite_group:
        f.write(sentence + '\n')

#And write everything in a file
with open('/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/SocialGroups/gender/Generated/all_generated.txt', 'w',encoding='utf-8') as f:
    for sentence in relevant_sentences:
        f.write(sentence + '\n')
