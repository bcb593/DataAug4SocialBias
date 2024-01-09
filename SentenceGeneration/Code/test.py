import os
os.environ['TRANSFORMERS_CACHE']='/scratch0/bashyalb/pretrained_models/'
from transformers import pipeline
from transformers import AutoTokenizer
import transformers
import torch
import re

model = "meta-llama/Llama-2-13b-chat-hf" # meta-llama/Llama-2-70b-hf
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
#prompt_path='/home/zhangxit/files/DataAug4SocialBias/SentenceGeneration/Data/prompts/prompt_gender.txt'
#sentences_path = 'home/zhangxit/files/DataAug4SocialBias/SentenceGeneration/Data/longer_text10k_gender.txt'
prompt_path='/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/prompts/samediversegen_unique.txt'
sentences_path = '/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/DebiasingCorpus/Original/corpus_10-40_10kCDA.txt'

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
prompt = cot(prompt_path)
print(prompt)
def get_llama_response(prompt: str) -> str:
    full_text = llama_pipeline(prompt, max_length=900)[0]['generated_text']
    return full_text

from tqdm import tqdm

def return_relevant_sentences(sentences_path, prompt_path):
    prompt = cot(prompt_path)

    relevant_sentences = []
    with open(sentences_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file]

    for i, sentence in enumerate(tqdm(sentences, desc="Processing sentences")):
        print(sentence)
        full_prompt = f"{prompt}#Original Sentence#:{sentence}\"Output:"
        aug_sent = get_llama_response(full_prompt)
        print(aug_sent)
        relevant_sentences.append(aug_sent)

    return relevant_sentences


relevant_sentences = return_relevant_sentences(sentences_path, prompt_path)

## Extract the sentences from the generation: 
def extract_groups(relevant_sentences):
    # Adjusted pattern to handle complex sentences and multiline matches
    pattern = r"For the same social group:(.*?)(?:\n|$)For the opposite social group:(.*?)(?:\n|$)"

    same_social_group_sentences = []
    opposite_social_group_sentences = []

    for element in relevant_sentences:
        matches = re.finditer(pattern, element, re.DOTALL)
        for match in matches:
            same_social_group_sentences.append(match.group(1).strip())
            opposite_social_group_sentences.append(match.group(2).strip())

    return same_social_group_sentences, opposite_social_group_sentences

same_group,opposite_group=extract_groups(relevant_sentences)
with open('/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/SocialGroups/gender/Generated/Same_Social_Group.txt', 'w',encoding='utf-8') as f:
    for sentence in same_group:
        f.write(sentence + '\n')

with open('/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/SocialGroups/gender/Generated/Opposite_Social_Group.txt', 'w',encoding='utf-8') as f:
    for sentence in opposite_group:
        f.write(sentence + '\n')

#And write everything in a file
with open('/scratch0/bashyalb/DataAug4SocialBias/SentenceGeneration/Data/SocialGroups/gender/Generated/samegroupgen/samegroup_low_ttr.txt', 'w',encoding='utf-8') as f:
    for sentence in relevant_sentences:
        f.write(sentence + '\n')
