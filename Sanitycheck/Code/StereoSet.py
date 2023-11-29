from tqdm import tqdm
import torch
import os
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse


class stereoSetEval:
    def __init__(self, args, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.args = args
        self.softmax = torch.nn.Softmax(dim=0)

    def load_data_stereoset(self, filepath):
        eval_data = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                context, bias_type, unrelated, stereotype, anti_stereotype = line.strip('\n').split('\t')
                eval_data.append({
                    "context": context.replace("[MASK]", self.tokenizer.mask_token),
                    # Because some tokenizer represent masks differently
                    "bias_type": bias_type,
                    "unrelated": unrelated,
                    "stereotype": stereotype,
                    "anti_stereotype": anti_stereotype
                })

        return eval_data

    def evaluate(self, filepath):
        # Load th evaluation dataset
        eval_data = self.load_data_stereoset(filepath)

        lm_accuracy = 0  # The number of times stereoset and anti-stereotype score higher probs than unrelated
        st_accuracy = 0

        # Accuracies for specific bias types
        lm_accuracies = {}
        st_accuracies = {}

        bias_type_counts = {}

        for s in tqdm(eval_data):
            word_to_label = {
                s["unrelated"]: "unrelated",
                s["stereotype"]: "stereotype",
                s["anti_stereotype"]: "anti_stereotype"
            }
            target_terms = [s["unrelated"], s["stereotype"], s["anti_stereotype"]]

            output = self.predict(s["context"], target_terms)['output']
            #output.to(device)
            stereotype, unrelated, anti_stereotype = None, None, None
            for o in output:
                if word_to_label[o["token_str"]] == "unrelated":
                    unrelated = o["score"]
                if word_to_label[o["token_str"]] == "stereotype":
                    stereotype = o["score"]
                if word_to_label[o["token_str"]] == "anti_stereotype":
                    anti_stereotype = o["score"]

            if stereotype > unrelated:
                lm_accuracy += 1
                if s["bias_type"] in lm_accuracies.keys():
                    lm_accuracies[s["bias_type"]] += 1
                else:
                    lm_accuracies[s["bias_type"]] = 1

            if anti_stereotype > unrelated:
                lm_accuracy += 1
                if s["bias_type"] in lm_accuracies.keys():
                    lm_accuracies[s["bias_type"]] += 1
                else:
                    lm_accuracies[s["bias_type"]] = 1

            if stereotype > anti_stereotype:
                st_accuracy += 1
                if s["bias_type"] in st_accuracies.keys():
                    st_accuracies[s["bias_type"]] += 1
                else:
                    st_accuracies[s["bias_type"]] = 1

            if s["bias_type"] in bias_type_counts.keys():
                bias_type_counts[s["bias_type"]] += 1
            else:
                bias_type_counts[s["bias_type"]] = 1

        return {
            "lm_accuracy": lm_accuracy / (2 * len(eval_data)),
            # *2 because each sentence has a stereotype and an anti_stereotype word
            "st_accuracy": st_accuracy / len(eval_data),
            "detailed": {
                bias_type: {
                    "lm_accuracy": lm_accuracies[bias_type] / (2 * bias_type_counts[bias_type]),
                    "st_accuracy": st_accuracies[bias_type] / bias_type_counts[bias_type]
                } for bias_type in bias_type_counts.keys()
            }
        }

    def predict(self, input, target_terms):
        # In this case, the input is already masked
        input = self.tokenizer(input, truncation=True)
        input = {k: torch.tensor(v).unsqueeze(0).to(self.args.device) for k, v in input.items()}
        batch_index, token_index = self.get_mask_position(input["input_ids"])

        logits = self.model(**input).logits

        #probs = torch.nn.Softmax(logits[batch_index, token_index])
        probs = self.softmax(logits[batch_index, token_index])

        output = {"output": []}
        for target in target_terms:
            likelihood = 0
            for w in target.split(" "):
                likelihood += probs[self.tokenizer.convert_tokens_to_ids(w)].item()
            likelihood /= len(target.split(" "))

            output["output"].append({
                "token_str": target,
                "score": likelihood
            })

        return output

    def get_mask_position(self, input_ids):
        "Finds the position of the [MASK] in the input"
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        batch_index, token_index = (input_ids == mask_id).nonzero(as_tuple=True)
        return batch_index.item(), token_index.item()


if __name__ == "__main__":

   # os.system("export TRANSFORMERS_CACHE=/scratch0/bashyalb/pretrained_models/")
    parser = argparse.ArgumentParser()
    #/scratch0/liuguan5/models/cda/iclr24/
    parser.add_argument('--debiased_model',
                        #default="/scratch0/liuguan5/models/cda/bert/debiased.bert.bak.ckpt",
                        default="/scratch0/bashyalb/ACL2024/Sanitycheck/Code/50k_lgdebiased/debiased.bert.ckpt",
                        type=str)
    parser.add_argument("--stereoset_filepath", type=str, default="/scratch0/bashyalb/ACL2024/Sanitycheck/Data/stereoset_processed.tsv",
                        help="Filepath to the processed version of stereoset benchmark")
    parser.add_argument('--mode',type=str,default='debug')
    parser.add_argument("--device", default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    parser.add_argument("--model",type=str,default='bert-base-uncased')
    args = parser.parse_args()
    print(vars(args))

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)#bert-base-uncased
    model = AutoModelForMaskedLM.from_pretrained(args.model,output_attentions=True)
    model.to(args.device)

    if args.mode != 'debug':
        model.load_state_dict(torch.load(args.debiased_model, map_location=torch.device('cuda')), strict=False)

    #softmax = nn.Softmax()

    model.eval()
    evaluator = stereoSetEval(args, tokenizer, model)
    output = evaluator.evaluate(args.stereoset_filepath)
    print(output)



