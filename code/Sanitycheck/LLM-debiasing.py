import os

import pytorch_lightning as pl
import transformers

from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AdamW,
    DataCollatorForLanguageModeling,
    AutoConfig
)
import torch
from torch.utils.data import DataLoader, Dataset

transformers.logging.set_verbosity_error()

os.system("export TRANSFORMERS_CACHE=/scratch0/bashyalb/pretrained_models/")


class myMaskedLMDataset(Dataset):
    def __init__(self, file, tokenizer):
        self.tokenizer = tokenizer
        self.lines = self.load_lines(file)
        self.ids = self.encode_lines(self.lines)

    def load_lines(self, file):
        with open(file) as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines

    def encode_lines(self, lines):
        batch_encoding = self.tokenizer(
            lines, add_special_tokens=True, truncation=True, padding='max_length'
        )
        return batch_encoding["input_ids"]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)


class myMLM(pl.LightningModule):

    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.mlm = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)

    def forward(self, input_ids, labels):
        #print(labels) 
        return self.mlm(input_ids=input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        outputs = self.forward(input_ids=input_ids, labels=labels)
        loss = outputs[0]
        return {"loss": loss}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=args.lr)


if __name__ == "__main__":
    #os.system("export TRANSFORMERS_CACHE=/scratch0/bashyalb/pretrained_models/")
    print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='bert-base-uncased',
                        help="Full name or path or URL to trained NLI model")  # bert-base-uncased
    parser.add_argument("--train_data", type=str, default="data/debiasing.train.txt",
                        help="Filepath to training data")
    #parser.add_argument("--valid_data", type=str, default="data/debiasing.test.txt",
                      #  help="Filepath to training data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=64,
                        help="Maximum length allowed for input sentences. If longer, truncate.")
    parser.add_argument("--mlm_prob", type=float, default=0.15, help="Probability to mask random tokens in the input")
    parser.add_argument("--output", type=str, default='roberta',
                        help="ame of trained language model. Will be saved in 'saved_models/tasks/mlm/'")
    parser.add_argument("--no_cuda", action="store_true", help="No GPU?")
    parser.add_argument("--checkpoint_callback", action="store_true", help="Checkpoint callback?")
    parser.add_argument("--logger", action="store_true", help="Do log?")
    parser.add_argument("--pruning", type=bool, default=False)
    args = parser.parse_args()
    # print('into debiasing')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = myMaskedLMDataset(args.train_data, tokenizer)
    #valid_dataset = myMaskedLMDataset(args.valid_data, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=12
    )
    
    # print('start to init model')
    model = myMLM(args.model)
    ##   print(param.requires_grad)

    '''
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"{name} does not require grad!")
        else:
            print(f"{name} requires grad!")
    '''
    log_dir = "./"

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        mode="min")
    # print("debiasing fine-tuning")
    trainer = pl.Trainer(max_epochs=args.epochs, devices=[0,1,2]'' accelerator="gpu", strategy="ddp",
                         callbacks=[checkpoint_callback],
                         default_root_dir="./{}/".format(args.output))
    trainer.fit(model, train_loader)
    print(checkpoint_callback.best_model_path, "88888888888888888888888888888888888")
    torch.save(model.mlm.state_dict(),
               "./debiased..{0}.ckpt".format(args.model.split("-")[0]))
