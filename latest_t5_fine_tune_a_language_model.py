from huggingface_hub import notebook_login

notebook_login()

import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.empty_cache()

from typing import Dict, Tuple
from datasets import list_datasets, load_dataset, DatasetDict,Dataset
from collections import Counter
from typing import List, Dict, Union, Callable, Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pprint import pprint
import torch.nn as nn
import evaluate
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig
from evaluate import TextClassificationEvaluator, Metric, EvaluationModuleInfo

import transformers

print(transformers.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

"""## Preparing the dataset"""

ds_train = load_dataset("Sree1994/blm_strict_small", split="train")
ds_valid = load_dataset("Sree1994/blm_strict_small", split="valid")

raw_datasets = DatasetDict(
    {
        "train": ds_train,
        "valid": ds_valid
    }
)

raw_datasets

model_checkpoint = "t5-small"

from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Trainer, TrainingArguments,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
context_length = 128
vocab_size = tokenizer.vocab_size

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length <= context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
tokenized_datasets

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")

config = AutoConfig.from_pretrained(
    "t5-small",
    vocab_size=len(tokenizer),
    random_init=True,
    is_decoder=True
)
print(len(tokenizer))

model = T5ForConditionalGeneration(config)
model.init_weights()
model_size = sum(t.numel() for t in model.parameters())
print(f"T5 size: {model_size/1000**2:.1f}M parameters")

class Cal_Perplexity(Metric):
    """
    You can define custom metrics! In this case I do this to compute Macro-F1, which averages per-class F1 scores
    """
    pp_metric_info: EvaluationModuleInfo = evaluate.load("perplexity")._info()

    def _info(self) -> EvaluationModuleInfo:
        # we'll just say the info is the same in this case
        return MyMacroF1Metric.pp_metric_info

    def _compute(self, loss) -> Dict[str, Any]:
        # we can just call the sklearn implementation! Metrics in huggingface generally correspond with sklearn metrics
        # when applicable
        pp = torch.exp()
        return {"perplexity": float(pp) if pp.size == 1 else pp}

from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./output_dir",
    do_train=True,
    do_eval=True,
    do_predict=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=4e-4,
    save_steps=500,
    #fp16=True,
    push_to_hub=False,
    
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trn = trainer.train()
model = trainer.model  # make sure to load_best_model_at_end=True!

# run a final evaluation on the test set
val = trainer.evaluate(metric_key_prefix="test", eval_dataset=tokenized_datasets["valid"])

trn.training_loss

valid = val.get("test_loss")
torch.exp(torch.tensor(valid))