
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import logging
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import evaluate
import pandas as pd
import gc

import utils

logging.set_verbosity_warning()
INDEX = 0

# google-bert/bert-base-cased
# FacebookAI/roberta-base
# google-bert/bert-base-multilingual-cased      multiligual
# FacebookAI/xlm-roberta-base                   multiligual
# intfloat/multilingual-e5-large                multiligual
LLM_HF_NAME = "intfloat/multilingual-e5-large"

# andricValdez/multilingual-e5-large-finetuned-autext24
# andricValdez/bert-base-multilingual-cased-finetuned-autext24
# andricValdez/bert-base-multilingual-cased-finetuned-autext24-subtask2
# andricValdez/multilingual-e5-large-finetuned-autext24-subtask2
LLM_HF_FINETUNED_NAME = "andricValdez/bert-base-multilingual-cased-finetuned-autext24"


def llm_tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors='pt')
 

def llm_compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def llm_fine_tuning(model_name, train_set_df, val_set_df, device, num_labels=6):

    model_name = f"{LLM_HF_NAME}-finetuned-{model_name}"   
    dataset = DatasetDict({
        "train": Dataset.from_dict(train_set_df),
        "validation": Dataset.from_dict(pd.DataFrame(data=val_set_df))
    })
    print(dataset)
    tokenizer = AutoTokenizer.from_pretrained(LLM_HF_NAME) 
    tokenized_dataset = dataset.map(llm_tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    print(tokenized_dataset)

    batch_size = 8
    model = (AutoModelForSequenceClassification.from_pretrained(LLM_HF_NAME, num_labels=num_labels).to(device))
    logging_steps = len(tokenized_dataset["train"]) // batch_size
    
    training_args = TrainingArguments(
        output_dir=utils.OUTPUT_DIR_PATH + 'finetuned_hf_models/' + model_name,
        num_train_epochs=5,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=True,
        log_level="error"
    )

    trainer = Trainer(
        model=model, 
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer
    )
    trainer.train()
    
    preds_output = trainer.predict(tokenized_dataset["validation"])
    print(preds_output.metrics)
    y_preds = np.argmax(preds_output.predictions, axis=1)
    
    trainer.push_to_hub()


def llm_get_embbedings(dataset, subset, emb_type='llm_cls', device='cpu', output_path='', save_emb=False, llm_finetuned_name=LLM_HF_FINETUNED_NAME, num_labels=2):

    dataset = Dataset.from_dict(pd.DataFrame(data=dataset))
    dataset = dataset.with_format("torch", device=device)
    dataset = DatasetDict({"dataset": dataset})

    print(f"NFI -> device: {device} | subset: {subset} | emb_type: {emb_type} | save_emb: {save_emb} ")
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(llm_finetuned_name) 
    tokenized_datasets = dataset.map(llm_tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    dataset_loader = DataLoader(tokenized_datasets['dataset'], batch_size=utils.LLM_GET_EMB_BATCH_SIZE_DATALOADER)
    model = AutoModelForSequenceClassification.from_pretrained(llm_finetuned_name, num_labels=num_labels)
    model.to(device)

    global INDEX 
    INDEX = 0
    emb_output_model = []
    for step, batch in enumerate(tqdm(dataset_loader)):

        with torch.no_grad():
            outputs_model = model(batch['input_ids'].to(device), output_hidden_states=True)
            last_hidden_state = outputs_model.hidden_states[-1]

            embeddings_lst = llm_extract_emb(
                outputs_model=outputs_model, 
                batch=batch, batch_step=step, 
                subset=subset, tokenizer=tokenizer, 
                tokenized_datasets=tokenized_datasets, 
                emb_type=emb_type
            )
            if save_emb == True:
                utils.save_llm_embedings(embeddings_data=embeddings_lst, emb_type=emb_type, batch_step=step, file_path=output_path)
            else:    
                emb_output_model += embeddings_lst

        torch.cuda.empty_cache()
        gc.collect()

    return emb_output_model


def llm_extract_emb(outputs_model, batch, batch_step, subset, tokenizer, tokenized_datasets, emb_type):
    embeddings_dict_lst = []
    global INDEX
    #INDEX = 0
    last_hidden_state = outputs_model.hidden_states[-1]
    if emb_type == 'llm_cls': # llm_doc
        embeddings = last_hidden_state[:,0] # get cls
        embeddings_dict_lst.append({"batch": batch_step, "subset": subset, "doc_id": batch['id'], "labels": batch['label'], "embedding": embeddings})
        return embeddings_dict_lst
    else: # llm_word
        for i in range(0, len(last_hidden_state)):
            raw_tokens = [tokenizer.decode([token_id]) for token_id in tokenized_datasets['dataset'][INDEX]['input_ids']]
            doc_id = tokenized_datasets['dataset'][INDEX]['id']
            label = tokenized_datasets['dataset'][INDEX]['label']
            d = {"doc_id": doc_id, "doc_index": INDEX, 'label': label, 'embedding': {}} # ver forma de obtener ID de docu
            for token, embedding in zip(raw_tokens, last_hidden_state[i]):
                d['embedding'][token] = embedding.cpu().detach().numpy().tolist()
            embeddings_dict_lst.append(d) 
            INDEX += 1

        return embeddings_dict_lst
