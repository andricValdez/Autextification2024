
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
import evaluate
import pandas as pd
import gc

import utils

logging.set_verbosity_warning()
INDEX = 0


def llm_tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors='pt')

def llm_compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def llm_fine_tuning(train_text_docs, test_text_docs):
    #dataset_tst = load_dataset("yelp_review_full") # {'label': 0, 'text': "abc..."}
    #print(type(dataset_tst))
    #return
    
    train_dataset = [{'label': d['context']['target'], 'text': d['doc']} for d in train_text_docs]
    test_dataset = [{'label': d['context']['target'], 'text': d['doc']} for d in test_text_docs]
    merge_dataset = train_dataset + test_dataset

    train_dataset = Dataset.from_dict(pd.DataFrame(data=train_dataset))
    test_dataset = Dataset.from_dict(pd.DataFrame(data=test_dataset))
    merge_dataset = Dataset.from_dict(pd.DataFrame(data=merge_dataset))

    #dataset = DatasetDict({"train":train_dataset,"test":test_dataset})
    dataset = DatasetDict({"train":merge_dataset})

    print(type(dataset))
    print(dataset)
    #print(dataset["train"][100])

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased") 
    tokenized_datasets = dataset.map(llm_tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets)
    print(tokenized_datasets['train'].features)

    merge_loader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8)
    #train_loader = DataLoader(tokenized_datasets['train], shuffle=True, batch_size=8)
    #test_loader = DataLoader(tokenized_datasets['test], batch_size=8)
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()  
    
    num_epochs = 5
    num_training_steps = num_epochs * len(merge_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print("device: ", device)

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in merge_loader:
            #print(batch)
            #assert 0
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = evaluate.load("accuracy")
    model.eval()
    for batch in merge_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(metric.compute())
    torch.save(model.state_dict(), utils.OUTPUTS_PATH + 'llm_trained.pt')


def llm_get_embbedings(dataset, device='cpu'):
    print("llm_get_embbedings device: ", device)
    #pan24_dataset_train = utils.read_pan24_dataset(subset='train')
    #pan24_dataset_test = utils.read_pan24_dataset(subset='test')
    #pan24_dataset = pan24_dataset_train + pan24_dataset_test
    
    
    dataset = Dataset.from_dict(pd.DataFrame(data=dataset))
    dataset = dataset.with_format("torch", device=device)
    #print(dataset)
    dataset = DatasetDict({"dataset": dataset})

    #print(type(dataset))
    #print(dataset)

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased") 

    tokenized_datasets = dataset.map(llm_tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    #tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    #print(tokenized_datasets)
    #print(tokenized_datasets['dataset'].features)

    dataset_loader = DataLoader(tokenized_datasets['dataset'], batch_size=64)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
    model.load_state_dict(torch.load(utils.OUTPUTS_PATH + 'llm_trained.pt', map_location = device))

    model.to(device)
    #print("device: ", device)

    global INDEX 
    INDEX = 0
    test_outputs_model = []
    for batch in dataset_loader:
        #batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs_test = model(batch['input_ids'].to(device), output_hidden_states=True)
            last_hidden_states_test = outputs_test.hidden_states[-1]
            #test_outputs_model.append(outputs_test)
            # SAVE DATA
            #print(last_hidden_states_test.shape)
            emb = llm_extract_emb(outputs_test, tokenizer, tokenized_datasets)
            test_outputs_model += emb

    torch.cuda.empty_cache()
    gc.collect()

    #utils.save_json(test_outputs_model, file_path=utils.OUTPUTS_PATH + 'embeddings.json')
    #utils.save_data(test_outputs_model, file_name='embeddings')

    #print(len(test_outputs_model))
    return test_outputs_model
    #print(test_outputs_model[0]['id'])
    #print(test_outputs_model[0]['doc_index'])
    #print(test_outputs_model[0]['embeddings']['[CLS]'])

    #for o in test_outputs_model:
    #    print(o.hidden_states[-1].shape)


def llm_extract_emb(model, tokenizer, tokenized_datasets):
    embb_test_dict = []
    global INDEX
    #INDEX = 0
    batch = model.hidden_states[-1]
    #print(batch.shape, len(batch))
    for i in range(0, len(batch)):
        #print('--------------------------> ', i, INDEX)
        raw_tokens = [tokenizer.decode([token_id]) for token_id in tokenized_datasets['dataset'][INDEX]['input_ids']]
        doc_id = tokenized_datasets['dataset'][INDEX]['id']
        #label = tokenized_datasets['dataset'][INDEX]['labels']
        #print(raw_tokens)
        d = {"id": doc_id, "doc_index": INDEX, 'embeddings': {}} # ver forma de obtener ID de docu
        for token, embedding in zip(raw_tokens, batch[i]):
            #print(f"Token: {token}")
            #print(f"Token Len: {len(tokenized_text)}")
            #print(f"Embedding: {len(embedding)}")
            #print(f"Embedding: {embedding}")

            #d['embeddings'].append({"token": token, 'embedding': embedding })
            d['embeddings'][token] = embedding.cpu().detach().numpy().tolist()
            #d['embeddings'][token] = embedding
        embb_test_dict.append(d) 
        INDEX += 1

    #print("embb_test_dict: ", len(embb_test_dict))
    return embb_test_dict
    #embb_test_dict[0]['embeddings']['[CLS]']