import transformers
import torch
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from BugTriageDataset import BugTriageDataset
import torch
torch.cuda.empty_cache()

def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True): 
    return accuracy_score(np.argmax(y_true, axis=1),np.argmax(y_pred, axis=1))
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {'accuracy_thresh': accuracy_thresh(predictions, labels)}

model_ckpt = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, problem_type="multi_label_classification")

for f in range(1, 11):

    df_train = pd.read_csv(f"../Data/mozilla_seamonkey_fold{f}_train.csv")
    df_test = pd.read_csv(f"../Data/mozilla_seamonkey_fold{f}_test.csv")

    tranin_dev = df_train['Component'].values
    test_dev = df_test['Component'].values

    not_present_in_training = [i for i in test_dev if i not in tranin_dev]
    not_present_in_test = [i for i in tranin_dev if i not in test_dev]

    df_test = df_test[~df_test['Component'].isin(not_present_in_training)]
    df_train = df_train[~df_train['Component'].isin(not_present_in_test)]

    unique_assign_to = dict([(v, k) for k,v  in zip(range(1, len(df_train["Component"].unique())+1), df_train["Component"].unique())])

    df_train['label'] = df_train['Component'].map(lambda x: unique_assign_to[x])
    df_test['label'] = df_test['Component'].map(lambda x: unique_assign_to[x])

    df_train.rename(columns = {'label':'Y', 'text':'X'}, inplace = True)


    df_test.rename(columns = {'label':'Y', 'text':'X'}, inplace = True)

    df_train = df_train[['X','Y']]

    df_test = df_test[['X','Y']]

    df_train = pd.get_dummies(df_train, columns=["Y"])
    df_test = pd.get_dummies(df_test, columns=["Y"])

    dev_cols = [c for c in df_train.columns if c not in ["X", "Y"]]
    df_train["labels"] = df_train[dev_cols].values.tolist()
    df_test["labels"] = df_test[dev_cols].values.tolist()

    train_encodings = tokenizer(df_train["X"].values.tolist(), truncation=True)
    test_encodings = tokenizer(df_test["X"].values.tolist(), truncation=True)

    train_labels = [torch.tensor([float(f) for f in x]) for x in df_train["labels"].values.tolist()]
    test_labels = [torch.tensor([float(f) for f in x]) for x in df_test["labels"].values.tolist()]

    train_dataset = BugTriageDataset(train_encodings, train_labels)
    test_dataset = BugTriageDataset(test_encodings, test_labels)

    num_labels=len(dev_cols)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels, problem_type="multi_label_classification").to('cuda')


    batch_size = 32
    logging_steps = len(train_dataset) // batch_size

    args = TrainingArguments(
        output_dir="bug_logs",
        overwrite_output_dir = True,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_steps=logging_steps,
        save_total_limit=1
    )


    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=test_dataset, tokenizer=tokenizer,compute_metrics=compute_metrics)
    
    trainer.train()

    trainer.evaluate()

    pred = trainer.predict(test_dataset)
    
    np.save(f"pred_mozilla_seamonkey_roberta_fold{f}.npy", pred)