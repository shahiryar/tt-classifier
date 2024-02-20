import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import create_optimizer
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from transformers.keras_callbacks import PushToHubCallback
import json
import os

#Load Log data
log_df = pd.read_csv('dataset/log.csv')
log_df = log_df[["abstract", "label"]]
log_df.columns = ['text', 'label']

#Load previous dataset
labeled_dataset_path = 'dataset/Paper Abstracts and Categorization Shahiryar.csv'
labled_df = pd.read_csv(labeled_dataset_path)
df_clean = labled_df[["Paper ID", "Title", "Abstract", "TT", "TC"]]
to_lower = lambda s: s.lower() if s==s else ""
strip = lambda s: s.strip() if s==s else ""
df_clean.Title = df_clean["Title"].map(to_lower)
df_clean.Abstract = df_clean.Abstract.map(to_lower)
df_clean.TT = df_clean.TT.map(strip)
df_clean.TC = df_clean.TC.map(strip)
df_clean = df_clean.iloc[:-3, :-1]
df_clean['text'] = df_clean['Title']+" "+df_clean['Abstract']
df_clean['label'] = df_clean['TT']
df_clean = df_clean.iloc[:, -2:]

#Concatinate the two dataframes and drop duplicates
df = pd.concat([df_clean, log_df])
df.drop_duplicates(subset=["text"], inplace=True)

#Count and register classes
classes = sorted(list(df.label.value_counts().index))
NUM_CLASSES = len(classes)
id2label = { i: el for i, el in enumerate(classes)}
label2id = { el: i for i, el in enumerate(classes)}

#preprocess data
df.label = df.label.replace(label2id)
df.reset_index(drop=True, inplace=True)
train_dataset = Dataset.from_pandas(df, preserve_index=False)

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#preprocessing function to tokenize text and truncate sequences to be no longer than DistilBERT’s maximum input length:
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

#apply tokenizer
tokenized_papers = train_dataset.map(preprocess_function)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

#preparing model
accuracy = evaluate.load("accuracy")
#function that passes your predictions and labels to compute to calculate the accuracy:
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


#defining model hyperparameters
batch_size = 7
num_epochs = 15
batches_per_epoch = len(tokenized_papers) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

#defining model
model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=NUM_CLASSES, id2label=id2label, label2id=label2id
)

#preparing dataset for model training
tf_train_set = model.prepare_tf_dataset(
    tokenized_papers,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

#compiling model
model.compile(optimizer=optimizer)


#defining callbacks

ACCESS_TOKEN = os.environ['HUGGINGFACE_HUB_ACCESS_CODE']
push_to_hub_callback = PushToHubCallback(
    output_dir="tt_abstract_classifier",
    tokenizer=tokenizer,
    hub_token=ACCESS_TOKEN
)
callbacks = [push_to_hub_callback]

#train the model
model.fit(x=tf_train_set, epochs=1, callbacks=callbacks)

print("Model trained Successfully")
#store classes in the classes file
with open('resources/label2id.json', 'w') as classes_file:
    json.dump(label2id, classes_file)