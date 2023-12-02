from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer, pipeline
import evaluate
import numpy as np
import timeit

removeColumns = ['url', 'source', 'author', 'date', 'title', 'abstract', 'tags']
labels = []
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")


def collectLabels(dataset):
    for item in enumerate(dataset):
        if item[1]['topic'] not in labels:
            labels.append(item[1]['topic'])


def datasetSetLabelId(item):
    item['label'] = labels.index(item['label'])
    return item


def preProcessDataset(dataset):
    dataset = dataset.remove_columns(removeColumns)
    dataset = dataset.rename_column("topic", "label")
    dataset = dataset.rename_column("content", "text")
    dataset = dataset.map(datasetSetLabelId)

    dataset["train"] = dataset["train"].shard(num_shards=20, index=0)

    return dataset


def preprocessTokenize(examples):
    return tokenizer(examples["text"], truncation=True)


def computeMetrics(eval_pred):
    predictions, label = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=label)



def textClassificationTraining():
    # load dataset
    dataset = load_dataset("batubayk/HU-News")
    print(dataset)
    print(dataset["train"][0])

    # collect labels
    collectLabels(dataset["train"])
    collectLabels(dataset["test"])
    collectLabels(dataset["validation"])
    print(labels)

    # pre process dataset
    dataset = preProcessDataset(dataset)
    print(dataset)
    print(dataset["train"][0])

    tokenized_dataset = dataset.map(preprocessTokenize, batched=True)

    # train
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(labels)
    )

    training_args = TrainingArguments(
        output_dir="model",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=computeMetrics,
    )

    train_start_time = timeit.default_timer()

    trainer.train()

    train_end_time = timeit.default_timer()

    print("Training time: ", train_end_time - train_start_time)

    trainer.save_model("model_exported")

    # inference, test prediction
    text = "Milyen lesz holnap az időjárás?"

    classifier = pipeline("text-classification", model="model_exported")
    result = classifier(text)

    print(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    textClassificationTraining()