---
title: 'MLOps Basics [Week 0]: Project Setup'
date: '2021-05-31'
lastmod: '2021-05-31'
tags: ['mlops', 'deeplearning', 'nlp', 'deployment']
draft: false
summary: "The goal of the series is to understand the basics of MLOps (model building, monitoring, configurations, testing, packaging, deployment, cicd). As a first step, Let's setup the project first."
images: ['/static/images/canada/mountains.jpg', '/static/images/canada/toronto.jpg']
---

# 🎬 Start of the series

The goal of the series is to understand the basics of MLOps (model building, monitoring, configurations, testing, packaging, deployment, cicd). As a first step, Let's setup the project first. I am particularly interested towards NLP (personal bias) but the process and tools stays the same irrespective of the project. I will be using a simple classification task.

In this post, I will be going through the following topics:

- `How to get the data?`
- `How to process the data?`
- `How to define dataloaders?`
- `How to declare the model?`
- `How to train the model?`
- `How to do the inference?`

_Note: Basic knowledge of Machine Learning is needed_

# 🛠 Deep Learning Library

There are many libraries available to develop deeplearning projects. The prominent ones are:

- [`Tensorflow`](https://www.tensorflow.org/)

- [`Pytorch`](https://pytorch.org/)

- [`Pytorch Lightning`](https://www.pytorchlightning.ai/) (Pytorch lightning is a wrapper around pytorch)

and many more...

I will be using `Pytorch Lightning` since it automates a lot of engineering code and comes with many cool features.

# 📚 Dataset

I will be using `CoLA`(Corpus of Linguistic Acceptability) dataset. The task is about given a sentence it has to be classified into one of the two classes.

- ❌ `Unacceptable`: Grammatically not correct
- ✅ `Acceptable`: Grammatically correct

I am using ([`Huggingface datasets`](https://huggingface.co/docs/datasets/quicktour.html)) to download and load the data. It supports `800+` datasets and also can be used with custom datasets.

Downloading the dataset is as easy as

```python
cola_dataset = load_dataset("glue", "cola")
print(cola_dataset)
```

```shell
DatasetDict({
    train: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 8551
    })
    validation: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1043
    })
    test: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1063
    })
})
```

Let's see a sample datapoint

```python
train_dataset = cola_dataset['train']
print(train_dataset[0])
```

```shell
{
    'idx': 0,
    'label': 1,
    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."
}
```

# 🛒 Loading data

Data pipelines can be created with:

- 🍦 Vanilla Pytorch [`DataLoaders`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- ⚡ Pytorch Lightning [`DataModules`](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html)

`DataModules` are more structured definition, which allows for additional optimizations such as automated distribution of workload between CPU & GPU.
Using `DataModules` is recommended whenever possible!

A `DataModule` is defined by an interface:

- `prepare_data` (optional) which is called only once and on 1 GPU -- typically something like the data download step we have below
- `setup`, which is called on each GPU separately and accepts **stage** to define if we are at **fit** or **test** step
- `train_dataloader`, `val_dataloader` and `test_dataloader` to load each dataset

A `DataModule` encapsulates the five steps involved in data processing in PyTorch:

- Download / tokenize / process.
- Clean and (maybe) save to disk.
- Load inside Dataset.
- Apply transforms (rotate, tokenize, etc…).
- Wrap inside a DataLoader.

The DataModule code for the project looks like:

```python
class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        # processing the data
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )
```

# 🏗️ Building a Model with Lightning

In PyTorch Lightning, models are built with [`LightningModule`](https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html), which has all the functionality of a vanilla `torch.nn.Module` (🍦) but with a few delicious cherries of added functionality on top (🍨).
These cherries are there to cut down on boilerplate and help separate out the ML engineering code from the actual machine learning.

For example, the mechanics of iterating over batches as part of an epoch are extracted away, so long as you define what happens on the `training_step`.

To make a working model out of a `LightningModule`, we need to define a new `class` and add a few methods on top.

A `LightningModule` is defined by an interface:

- `init` define the initialisations here
- `forward` what should for a given input (keep only the forward pass things here not the loss calculations / weight updates)
- `training_step` training step (loss calculation, any other metrics calculations.) No need to do weight updates
- `validation_step` validation step
- `test_step` (optional)
- `configure_optimizers` define what optimizer to use

There are a lot of other functions also which can be used. Check the [doucmentation](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) for all other methods.

The LightningModule code for the project looks like:

```python
class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)
        val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
        val_acc = torch.tensor(val_acc)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
```

# 👟 Training

The `DataLoader` and the `LightningModule` are brought together by a `Trainer`, which orchestrates data loading, gradient calculation, optimizer logic, and logging.

We setup `Trainer` and can customize several options, such as logging, gradient accumulation, half precision training, distributed computing, etc.

We'll stick to the basics for this example.

```python
cola_data = DataModule()
cola_model = ColaModel()

trainer = pl.Trainer(
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=1,
    fast_dev_run=False,
)
trainer.fit(cola_model, cola_data)

```

By enabling `fast_dev_run=True`, will run one batch of training step and one batch of validation step **(always good to do this)**. It can catch any mistakes happening the validation step right away rather than waiting for the whole training to be completed.

## 📝 Logging

`Logging` of the model training is as simple as

```python
cola_data = DataModule()
cola_model = ColaModel()

trainer = pl.Trainer(
    default_root_dir="logs",
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=1,
    fast_dev_run=False,
    logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
)
trainer.fit(cola_model, cola_data)
```

It will create a directory called `logs/cola` if not present. You can visualise the tensorboard logs using the following command

```bash
tensorboard --logdir logs/cola
```

You can see the tensorboard at `http://localhost:6006/`

## 🔁 Callback

`Callback` is a self-contained program that can be reused across projects.

As an example, I will be implementing **EarlyStopping** callback. This helps the model not to overfit by mointoring on a certain parameter (`val_loss` in this case)
The best model will be saved in the `dirpath`.

Refer to the [documentation](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html) to learn more about callbacks.

```python
cola_data = DataModule()
cola_model = ColaModel()

checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="val_loss", mode="min"
)

trainer = pl.Trainer(
    default_root_dir="logs",
    gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=1,
    fast_dev_run=False,
    logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
    callbacks=[checkpoint_callback],
)
trainer.fit(cola_model, cola_data)
```

# 🔍 Inference

Once the model is trained, we can use the trained model to get predictions on the run time data. Typically `Inference` contains:

- Load the trained model
- Get the run time (inference) input
- Convert the input in the required format
- Get the predictions

```python
class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        # loading the trained model
        self.model = ColaModel.load_from_checkpoint(model_path)
        # keep the model in eval mode
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text):
        # text => run time input
        inference_sample = {"sentence": text}
        # tokenizing the input
        processed = self.processor.tokenize_data(inference_sample)
        # predictions
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions
```

This conculdes the post. In the next post, I will be going through:

- `How to monitor model performance with Weights and Bias?`

Complete code for this post can also be found here: [Github](https://github.com/graviraja/MLOps-Basics)
