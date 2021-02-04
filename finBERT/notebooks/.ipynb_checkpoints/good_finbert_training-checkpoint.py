# -*- coding: utf-8 -*-

import logging
import os
import shutil
import sys
#sys.path.append("../")

os.chdir('../')
path = os.getcwd()

print("--------------------",path)

import finbert.utils as tools
from finbert.finbert import *
from pathlib import Path
from pprint import pprint

import horovod.torch as hvd
from sklearn.metrics import classification_report
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification



# sys.path.append('/content/gdrive/MyDrive/Projets/prosus_training/finBERT')



# %load_ext autoreload
# %autoreload 2

project_dir = Path.cwd().parent
pd.set_option("max_colwidth", -1)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.ERROR,
)

"""## Prepare the model

### Setting path variables:
1. `lm_path`: the path for the pre-trained language model (If vanilla Bert is used then no need to set this one).
2. `cl_path`: the path where the classification model is saved.
3. `cl_data_path`: the path of the directory that contains the data files of `train.csv`, `validation.csv`, `test.csv`.
---

In the initialization of `bertmodel`, we can either use the original pre-trained weights from Google by giving `bm = 'bert-base-uncased`, or our further pre-trained language model by `bm = lm_path`


---
All of the configurations with the model is controlled with the `config` variable.
"""

lm_path = project_dir / "models" / "language_model" / "finbertTRC2"
cl_path = project_dir / "models" / "classifier_model" / "finbert-sentiment"
cl_data_path = project_dir / "data" / "sentiment_data"
print("---------------------", cl_data_path)

"""###  Configuring training parameters

[texte du lien](https://)You can find the explanations of the training parameters in the class docsctrings.
"""

# Clean the cl_path
try:
    shutil.rmtree(cl_path)
except:
    pass
lm_path = project_dir / "models" / "language_model" / "finbertTRC2"

# bertmodel = AutoModelForSequenceClassification.from_pretrained(lm_path,cache_dir=None, num_labels=3)
bertmodel = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", cache_dir=None, num_labels=3
)

hvd.init()
config = Config(
    data_dir=cl_data_path,
    bert_model=bertmodel,
    num_train_epochs=2,
    model_dir=cl_path,
    max_seq_length=48,
    train_batch_size=16,
    learning_rate=2e-5,
    output_mode="classification",
    warm_up_proportion=0.2,
    local_rank=hvd.local_rank(),
    no_cuda=True,
    discriminate=True,
    gradual_unfreeze=True,
)

"""`finbert` is our main class that encapsulates all the functionality. The list of class labels should be given in the prepare_model method call with label_list parameter.

> Bloc en retrait


"""

finbert = FinBert(config)
finbert.base_model = "bert-base-uncased"
finbert.config.discriminate = True
finbert.config.gradual_unfreeze = True

finbert.prepare_model(label_list=["positive", "negative", "neutral"])

"""## Fine-tune the model"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/gdrive/MyDrive/Projets/prosus_training/finBERT/


# Get the training examples
train_data = finbert.get_data("train")
train_data = train_data[:100]
# train_data

model = finbert.create_the_model()

"""### [Optional] Fine-tune only a subset of the model
The variable `freeze` determines the last layer (out of 12) to be freezed. You can skip this part if you want to fine-tune the whole model.

<span style="color:red">Important: </span>
Execute this step if you want a shorter training time in the expense of accuracy.
"""

# This is for fine-tuning a subset of the model.

freeze = 6

for param in model.bert.embeddings.parameters():
    param.requires_grad = False

for i in range(freeze):
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False

"""### Training"""
finbert.load_data()
trained_model = finbert.train(train_examples=train_data, model=model)

"""## Test the model

`bert.evaluate` outputs the DataFrame, where true labels and logit values for each example is given
"""

test_data = finbert.get_data("test")

results = finbert.evaluate(examples=test_data, model=trained_model)

"""### Prepare the classification report"""


def report(df, cols=["label", "prediction", "logits"]):
    print("\n nous sommes dans le report")
    # print('Validation loss:{0:.2f}'.format(metrics['best_validation_loss']))
    cs = CrossEntropyLoss(weight=finbert.class_weights)
    loss = cs(torch.tensor(list(df[cols[2]])), torch.tensor(list(df[cols[0]])))
    print("Loss:{0:.2f}".format(loss))
    print("Accuracy:{0:.2f}".format((df[cols[0]] == df[cols[1]]).sum() / df.shape[0]))
    print("\nClassification Report:")
    print(classification_report(df[cols[0]], df[cols[1]]))


results["prediction"] = results.predictions.apply(lambda x: np.argmax(x, axis=0))

report(results, cols=["labels", "prediction", "predictions"])
