import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from pprint import pprint

import horovod.torch as hvd
import numpy as np
import torch
from sklearn.metrics import classification_report
from textblob import TextBlob
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification

import finbert.utils as tools
from finbert.finbert import *

hvd.init()
torch.manual_seed(0)
np.random.seed(0)

project_dir = Path.cwd().parent
pd.set_option("max_colwidth", -1)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.ERROR,
)


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

bertmodel = AutoModelForSequenceClassification.from_pretrained(
    lm_path, cache_dir=None, num_labels=3
)
# bertmodel = AutoModelForSequenceClassification.from_pretrained(
#    "bert-base-uncased", cache_dir=None, num_labels=3
# )


def metric_average(val, name):
    ###fonction spéciale pour Horovod
    tensor = val.clone().detach()
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def report(df, finbert, cols=["label", "prediction", "logits"]):
    # print("\n nous sommes dans le report")
    # print('Validation loss:{0:.2f}'.format(metrics['best_validation_loss']))
    cs = CrossEntropyLoss(weight=finbert.class_weights)
    loss = cs(torch.tensor(list(df[cols[2]])), torch.tensor(list(df[cols[0]])))

    test_loss = metric_average(loss, "avg_loss")
    test_accuracy = metric_average(
        (df[cols[0]] == df[cols[1]]).sum() / df.shape[0], "avg_accuracy"
    )

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        #print(
        #    "\n**************************************************************Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
        #        test_loss, 100.0 * test_accuracy
        #    )
        #)
        return test_loss, 100.0 * test_accuracy
    return test_loss, test_accuracy


def training(epoch=2, lr=2e-5, batch_size=16, no_cuda=True, freeze=6):

    config = Config(
        data_dir=cl_data_path,
        bert_model=bertmodel,
        num_train_epochs=epoch,
        model_dir=cl_path,
        max_seq_length=48,
        train_batch_size=16,
        learning_rate=2e-2,
        output_mode="classification",
        warm_up_proportion=0.2,
        local_rank=hvd.local_rank(),
        no_cuda=no_cuda,
        discriminate=True,
        gradual_unfreeze=True,
    )

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
    # train_data = train_data[:100]

    # train_data

    model = finbert.create_the_model()

    freeze = freeze

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    for i in range(freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

            finbert.load_data()
    trained_model = finbert.train(train_examples=train_data, model=model)

    """### Training"""

    """## Test the model

    `bert.evaluate` outputs the DataFrame, where true labels and logit values for each example is given
    """

    test_data = finbert.get_data("test")
    results = finbert.evaluate(examples=test_data, model=trained_model)

    """### Prepare the classification report"""

    results["prediction"] = results.predictions.apply(lambda x: np.argmax(x, axis=0))
    return report(results, finbert, cols=["labels", "prediction", "predictions"])


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--freeze", type=int, default=6)
args = parser.parse_args()

#if hvd.local_rank() == 0:
#from tensorboard import program

#tb = program.TensorBoard()
#tb.configure(logdir="runs", host="0.0.0.0", port=6666, reload_interval=180)
#url = tb.launch()
#print(f"################### TensorBoard URL {url} ##########################")

# for epoch  in range(2):
test_loss, test_accuracy = training(
    epoch=args.epoch, batch_size=args.batch_size, lr=args.lr, no_cuda=not args.cuda, freeze=args.freeze
)
print("accuracy_in_term_of number of process", test_loss, test_accuracy)
#% tensorboard --logdir=runs --bind_all
