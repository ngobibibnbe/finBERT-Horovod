from pathlib import Path
import shutil
import os
import logging
import sys
import pandas as pd
from textblob import TextBlob
from pprint import pprint
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification
#sys.path.append('direction to the finBERT folder')
from finbert.finbert import *
import finbert.utils as tools
#import 
#load_ext autoreload
#autoreload 2

epoch = int(sys.argv[1])
data_size = int(sys.argv[2])



project_dir = Path.cwd().parent
pd.set_option('max_colwidth', -1)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.ERROR)
lm_path = project_dir/'models'/'language_model'
cl_path = project_dir/'models'/'classifier_model'/'finbert-sentiment'
cl_data_path = project_dir/'data'/'sentiment_data'
print(lm_path)

"""# Clean the cl_path
try:
    shutil.rmtree(cl_path) 
except:
    pass"""
#lm_path = project_dir/'models'/'language_model'/'finbertTRC2'

#bertmodel = AutoModelForSequenceClassification.from_pretrained(project_dir)

bertmodel = AutoModelForSequenceClassification.from_pretrained( 'bert-base-uncased',cache_dir=None, num_labels=3)
config = Config(   data_dir=cl_data_path,
                   bert_model=bertmodel,
                   train_batch_size=16,
                   eval_batch_size=16,
                   num_train_epochs=2,
                   model_dir=cl_path,
                   max_seq_length = 48,
                   learning_rate = 2e-2,
                   output_mode='classification',
                   warm_up_proportion=0.2,
                   local_rank=-1,
                   no_cuda=True,
                   discriminate=True,
                   gradual_unfreeze=True,
                   encoder_no=4)

finbert = FinBert(config)
finbert.base_model = 'bert-base-uncased'
finbert.config.discriminate=True
finbert.config.gradual_unfreeze=True
finbert.prepare_model(label_list=['positive','negative','neutral'])


# Get the training examples
train_data = finbert.get_data('train')
print (len(train_data))

model = finbert.create_the_model()


freeze = 6
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for i in range(freeze):
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False

def report(df, cols=['label','prediction','logits']):
    #print('Validation loss:{0:.2f}'.format(metrics['best_validation_loss']))
    cs = CrossEntropyLoss(weight=finbert.class_weights)
    loss = cs(torch.tensor(list(df[cols[2]])),torch.tensor(list(df[cols[0]])))
    print("Loss:{0:.2f}".format(loss))
    print("Accuracy:{0:.2f}".format((df[cols[0]] == df[cols[1]]).sum() / df.shape[0]) )
    print("\nClassification Report:")
    print(classification_report(df[cols[0]], df[cols[1]]))        

trained_model = finbert.train(train_examples = train_data[:data_size], model = model)

results = finbert.evaluate(examples=train_data[:data_size], model=trained_model)
#finbert.finish(results)
results['prediction'] = results.predictions.apply(lambda x: np.argmax(x,axis=0))
finbert.report(results,cols=['labels','prediction','predictions'])





