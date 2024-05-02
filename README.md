# BertBC - _bert for binary classification_
### description:
BertBC is an LLM model based on the base version of the BERT transformer. BertBC is designed for binary classification tasks. This model consists of a pre-trained BERT that extracts contextualized word embeddings and then adds several fully connected layers on top for class classification.

_example tasks:_
- Spam filtering
- Sentiment analysis
- Detecting toxic comments
- Identifying generated texts

Thanks to the weighted loss function, the model becomes more robust to imbalanced datasets. This makes BertBC easy to use as it automatically adjusts the class weights according to their representation in the data, ensuring a more balanced training process and improved performance.
___
### Usage:
File 'training.py' contains the main code for training the model. You need to load your own data. Also, pay attention to the max_length parameter, which determines the number of tokens. Base version of BERT uses 512 tokens, modify with caution. Use the following code to get a histogram of sentences and determine the required token value:
```python
import matplotlib.pyplot as plt
seq_len = [len(str(i).split()) for i in train_text]
pd.Series(seq_len).hist(bins=50)
plt.show()
```
   
   
<br><br>To obtain a classification report, use the code from the 'metrics_calculation.py' file. Here's an example output:<br>
|          | precision | recall | f1-score | support |
|----------|-----------|--------|----------|---------|
|    0     |    0.99   |  0.98  |   0.99   |   3506  |
|    1     |    0.97   |  0.99  |   0.98   |   2323  |
|accuracy  |    0.98   |        |          |   5829  |
|macro avg |    0.98   |  0.98  |   0.98   |   5829  |
|weighted avg| 0.98   |  0.98  |   0.98   |   5829  |
___
### Installing dependencies:
To use this code, you need to install the dependencies from the 'requirements.txt' file. You can do this using Python package management tools such as pip.
```bash
pip install -r requirements.txt
```
