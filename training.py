import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tqdm.pandas()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

for param in bert.parameters():
    param.requires_grad = False

max_length = 512  # The maximum number of tokens. Warning: the base version of BERT uses 512 tokens, modify with caution
batch_size = 16

train_text = {}
train_labels = {}
test_text = {}
test_labels = {}
val_text = {}
val_labels = {}

tokens_train = tokenizer.batch_encode_plus(
    train_text.values,
    max_length=max_length,
    padding='max_length',
    truncation=True
)
tokens_val = tokenizer.batch_encode_plus(
    val_text.values,
    max_length=max_length,
    padding='max_length',
    truncation=True
)
tokens_test = tokenizer.batch_encode_plus(
    test_text.values,
    max_length=max_length,
    padding='max_length',
    truncation=True
)

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.values)

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.values)

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.values)


train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


class BertBC(nn.Module):
    def __init__(self, _bert):
        super(BertBC, self).__init__()
        self.bert = _bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


model = BertBC(bert)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=0.001)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
print('\n Class weights: {:}'.format(class_weights))

weights = torch.tensor(class_weights, dtype=torch.float)
weights = weights.to(device)
cross_entropy = nn.CrossEntropyLoss()
epochs = 20


def train():
    model.train()
    total_loss, total_correct, total_examples = 0, 0, 0
    total_preds = []

    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        _, predicted = torch.max(preds, 1)
        total_correct += (predicted == labels).sum().item()
        total_examples += labels.size(0)

        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    accuracy = total_correct / total_examples
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, accuracy, total_preds


def evaluate():
    model.eval()
    total_loss, total_correct, total_examples = 0, 0, 0
    total_preds = []

    for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()

            _, predicted = torch.max(preds, 1)
            total_correct += (predicted == labels).sum().item()
            total_examples += labels.size(0)

            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    avg_loss = total_loss / len(val_dataloader)
    accuracy = total_correct / total_examples
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, accuracy, total_preds


best_valid_loss = float('inf')

train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    train_loss, train_accuracy, _ = train()
    valid_loss, valid_accuracy, _ = evaluate()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model, 'bert-bc.pt')
        print('\nBest model has been saved at {:} epoch'.format(epoch + 1))

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    print(f'\nTraining loss: {train_loss:.3f}, accuracy: {train_accuracy:.3f}')
    print(f'Validation loss: {valid_loss:.3f}, accuracy: {valid_accuracy:.3f}')
