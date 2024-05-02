import torch
import torch.nn as nn
from transformers import BertTokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


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


model = torch.load('model')
model.eval()

text = "example text"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))

probs = nn.functional.softmax(outputs, dim=1)
predictions = torch.argmax(probs, dim=1).cpu().numpy()
probs = probs.cpu().detach().numpy()

print(f"First class: {probs[0][0]:.4f}")
print(f"Second class: {probs[0][1]:.4f}")
print(probs)
