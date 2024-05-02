import torch
import torch.nn as nn
from training import device, val_dataloader
from sklearn.metrics import classification_report


model = torch.load('bert-bc.pt')
model.eval()

all_preds = []
all_labels = []

for batch in val_dataloader:

    batch = tuple(t.to(device) for t in batch)
    input_ids, attention_mask, labels = batch

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    probs = nn.functional.softmax(outputs, dim=1)
    preds = probs.argmax(dim=1)

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    all_preds.extend(preds)
    all_labels.extend(labels)

print(classification_report(all_labels, all_preds))
