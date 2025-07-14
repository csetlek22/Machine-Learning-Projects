import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Hyperparameters
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_EPOCHS = 3
BATCH_SIZE = 32


tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# Load and build vocab
train_iter = AG_NEWS(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
VOCAB_SIZE = len(vocab)

# Label mapping
NUM_CLASSES = 4  # AG News has 4 classes


def text_pipeline(text):
    return vocab(tokenizer(text))

def label_pipeline(label):
    return label - 1  # Labels are 1-indexed, we shift to 0-indexed

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    return (
        torch.tensor(label_list, dtype=torch.int64).to(DEVICE),
        pad_sequence(text_list, batch_first=True).to(DEVICE),
    )


class SWEM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        embedded = self.embedding(x)                         # (B, L, E)
        embed_mean = embedded.mean(dim=1)                    # mean over sequence length (L)
        h = F.relu(self.fc1(embed_mean))                     # (B, H)
        out = self.fc2(h)                                    # (B, C)
        return out


# Load datasets
train_iter, test_iter = AG_NEWS()
train_list = list(train_iter)
test_list = list(test_iter)

# Create loaders
train_dataloader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# Instantiate model
model = SWEM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_acc, total_count = 0, 0
    for labels, texts in train_dataloader:
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_acc += (output.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

    print(f"Epoch {epoch + 1}: Train Accuracy: {total_acc / total_count:.4f}")


model.eval()
total_acc, total_count = 0, 0
with torch.no_grad():
    for labels, texts in test_dataloader:
        output = model(texts)
        total_acc += (output.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

print(f"Test Accuracy: {total_acc/total_count:.4f}")
