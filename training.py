# Import the necessary libraries.
import torch
import accelerate
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from nltk import word_tokenize
from sklearn.metrics import f1_score

# Set the seeds for reproducibility.
SEED = 4321
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Define the fields associated with the dataset.
TEXT = data.Field(tokenize = word_tokenize) # custom tokenizer is used
LABEL = data.LabelField(dtype = torch.float)

# Build a vocabulary.
TEXT.build_vocab(train_data, max_size = 25000, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)

# Create a data loader.
train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data), batch_size = 64, device = device)


# Define a custom loss function.
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        
    def forward(self, prediction, target):
        loss = (prediction - target)**2 # sample loss function
        return torch.mean(loss)

# Define a basic feed-forward network.
class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForward, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x)


# LArger network needed...




# Setting an input and output dimension
input_dim = len(TEXT.vocab)
output_dim = 1  # for binary classification

model = FeedForward(input_dim, output_dim)

# Initialize weights from normal distribution.
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)
        
model.apply(init_weights)

# Initialize optimizer and loss function.
optimizer = optim.Adam(model.parameters())
criterion = CustomLoss()

# Training function.
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Iterate over your epochs.
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_iterator, optimizer, criterion)
    print(f'Epoch: {epoch+1}, Loss: {train_loss:.3f}')



# Save model
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'model.pth')



# Load model
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()
