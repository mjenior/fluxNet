import torch
import torch.nn.functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()

        # Define hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, len(x)))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar




# Load checkpoint
checkpoint = torch.load('model.pth', map_location='cpu')
vae_model = VAE(input_dim, hidden_dim, z_dim)
vae_model.load_state_dict(checkpoint['model_state_dict'])

# Create data sample
sample = Variable(torch.randn(64, z_dim))

# Generate a batch of data from this sample
sample = vae_model.decode(sample).cpu()

# Distribution over the data
data_distribution = F.softmax(sample, dim=0)



# Parameter(s) for random generation
num_samples = 1000

# In case of VAE, we can just sample from normal distribution and then decode
z = torch.randn(num_samples, z_dim)
sampled_data = vae_model.decode(z).detach().numpy()  # Assuming VAE model is in CPU mode

# then I guess take the median and std of each reaction to get prediction and confidence for each reaction?
