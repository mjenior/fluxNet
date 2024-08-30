#!/usr/bin/env python3

import numpy
import pandas
import pyyaml
import argparse

import torch
from sklearn.preprocessing import StandardScaler

import splitter

#--------------------------------------------------------------------------------------#

# User defined arguments
parser = argparse.ArgumentParser()
parser.add_argument("--trainingfile", required=True)
parser.add_argument("--targetfile", required=True)
parser.add_argument("--testfile", required=True)
parser.add_argument("-r", "--ratio", default=0.2)
parser.add_argument("-e", "--epochs", default=100)
parser.add_argument("-l", "--layers", default=4)
parser.add_argument("-h", "--heads", default=3)
parser.add_argument("-s", "--size", default=500)
parser.add_argument("-n", "--learn_rate", default=0.001)
parser.add_argument("-m", "--momentum", default=0.9)
args = parser.parse_args()

numpy.random.seed(seed=123)

#--------------------------------------------------------------------------------------#

class TransformerNet(torch.nn.Module):
    def __init__(self, feature_size, num_layers, num_heads):
        super(TransformerNet, self).__init__()
        self.src_mask = None
        self.feature_size = feature_size        
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        self.decoder = torch.nn.Linear(feature_size, 10)

    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask

        x = self.transformer_encoder(x, self.src_mask)
        x = self.decoder(x)
        return x

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def transform_input_matrix(X: numpy.ndarray) -> numpy.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def train(net, criterion, optimizer, X_train, y_train, epochs=100):
    for epoch in range(epochs):
        inputs = torch.tensor(X_train).float()
        labels = torch.tensor(y_train).float()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Epoch: %d, Loss: %.3f' % (epoch + 1, loss.item()))

    print('Finished Training')


def predict(net, X_new):
    with torch.no_grad():
        inputs = torch.tensor(X_new)
        outputs = net(inputs)
        return outputs


#--------------------------------------------------------------------------------------#

if __name__ == "__main__":

    # Read and format data
    X_raw = pandas.read_csv(str(args.trainingfile))
    y = pandas.read_csv(str(args.targetfile))
    X = numpy.array(list(map(transform_input_matrix, X_raw)))
    
    # Define training and tests sets
    X_train, y_train, X_test, y_test = splitter.random(X, y, size=float(args.ratio))

    # Training the model
    net = TransformerNet(feature_size=int(args.size), num_layers=int(args.layers), num_heads=int(args.heads))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=float(args.learn_rate), momentum=float(args.momentum))
    train(net, criterion, optimizer, X_train, y_train, epochs=int(args.epochs))






    # Predicting on new data
    X_new_raw = pandas.read_csv(str(args.testfile))
    X_new = numpy.array(list(map(transform_input_matrix, X_new_raw))) # Apply transformation to each new sample
    predictions = predict(net, X_new)
    print(predictions)

