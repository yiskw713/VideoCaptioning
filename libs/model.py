import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, embed_size):
        """
            take as input video features from 3dcnn which have spatial and temporal dimensions
            return flattened features
        """
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm3d(in_channels, momentum=0.1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(in_channels, embed_size)
        self.bn2 = nn.BatchNorm1d(embed_size, momentum=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """return flattened features"""
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.squeeze()
        x = self.linear(x)
        x = self.bn2(x)
        return x


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param.data)
                    else:
                        nn.init.zeros_(param.data)
            elif isinstance(m, nn.LSTMCell):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param.data)
                    else:
                        nn.init.zeros_(param.data)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight.data)

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            # hiddens: (batch_size, 1, hidden_size)
            hiddens, states = self.lstm(inputs, states)
            # outputs:  (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))
            # predicted: (batch_size)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            # inputs: (batch_size, embed_size)
            inputs = self.embed(predicted)
            # inputs: (batch_size, 1, embed_size)
            inputs = inputs.unsqueeze(1)
        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
