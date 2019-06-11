import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F

class GatedBlock(nn.Module):
    def __init__(self, in_channels=10, out_channels=100, kernel_size=3, dropout=0.5):
        super(GatedBlock, self).__init__()
        self.cnn_input = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1)
        self.cnn_gate = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, embed_size, len)
        inp = self.cnn_input(x) # (N, out_channels, len)
        gate = self.cnn_gate(x) # (N, out_channels, len)
        out = inp * gate # (N, out_channels, len)
        return out

class GatedCNN(nn.Module):
    def __init__(self, vocab_size=10, embed_size=300, num_channels=[100,200,300], kernel_size=3, max_k=3):
        super(GatedCNN, self).__init__()
        self.max_k = max_k
        self.embed_size = embed_size
        num_layers = len(num_channels)
        layers = []
        in_channel = embed_size
        for i in range(num_layers):
            out_channel = num_channels[i]
            layers.append(GatedBlock(in_channel, out_channel, kernel_size, 0.5))
            in_channel = out_channel
        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1] * self.max_k, self.embed_size)

    def forward(self, x):
        # x: (bs, embedding_size, seq_len)
        bs = x.size(0)
        x = self.cnn(x)
        # --> x: (bs, out_channel, seq_len)
        x = self.kmax_pooling(x, 2, k=self.max_k).view(bs, -1)
        # --> x: (bs, out_channel * k)
        x = self.fc(x)
        # --> x: (bs, embedding_size)
        return x

    @staticmethod
    def kmax_pooling(x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)



class GatedCNNWrapper(nn.Module):
    def __init__(self, kernel_size=3, num_channels=[100, 200], num_classes =10, pretrained_embeddings=None, 
                n_pos=10, n_neg=50, static=False, use_cuda=True):
        super(GatedCNNWrapper, self).__init__()
        
        if not isinstance(pretrained_embeddings, np.ndarray):
            embedding = np.array(pretrained_embeddings)
        vocab_size, embedding_size = pretrained_embeddings.shape
        self._embedding = nn.Embedding(vocab_size, embedding_size)
        self._embedding.weight = nn.Parameter(torch.from_numpy(pretrained_embeddings).float())
        self._embedding.weight.requires_grad = not static

        self.gated_cnn = GatedCNN(vocab_size=vocab_size, embed_size=embedding_size, 
                                num_channels=num_channels, kernel_size=kernel_size, max_k=2)
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # x: Tensor (batch_size, seq_len)
        # y: Tensor (batch_size, pos+neg)

        x = self._embedding(x).permute(0, 2, 1)  # (batch_size, embedding_size, seq_len)
        # x = self.gated_cnn(x).unsqueeze(1)  # (batch_size, 1, embedding_size)
        
        # y = self._embedding(y).permute(0, 2, 1)  # (batch_size, embedding_size, pos+neg)
        # logits = torch.bmm(x, y).squeeze(1)  # (batch_size, 1, pos+neg)
        x = F.relu(self.gated_cnn(x))
        out = F.softmax(self.fc(x), dim=1)
        _, predictions = torch.max(out, dim=1)
        return out, predictions 