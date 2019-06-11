import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import print_params, assign_params

class DocFeatureExtractor(nn.Module):
    def __init__(self, num_class, vocab_size, embedding_size=300, 
            num_pooling_features=1, kernels = [3,4,5], channels = 128, 
            use_batchnorm=True, embedding_init = None, dropout=0.5, static=True, padding_value=0):
        super(DocFeatureExtractor, self).__init__()
        assign_params(self, locals())

        self.embed_table = nn.Embedding(vocab_size, embedding_size, padding_idx = self.padding_value)
        if embedding_init is not None:
            self.embed_table.weight.data.copy_(torch.from_numpy(embedding_init))
            self.embed_table.weight.requires_grad = self.static

        # 
        if self.dropout is not None:
            self.embedding_dropout = nn.Dropout(self.dropout)
        ml = []
        for kernel_size in kernels:
            conv_layer = []
            conv1d = nn.Conv1d(self.embedding_size, self.channels, kernel_size, stride = 1, bias=True)
            conv_layer.append(conv1d)
            if self.use_batchnorm:
                bn = nn.BatchNorm1d(self.channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
                conv_layer.append(bn)
            apool1d  = nn.AdaptiveMaxPool1d(self.num_pooling_features)
            conv_layer += [nn.ReLU(), apool1d]
            ml.append(nn.Sequential(*conv_layer))
        #
        self.convList = nn.ModuleList(ml)
        fc1_input = self.channels*self.num_pooling_features*len(self.kernels)
        self.features_dim = fc1_input

    def forward(self, x):
        emb = self.embed_table(x) # e.g. (10,500,300)
        if self.dropout is not None:
            emb = self.embedding_dropout(emb).transpose(1,2) #(bs, embed, seq)
        else:
            emb = emb.transpose(1,2) # e.g. (10,300,500)
        # if emb.size()[-1] < max(self.kernels):
        #     pad_size = max(self.kernels) - emb.size()[-1] + 1
        #     emb = F.pad(emb,(0,pad_size), "constant", 0)
        features = [conv(emb) for conv in self.convList] # e.g. [(10,128,1)*3]
        features = torch.cat(features,dim = 2)
        features = features.view(len(features),-1) # e.g. (10,`18*3`)
        return features

class MultiChannelCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, kernel_sizes, num_kernels, num_class, 
                num_pooling_features, hidden_size, embed_vectors=None, 
                static=False, dropout=0.5, use_batchnorm=True, threshold=0.5,
                padding_value=0, max_length=500):
        '''
        Parameters:
            kernel_sizes: size of kernel for each channel
            num_kernels: number of kernels for each channel in hidden layers
        '''
        super(MultiChannelCNN, self).__init__()
        print_params(locals(), ['class_weights', 'embed_vectors'])
        self.padding_type = "FIXED"
        assign_params(self, locals())

        self.feature_extractor = DocFeatureExtractor(num_class, vocab_size, embed_dim, 
                                    num_pooling_features, kernel_sizes, num_kernels, 
                                    use_batchnorm=use_batchnorm, embedding_init=embed_vectors, 
                                    dropout=self.dropout, static=self.static, padding_value=self.padding_value)
        
        fc1 = []
        fc = nn.Linear(self.feature_extractor.features_dim, self.hidden_size, bias=True)
        fc1.append(fc)
        if self.use_batchnorm:
            bn = nn.BatchNorm1d(self.hidden_size, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
            fc1.append(bn)
        fc1.append(nn.ReLU())
        if self.dropout is not None:
            fc1.append(nn.Dropout(self.dropout))
        self.fc1 = nn.Sequential(*fc1)
        self.classifier = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.fc1(features) # e.g (10,512)
        logit = self.classifier(features)
        logit = torch.softmax(logit, dim=1)
        _, predictions = torch.max(logit, dim=1)
        return logit, predictions


class XMLCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, kernel_sizes=[3,4,5], num_filters=100, 
                num_pooling_features=1, hidden_size=256, num_classes =10, pretrained_embeddings=None, 
                static=False, use_cuda=True, sentence_len=300):
        super(XMLCNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = not static
        self.num_pooling_features = num_pooling_features

        conv_blocks = []
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            # maxpool_kernel_size = sentence_len - kernel_size + 1
            conv1d = nn.Conv1d(in_channels = embedding_dim, out_channels = num_filters, kernel_size = kernel_size, stride = 1)
            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(output_size=self.num_pooling_features)
            )

            conv_blocks.append(component)

        self.conv_blocks = nn.ModuleList(conv_blocks)   # ModuleList is needed for registering parameters in conv_blocks

        self.fc = nn.Sequential(
                    nn.Linear(num_filters*len(kernel_sizes), self.hidden_size),
                    nn.ReLU()
                )
        self.clf = nn.Linear(self.hidden_size, num_classes)
        if use_cuda: 
            self.embedding = self.embedding.cuda()
            self.cuda()

    def forward(self, x):       # x: (batch, sentence_len)
        x = self.embedding(x)   # embedded x: (batch, sentence_len, embedding_dim)
        x = x.transpose(1,2)  # needs to convert x to (batch, embedding_dim, sentence_len)
        x_list= [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.clf(out)
        _, predictions = torch.max(out, dim=1)
        return out, predictions 

class YoonKimCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, kernel_sizes=[3,4,5], num_filters=100, num_pooling_features=1, 
                num_classes =10, pretrained_embeddings=None, static=False, use_cuda=True, 
                sentence_len=300):
        super(YoonKimCNN, self).__init__()
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = not static
        self.num_pooling_features = num_pooling_features

        conv_blocks = []
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size = sentence_len - kernel_size + 1
            conv1d = nn.Conv1d(in_channels = embedding_dim, out_channels = num_filters, kernel_size = kernel_size, stride = 1)
            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size)
            )

            conv_blocks.append(component)

        self.conv_blocks = nn.ModuleList(conv_blocks)   # ModuleList is needed for registering parameters in conv_blocks

        self.fc = nn.Linear(num_filters*len(kernel_sizes), num_classes)
        if use_cuda: 
            self.embedding = self.embedding.cuda()
            self.cuda()

    def forward(self, x):       # x: (batch, sentence_len)
        x = self.embedding(x)   # embedded x: (batch, sentence_len, embedding_dim)
        x = x.transpose(1,2)  # needs to convert x to (batch, embedding_dim, sentence_len)
        x_list= [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = F.softmax(self.fc(out), dim=1)
        _, predictions = torch.max(out, dim=1)
        return out, predictions 