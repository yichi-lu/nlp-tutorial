#   https://github.com/IKMLab/skipgram/blob/master/Skip-Gram%20Practice_ANSWERS.ipynb

import sys
import os
import collections

import numpy as np
import pandas as pd
import nltk
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import dataset, dataloader
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

train_path = 'data/train-full.txt'
dev_path = 'data/dev-full.txt'
test_path = 'data/test-only-data.txt'
test_labels_path = 'data/test-labels.txt'

train = pd.read_csv(train_path, delimiter='\t')
dev = pd.read_csv(dev_path, delimiter='\t')
test = pd.read_csv(test_path, delimiter='\t')
test_labels = pd.read_csv(test_labels_path, delimiter='\t', header=None)
test_labels.columns = ['#id', 'correctLabelW0orW1']
test = pd.concat([test, test_labels])

print(f"train keys: {train.keys()}\ndev keys: {dev.keys()}\ntest keys: {test.keys()}\ntest label keys: {test_labels.keys()}, ")

""" Tokenization
We need to determine the set of all tokens in our dataset. We therefore need to separate each comment string into individual tokens,
then determine the unique set of those tokens. We focus on the tokenization step first.

We will use nltk for tokenization because it is lightweight. The nltk package defines a function called word_tokenize() that is useful for this.
"""
text_columns = ['claim', 'reason', 'debateInfo', 'debateTitle', 'warrant0', 'warrant1']
sents = []
for column in text_columns:
    sents += list(train[column].values)
    sents += list(dev[column].values)
    sents += list(test[column].values)
# Some values are nan - remove those
sents = [s for s in sents if not pd.isnull(s)]

token_set = set([])
for sent in sents:
    token_set.update(nltk.word_tokenize(sent))

assert len(token_set) == 6116

"""
Build the Vocab Dictionary
We need to associate a unique int index with every unique token, and provide a map for lookup. A high-level view of text processing is often:

receive text as input
tokenize that text to obtain tokens
map those tokens to integer indices
use those indices to lookup word vectors
use those vectors as input to a neural network.
We focus on (3) now.

To Do
Use the token_set to build a dict object called vocab that has every unique token in the token_set as an index and unique integers as values. Also add a token '<PAD>' to the vocab. We will need it when we deal with recurrent neural networks.

Tips:

The python zip() function can be used to bring two lists together - e.g. tokens and indices
The dict() constructor can take a zipped object as input, mapping the first position to index and second to value
"""
vocab = dict(zip(['<PAD>'] + list(token_set), range(len(token_set) + 1)))
assert len(vocab) == 6117
assert isinstance(list(vocab.keys())[0], str)
assert isinstance(list(vocab.values())[0], int)
assert '<PAD>' in vocab.keys()

rev_vocab = {v: k for k, v in vocab.items()}

print(f"size of our vocabrury: {len(vocab)}")

"""
Prepare the Training Pairs
We need to present two words at a time to the network to train our Skip-Gram: a center word and a context word. We therefore need to determine these pairs beforehand.

Before coding deep learning models it is necessary to first fully think through how we are going to present the data to the network. This will avoid having to make annoying changes that might follow from small details that are easy to overlook.

We know we are going to present two words at a time: a center word, and a context word. But how are we going to present them: as tokens, or as indices? These details matter when you code the forward pass of the network: if you try a word vector lookup on an embedding matrix with a string, you will see an error. We will use integer indices as it will be slightly faster than adding a dictionary lookup as well at training time.

Since finding the context tokens for all words over all instances in the dataset is not a generally useful skill, we do that for you.
"""
m = 5  # our context window size - you can experiment with this
contexts = {k: set() for k in range(1, len(vocab) + 1)}
for sent in sents:
    tokens = nltk.word_tokenize(sent)
    for i, center in enumerate(tokens):
        center = vocab[center]
        left_context = [vocab[t] for t in tokens[max(0, i - m):i - 1]]
        right_context = [vocab[t] for t in tokens[i + 1: min(len(tokens), i + m)]]
        contexts[center].update(left_context + right_context)

# the frequencies of our words for the negative sampling algorithm.
frequencies = collections.Counter()
for sent in sents:
    tokens = nltk.word_tokenize(sent)
    frequencies.update(tokens)

assert len(frequencies) == len(vocab) - 1  # we don't see <PAD> in the data
assert isinstance(list(frequencies.keys())[0], str)
assert isinstance(list(frequencies.values())[0], int)

# We'll add <PAD> to the frequencies to even the lengths of the probability distributions for later
frequencies['<PAD>'] = 0

# The last thing we need to prepare our data for training is define it as a set of pairs of words. Making a complete pass over this set constitutes one epoch of the data.
pairs = set([])
for center in contexts.keys():
    pairs.update(tuple(zip([center] * len(contexts[center]), list(contexts[center]))))
data = list(pairs)
print('Number of pairs in the dataset: %s' % len(data))
sys.exit(0)

"""
Negative Sampling
To perform negative sampling, we need a function that

Takes a token index as argument
Returns the number of negative samples we desire
Randomly chooses those samples according to
"""
class NegativeSampler:
    
    def __init__(self, vocab, frequencies, contexts, num_negs):
        """Create a new NegativeSampler.
        
        Args:
          vocab: Dictionary.
          frequencies: List of integers, the frequencies of each word,
            sorted in word index order.
          contexts: Dictionary.
          num_negs: Integer, how many to negatives to sample.
        """
        self.vocab = vocab
        self.n = len(vocab)
        self.contexts = contexts
        self.num_negs = num_negs
        self.distribution = self.p(list(frequencies.values()))
    
    def __call__(self, tok_ix):
        """Get negative samples.
        
        Args:
          tok_ix: Integer, the index of the center word.
        
        Returns:
          List of integers.
        """
        neg_samples = np.random.choice(
            self.n, 
            size=self.num_negs, 
            p=self.distribution)
        # make sure we haven't sampled center word or its context
        invalid = [-1, tok_ix] + list(self.contexts[tok_ix])
        for i, ix in enumerate(neg_samples):
            if ix in invalid:
                new_ix = -1
                while new_ix in invalid:
                    new_ix = np.random.choice(self.n, 
                                              size=1, 
                                              p=self.distribution)[0]
                neg_samples[i] = new_ix
        return [int(s) for s in neg_samples]
    
    def p(self, freqs):
        """Determine the probability distribution for negative sampling.
        
        Args:
          freqs: List of integers.
        
        Returns:
          numpy.ndarray.
        """
        ### Impelement Me ###
        freqs = np.array(freqs)
        return np.power(freqs, 3/4) / np.sum(np.power(freqs, 3/4))

class Collate:
    
    def __init__(self, neg_sampler):
        self.sampler = neg_sampler
    
    def __call__(self, pairs):
        ### Implement Me ###
        batch_size = len(pairs)
        centers = [x[0] for x in pairs]
        contexts = [x[1] for x in pairs]
        context_and_negs = []
        for i in range(batch_size):
            neg_samples = self.sampler(centers[i])
            context_and_negs.append([contexts[i]] + list(neg_samples))
        return centers, context_and_negs

def get_data_loader(data, batch_size, collate_fn):
    return dataloader.DataLoader(data, 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=1, 
                                 collate_fn=collate_fn)

class SkipGram(nn.Module):
    """SkipGram Model."""
    
    def __init__(self, vocab, emb_dim, num_negs, lr):
        """Create a new SkipGram.
        
        Args:
          vocab: Dictionary, our vocab dict with token keys and index values.
          emb_dim: Integer, the size of word embeddings.
          num_negs: Integer, the number of non-context words to sample.
          lr: Float, the learning rate for gradient descent.
        """
        super(SkipGram, self).__init__()
        self.vocab = vocab
        self.n = len(vocab)  # size of the vocab
        self.emb_dim = emb_dim
        self.num_negs = num_negs
        
        ### Implement Me: define V and U ###
        
        ### Implement Me: initialize V and U with unform distribution in [-0.01, 0.01] ###
        
        self.V = nn.Embedding(self.n, emb_dim)
        self.U = nn.Embedding(self.n, emb_dim)
        nn.init.uniform_(self.V.weight, a=-1., b=1.)
        nn.init.uniform_(self.U.weight, a=-1., b=1.)        
        
        # Adam is a good optimizer and will converge faster than SGD
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, centers, contexts_negs):
        """Compute the forward pass of the network.
        
        Args:
          centers: List of integers.
          contexts_negs: List of integers
        
        Returns:
          loss (torch.autograd.Variable).
        """
        centers = self.V(self.lookup_tensor(centers)).unsqueeze(1)
        contexts_negs = self.U(self.lookup_tensor(contexts_negs)).permute([0, 2, 1])
        logits = centers.matmul(contexts_negs).squeeze(1)
        predictions = self.softmax(logits)
        targets = self.targets(centers.shape[0])
        loss = self.loss(predictions, targets)
        return loss
    
    def lookup_tensor(self, indices):
        """Lookup embeddings given indices.
        
        Args:
          embedding: nn.Parameter, an embedding matrix.
          indices: List of integers, the indices to lookup.
        
        Returns:
          torch.autograd.Variable of shape [len(indices), emb_dim]. A matrix 
            with horizontally stacked word vectors.
        """
        if torch.cuda.is_available():
            return Variable(torch.LongTensor(indices),
                            requires_grad=False).cuda()
        else:
            return Variable(torch.LongTensor(indices),
                            requires_grad=False)

    def loss(self, predictions, targets):
        """Cross-Entropy Loss.
        
        Args:
          predictions: torch.autograd.Variable (float) of shape (batch_size, num_negs + 1).
          targets: torch.autograd.Variable (long) of shape (batch_size, num_negs + 1).
        """
        return -1 * torch.sum(targets * torch.log(predictions))
        
    def optimize(self, loss):
        """Optimization step.
        
        Args:
          loss: Scalar.
        """
        # Remove any previous gradient from our tensors before calculating again.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        
    
    def softmax(self, logits):
        """Softmax function.
        
        Args:
          logits: torch.autograd.Variable of shape (batch_size, num_negs + 1)
        
        Returns:
          torch.autograd.Variable of shape (batch_size, num_negs + 1).
        """
        return torch.exp(logits) / torch.sum(torch.exp(logits))
    
    def targets(self, batch_size):
        """Get the conventional targets for the batch.
        
        Args:
          batch_size: Integer.
        
        Returns:
          torch.autograd.Variable (float) of shape (batch_size, num_negs + 1.
            the first column is ones the rest are zeros.
        """
        if torch.cuda.is_available():
            targets = Variable(torch.zeros((batch_size, self.num_negs + 1)), requires_grad=False).cuda()
        else:
            targets = Variable(torch.zeros((batch_size, self.num_negs + 1)), requires_grad=False)
        targets[:, 0] = 1
        return targets

# Hyperparameters (play with these yourself)
max_epochs = 5
emb_dim = 10
num_negs = 5
lr = 0.01
batch_size = 16

sampler = NegativeSampler(vocab, frequencies, contexts, num_negs)
collate = Collate(sampler)
data_loader = get_data_loader(list(pairs), batch_size, collate_fn=collate)
skipgram = SkipGram(vocab, emb_dim, num_negs, lr)
if torch.cuda.is_available():
    skipgram.cuda()

epoch = 0
global_step = 0
cum_loss = 0.
while epoch < max_epochs:
    epoch += 1
    print('Epoch %s' % epoch)
    for step, batch in enumerate(data_loader):
        global_step  += 1
        loss = skipgram.forward(*batch)
        skipgram.optimize(loss)
        loss = loss.data.cpu().numpy()
        cum_loss += loss
        if step % 1000 == 0:
            print('Step %s\t\tLoss %8.4f' % (step, cum_loss / (global_step * batch_size)))

choices = ['man', 'woman', 'queen', 'king', 
           'Jewish', 'massacre', 'holocaust', 
           'Globalization', 'Malthusian', 'Privatization', 'immigration', 'Economics', 
           'selfish', 'selfless' ,
           'epidemic', 'harming', 'combat']
choice_ixs = [vocab[c] for c in choices]
random_ixs = [int(x) for x in np.random.choice(range(len(vocab)), 10)]
tok_ixs = choice_ixs + random_ixs
lookup_ixs = torch.LongTensor(tok_ixs)
embeddings = skipgram.V.weight[lookup_ixs].data.cpu().numpy()
tsne = TSNE(n_components=2, perplexity=2)
X_tsne = tsne.fit_transform(embeddings)
df = pd.DataFrame(X_tsne, index={rev_vocab[t]: t for t in tok_ixs}, columns=['x', 'y'])
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])
for word, pos in df.iterrows():
    ax.annotate(word, pos)

# plt.show()
plt.savefig("skipgramplot.pdf")

