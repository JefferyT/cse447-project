#!/usr/bin/env python
import os
import string
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class LangModel(nn.Module):
    def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
        super().__init__()
        
        self.drop_prob = drop_prob
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lr = lr
        
        self.emb_layer = nn.Embedding(vocab_size, 200)

        ## define the LSTM
        self.lstm = nn.LSTM(200, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## define the fully-connected layer
        self.fc = nn.Linear(n_hidden, vocab_size)
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        embedded = self.emb_layer(x.long())
        
        ## Get the outputs and the new hidden state from the lstm
        lstm_output, hidden = self.lstm(embedded, hidden)
        
        ## pass through a dropout layer
        out = self.dropout(lstm_output)
        
        #out = out.contiguous().view(-1, self.n_hidden) 
        out = out.reshape(-1, self.n_hidden) 

        ## put "out" through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        ''' initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
        return hidden

class MyModel:

    n = 3
    # lang -> ngrams -> list[unigram, bigram ...], each model is a dictionary {prefix:probablity}
    lang_to_ngrams = {}

    def create_seq(text, seq_len=5):
        sequences = []
        if len(text) > seq_len:
            for i in range(seq_len, len(text)):
                seq = list(text)[i-seq_len:i+1]
                sequences.append("".join(seq))
            return sequences
        else:
            return []

    def get_integer_seq(seq):
        return [token2int[c] for c in seq]

    def getData(data):

        seqs = [create_seq(i) for i in data]
        seqs = sum(seqs, [])
        x = []
        y = []

        for s in seqs:
            x.append("".join(list(s)[:-1]))
            y.append("".join(list(s)[1:]))
        
        int2token = {}
        cnt = 0
        for c in set(list("".join(data))):
            int2token[cnt] = c
            cnt += 1
        token2int = {t: i for i, t in int2token.items()}
        vocab_size = len(int2token)
        # convert char sequences to integer sequences
        x_int = [get_integer_seq(i) for i in x]
        y_int = [get_integer_seq(i) for i in y]

        # convert lists to numpy arrays
        x_int = np.array(x_int)
        y_int = np.array(y_int)
        return x_int, y_int

    @classmethod
    def load_training_data(cls):
        # trainPath = r'../shortTranslations/AllTrain'
        trainPath = r'data/train'
        files = os.listdir(trainPath)
        out = {}
        outXY = {}
        for f in files:
            lang_code = f[:f.find('train')]
            f_ = open(os.path.join(trainPath, f), "r", encoding='utf-16')
            out[lang_code] = f_.read().split("\n")
        for lang in out:
            x_int, y_int = getData(out[lang])
            outXY[lang] = tuple(x_int, y_int)
        return out, outXY

    @classmethod
    def load_test_data(cls, fname):
        f_ = open(fname, "r")
        return f_.read().split("\n")

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))



    def get_batches(arr_x, arr_y, batch_size):
        # iterate through the arrays
        prv = 0
        for n in range(batch_size, arr_x.shape[0], batch_size):
            x = arr_x[prv:n,:]
            y = arr_y[prv:n,:]
            prv = n
            yield x, y


    def train_single(net, device, x_int, y_int, epochs=10, batch_size=32, lr=0.001, clip=1, print_every=32):
        
        # optimizer
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        
        # loss
        criterion = nn.CrossEntropyLoss()
        
        # push model to GPU
        net.to(device)
        
        counter = 0

        net.train()

        for e in tqdm(range(epochs)):

            # initialize hidden state
            h = net.init_hidden(batch_size, device)
            
            for x, y in get_batches(x_int, y_int, batch_size):
                counter += 1
                
                # convert numpy arrays to PyTorch arrays
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                
                # push tensors to GPU
                inputs, targets = inputs.to(device), targets.to(device)

                # detach hidden states
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                net.zero_grad()
                
                # get the output from the model
                output, h = net(inputs, h)
                
                # calculate the loss and perform backprop
                loss = criterion(output, targets.view(-1).long())

                # back-propagate error
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(net.parameters(), clip)

                # update weigths
                opt.step()

    def run_train(self, data, work_dir):
        


    # predict next token
    def predict(net, tkn, device, h=None):
            
        # tensor inputs
        x = np.array([[token2int[tkn]]])
        inputs = torch.from_numpy(x)

        # push to GPU
        inputs = inputs.to(device)

        # detach hidden state from history
        h = tuple([each.data for each in h])

        # get the output of the model
        out, h = net(inputs, h)

        # get the token probabilities
        p = F.softmax(out, dim=1).data

        p = p.cpu()
        

        p = p.numpy()
        p = p.reshape(p.shape[1],)

        # get indices of top 3 values
        top_n_idx = p.argsort()[-3:][::-1]
        # return the encoded value of the predicted char and the hidden state
        return [int2token[i] for i in top_n_idx], [p[i] for i in top_n_idx], h

    # function to generate text
    def sample(net, size, device, prime='it is'):
        net.to(device)
        net.eval()
        h = net.init_hidden(1, device)
        # predict next token
        for t in prime:
            token, probs, h = predict(net, t, device, h)
        return token, probs

    def run_pred(self, data):
        # your code here
        for inp in data:

            # create a list of languages that it could be, creating a score for each language
            langScores = {}
            maxScore = 0
            for lang in self.lang_to_ngrams:
                score = 0
                unigram = self.lang_to_ngrams[lang][0]
                for char in inp:
                    curScore = unigram.get(char, 0)
                    score += curScore
                if score > maxScore:
                    maxScore = score
                langScores[lang] = score
            for lang in self.lang_to_ngrams:
                norm = 0
                if maxScore != 0:
                    norm = langScores[lang] / maxScore
                langScores[lang] = norm
            
            inp = "¢¢" + inp  # start padding
            # prefix = inp[-2:]  # last 2 chars in input
            prefix = inp[-(self.n-1):]  # last n-1 chars in input

            # dict of lang code -> list of dictionaries
            # unigram char to prob
            # bigram, trigram -> dict is prefix in tuple to any suffix probability

            # prefix = "lo"
            top_guesses = dict()
            for lang in self.lang_to_ngrams:
                lang_models = self.lang_to_ngrams[lang]

                unigram_model = lang_models[0]
                bigram_model = lang_models[1]
                trigram_model = lang_models[2]

                token_to_prob = dict()
                for token in unigram_model:
                    # Probability
                    unigram_probability = unigram_model[token]

                    # Check to ensure tokens in models
                    bigram_probability = 0.0
                    if prefix[1] in bigram_model and token in bigram_model[prefix[1]]:
                        bigram_probability += bigram_model[prefix[1]][token]

                    trigram_probability = 0.0
                    if (prefix[0] + prefix[1]) in trigram_model and token in trigram_model[(prefix[0] + prefix[1])]:
                        trigram_probability += trigram_model[(prefix[0] + prefix[1])][token]

                    interpolated_probability = l1 * unigram_probability + l2 * bigram_probability + l3 * trigram_probability
                    interpolated_probability *= langScores[lang]
                    token_to_prob[token] = interpolated_probability

                # Get current three chars with highest probability
                highest = sorted(token_to_prob, key=token_to_prob.get, reverse=True)[:3]

                for token in highest:
                    oldScore = top_guesses.get(token, 0) + token_to_prob[token]
                    top_guesses[token] = oldScore
                    # if token not in top_guesses or (token in top_guesses and token_to_prob[token] > top_guesses[token]):
                    #     top_guesses[token] = token_to_prob[token]

            # Get overall three chars with highest prob
            top_guesses = sorted(token_to_prob, key=token_to_prob.get, reverse=True)[:3]

            preds.append(''.join(top_guesses))

        return preds

    def save(self, work_dir):
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(self.lang_to_ngrams, f)

    @ classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            lang_to_ngrams = pickle.load(f)
        model = MyModel()
        model.lang_to_ngrams = lang_to_ngrams
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=(
        'train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data',
                        default='example/input.txt')
    parser.add_argument(
        '--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(
            len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
