#!/usr/bin/env python
import os
import math
import string
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

class LangModel(nn.Module):
    def __init__(self, vocab_size, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
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
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    # lang -> list[unigram, NLM, int2token, token2int]
    model = {}

    @classmethod
    def create_seq(cls, text, seq_len=5):
        sequences = []
        if len(text) > seq_len:
            for i in range(seq_len, len(text)):
                seq = list(text)[i-seq_len:i+1]
                sequences.append("".join(seq))
            return sequences
        else:
            return []

    # gets the integer sequence given token2int
    @classmethod
    def get_integer_seq(cls, seq, token2int):
        return [token2int[c] for c in seq]

    # data is the lines of the training data
    # returns an int array for the input and output of the data
    # returns int2token and token2int
    @classmethod
    def getData(cls, data):
        seqs = [MyModel.create_seq(i) for i in data]
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
        x_int = [MyModel.get_integer_seq(i, token2int) for i in x]
        y_int = [MyModel.get_integer_seq(i, token2int) for i in y]

        # convert lists to numpy arrays
        x_int = np.array(x_int)
        y_int = np.array(y_int)
        return x_int, y_int, int2token, token2int, vocab_size

    @classmethod
    def load_training_data(cls):
        # trainPath = r'../shortTranslations/AllTrain'
        trainPath = r'data/shortTrain'
        files = os.listdir(trainPath)
        out = {}
        for f in files:
            lang_code = f[:f.find('train')]
            f_ = open(os.path.join(trainPath, f), "r")
            out[lang_code] = [f_.read().split("\n")]
        for lang in out:
            x_int, y_int, int2token, token2int, vocab_size = MyModel.getData(out[lang][0])
            out[lang].append(x_int)
            out[lang].append(y_int)
            out[lang].append(int2token)
            out[lang].append(token2int)
            out[lang].append(vocab_size)
        return out

    @classmethod
    def load_test_data(cls, fname):
        f_ = open(fname, "r")
        return f_.read().split("\n")

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    @classmethod
    def get_batches(cls, arr_x, arr_y, batch_size):
        # iterate through the arrays
        prv = 0
        for n in range(batch_size, arr_x.shape[0], batch_size):
            x = arr_x[prv:n,:]
            y = arr_y[prv:n,:]
            prv = n
            yield x, y

    @classmethod
    def train_single(cls, net, device, x_int, y_int, epochs=10, batch_size=32, lr=0.001, clip=1):
        
        # optimizer
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        # loss
        criterion = nn.CrossEntropyLoss()

        net.to(device)
        counter = 0
        net.train()
        for e in tqdm(range(epochs)):
            # initialize hidden state
            h = net.init_hidden(batch_size, device)
            for x, y in MyModel.get_batches(x_int, y_int, batch_size):
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


    def run_train(self, data, work_dir, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        for lang in data:
            self.model[lang] = []
            curUnigram = {}
            total_counts = 0
            unigramData = data[lang][0]
            x_int = data[lang][1]
            y_int = data[lang][2]
            int2token = data[lang][3]
            token2int = data[lang][4]
            vocab_size = data[lang][5]
            # train the unigram model
            for line in unigramData:
                for c in line:
                    count = curUnigram.get(c, 0)
                    curUnigram[c] = count + 1
                    total_counts += 1

            for c in curUnigram:
                curUnigram[c] /= total_counts
            # train and create the NLM
            self.model[lang].append(curUnigram)
            net = LangModel(vocab_size)
            MyModel.train_single(net, device, x_int, y_int, epochs=20, batch_size=1800)
            self.model[lang].append(net)
            self.model[lang].append(int2token)
            self.model[lang].append(token2int)


    # predict next token
    @classmethod
    def predict(cls, net, tkn, device, int2token, token2int, h=None):
            
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
    @classmethod
    def sample(cls, net, device, int2token, token2int, prime):
        net.to(device)
        net.eval()
        h = net.init_hidden(1, device)
        # predict next token
        for t in prime:
            print(t)
            token, probs, h = MyModel.predict(net, t, device, int2token, token2int, h)
        return token, probs

    def run_pred(self, data, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        preds = []
        for inp in data:

            # create a list of languages that it could be, creating a score for each language
            langScores = {}
            langProbabilities = {}
            maxScore = 0
            z = 0
            for lang in self.model:
                score = 0
                unigram = self.model[lang][0]
                for char in inp:
                    curScore = unigram.get(char, 0)
                    score += curScore

                if score > maxScore:
                    maxScore = score

                langScores[lang] = score
                z += math.exp(score)

            for lang in self.model:
                norm = 0
                if maxScore != 0:
                    norm = langScores[lang] / maxScore
                langScores[lang] = norm

            for lang in self.model:
                langProbabilities[lang] = math.exp(langScores[lang]) / z
            charGuesses = dict()
            # wont work if there is not tokens so we need to make sure 
            for lang in langScores:
                if langScores[lang] > 0:
                    net = self.model[lang][1]
                    int2token = self.model[lang][2]
                    token2int = self.model[lang][3]
                    letters, scores = MyModel.sample(net, device, int2token, token2int, inp)
                    for i in range(len(letters)):
                        curLetter = letters[i]
                        oldScore = charGuesses.get(curLetter, 0)
                        thisScore = scores[i]
                        thisScore *= langProbabilities[lang]
                        charGuesses[letters[i]] = oldScore + thisScore
            top_guesses = sorted(charGuesses, key=charGuesses.get, reverse=True)[:3]
            preds.append(''.join(top_guesses))

        return preds

    def save(self, work_dir):
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            savedModel = pickle.load(f)
        model = MyModel()
        model.model = savedModel
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
