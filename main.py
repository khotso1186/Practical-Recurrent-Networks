import torch
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm

class Dictionary(object):

    def __init__(self):
        
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        
        if word not in self.word2idx:
            
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __len__(self):
        
        return len(self.word2idx)

class TextProcess(object):
    
    def __init__(self):
        
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        
        with open(path, 'r') as f:
            
            tokens = 0
            
            for line in f:
                
                words = line.split() + ['<eos>']
                tokens += len(words)
                
                for word in words: 
                    self.dictionary.add_word(word)  
        
        #Create a 1-D tensor that contains the index of all the words in the file
        rep_tensor = torch.LongTensor(tokens)
        index = 0
        
        with open(path, 'r') as f:
            
            for line in f:
                
                words = line.split() + ['<eos>']
                
                for word in words:
                    
                    rep_tensor[index] = self.dictionary.word2idx[word]
                    index += 1
        #Find out how many batches we need            
        num_batches = rep_tensor.shape[0] // batch_size     
        #Remove the remainder (Filter out the ones that don't fit)
        rep_tensor = rep_tensor[:num_batches*batch_size]
        # return (batch_size,num_batches)
        rep_tensor = rep_tensor.view(batch_size, -1)
        
        return rep_tensor

embed_size = 128    #Input features to the LSTM
hidden_size = 1024  #Number of LSTM units
num_layers = 1
num_epochs = 20
batch_size = 20
timesteps = 30
learning_rate = 0.002

corpus = TextProcess()

rep_tensor = corpus.get_data('/home/khotso/Practical-Recurrent-Networks/data/alice.txt', batch_size)

vocab_size = len(corpus.dictionary)

num_batches = rep_tensor.shape[1] // timesteps

class TextGenerator(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        
        super(TextGenerator, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        
        # Perform Word Embedding 
        x = self.embed(x)
        #Reshape the input tensor
        #x = x.view(batch_size,timesteps,embed_size)
        out, (h, c) = self.lstm(x, h)
        # Reshape the output from (samples,timesteps,output_features) to a shape appropriate for the FC layer 
        # (batch_size*timesteps, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        # Decode hidden states of all time steps
        out = self.linear(out)
        
        return out, (h, c)

model = TextGenerator(vocab_size, embed_size, hidden_size, num_layers)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def detach(states):
    
    return [state.detach() for state in states] 

for epoch in range(num_epochs):
    
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size))

    for i in range(0, rep_tensor.size(1) - timesteps, timesteps):

        # Get mini-batch inputs and targets
        inputs = rep_tensor[:, i:i+timesteps]  
        targets = rep_tensor[:, (i+1):(i+1)+timesteps]
        
        outputs,_ = model(inputs, states)
        loss = loss_fn(outputs, targets.reshape(-1))

        model.zero_grad()
        loss.backward()
        
        #Perform Gradient Clipping. clip_value (float or int) is the maximum allowed value of the gradients 
        #The gradients are clipped in the range [-clip_value, clip_value]. This is to prevent the exploding gradient problem
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
              
        step = (i+1) // timesteps

        if step % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Test the model
with torch.no_grad():

    with open('/home/khotso/Practical-Recurrent-Networks/results/results.txt', 'w') as f:
        
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size), torch.zeros(num_layers, 1, hidden_size))
        
        # Select one word id randomly and convert it to shape (1,1)
        input = torch.randint(0,vocab_size, (1,)).long().unsqueeze(1)

        for i in range(500):
            
            output, _ = model(input, state)
            print(output.shape)
            # Sample a word id from the exponential of the output 
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()
            print(word_id)
            # Replace the input with sampled word id for the next time step
            input.fill_(word_id)

            # Write the results to file
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)
            
            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, 500, 'results.txt'))