import torch
from torch.utils.data import Dataset 
import cv2
import numpy as np 
import pandas as pd

import torch.nn as nn
from torchvision import transforms
import sys
import math
from LoadDataset import * # VideoDataset, VideoFolderPathToTensor
from config import *
from torch.nn import functional
import torch.utils.data as data
import numpy as np
import os
import requests
import time
from Model1 import *
import json
import csv

def return_one_hot(captions):
  for i in range(batch_size):
    for j in range(captions.shape[1]):
      label2one_hot = functional.one_hot(captions[i,j], num_classes=vocab_size)
      if j==0:
        one_sentence=label2one_hot
      else:
        one_sentence=torch.cat([one_sentence,label2one_hot],axis=0)
    if i==0:
      final_captions=one_sentence.unsqueeze(0)
    else:
      final_captions=torch.cat([final_captions,one_sentence.unsqueeze(0)],axis=0)
  return final_captions

def index_to_word(index, type_):
    relations = pd.read_json(location + "/{}".format(relationship_json), orient='index')
    relations['word'] = relations.index
    relations.index = relations[0]
    relations = relations.drop([0], axis=1)

    objects = pd.read_json(location + "/{}".format(object1_object2_json), orient='index')
    objects['word'] = objects.index
    objects.index = objects[0]
    objects = objects.drop([0], axis=1)
    if type_ == 'relation':
        word = relations[relations.index == index].iloc[-1].word
    else:
        word = objects[objects.index == index].iloc[-1].word
    return word

def caption(object1,relation,object2):
  sample_caption = []
  for i in range(29):
      sample_caption.append(word2idx['<pad>'])
  sample_caption.append(word2idx['<start>'])
  sample_caption.append(word2idx[index_to_word(object1, 'object')]) #Object1 is a tuple
  sample_caption.append(word2idx[index_to_word(relation, 'relation')]) #relation is a tuple
  sample_caption.append(word2idx[index_to_word(object2, 'object')]) #Object2 is a tuple
  sample_caption.append(word2idx['<end>'])
  sample_caption = torch.Tensor(sample_caption).long()
  return sample_caption

if __name__ == "__main__":
    with open(os.path.join(location, word2idx_json)) as f:
        word2idx = json.load(f)
    with open(os.path.join(location, idx2word_json)) as f:
        idx2word = json.load(f)
    vocab = pd.read_csv('{}/{}'.format(location,vocab_file))
    # The size of the vocabulary.
    vocab_size = vocab.iloc[-1,1]+1

    # Initialize the encoder and decoder.
    model = FirstModel(embed_size, hidden_size, vocab_size)

    # Move models to GPU if CUDA is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function.
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # TODO #3: Specify the learnable parameters of the model.
    params = list(model.parameters())

    # TODO #4: Define the optimizer.
    optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = torch.optim.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08)

    # Set the total number of training steps per epoch.
    #total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

    old_time = time.time()
    dataset = VideoDataset(
        os.path.join(location,annotated_file),transform = VideoFolderPathToTensor()
    )



    dataset = VideoDataset(
        os.path.join(location,annotated_file),transform = VideoFolderPathToTensor()
    )

    train, valid = torch.utils.data.random_split(dataset,[train_len,val_len])
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size)
    min_valid_loss = np.inf
    for epoch in range(1, num_epochs+1):
        total_step = math.ceil(trainloader.sampler.num_samples / trainloader.batch_sampler.batch_size)
        for i_step in range(1, total_step+1):

            # Obtain the batch.
            images, object1, relation, object2  = next(iter(trainloader))

            #For caption preprocessing we initialize an empty list and append an integer to mark the start of a caption. We use a special start and end word to mark the beginning and end of a caption.
            #We append integers to the list that correspond to each of the tokens in the caption. Finally, we convert the list of integers to a PyTorch tensor and cast it to long type.

            for i in range(len(object1)):
              if i==0:
                sample_caption=caption(object1[i],relation[i],object2[i]).unsqueeze(0)
                print(sample_caption)
              else:
                sample_caption=torch.cat((sample_caption,caption(object1[i],relation[i],object2[i]).unsqueeze(0)),axis=0)
            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions=sample_caption[:,29:]
            captions = captions.to(device)

            # Zero the gradients.
            model.zero_grad()

            # Pass the inputs through the CNN-RNN model.
            outputs = model(images, captions)
            compare_captions = sample_caption

            # Calculate the batch loss.
            loss = criterion(outputs.contiguous().view(-1, outputs.shape[-1]), compare_captions.view(-1))

            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))

            # Print training statistics (on same line).
            print('\r' + stats)

            # Print training statistics (on different line).
            if i_step % print_every == 0:
                print('\r' + stats)

        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer
        for images, object1, relation, object2 in validloader:

            for i in range(len(object1)):
              if i==0:
                sample_caption=caption(object1[i],relation[i],object2[i]).unsqueeze(0)
              else:
                sample_caption=torch.cat((sample_caption,caption(object1[i],relation[i],object2[i]).unsqueeze(0)),axis=0)
            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions=sample_caption[:,29:]
            captions = captions.to(device)

            target = model(data)
            compare_captions = sample_caption

            # Calculate the batch loss.
            valid_loss = criterion(outputs.contiguous().view(-1, outputs.shape[-1]), compare_captions.view(-1))
            #valid_loss = loss.item() * data.size(0)

            # Get training statistics.
            stats = 'Epoch [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
            epoch, num_epochs, valid_loss.item(), np.exp(valid_loss.item()))

            # Print training statistics (on same line).
            print('\r' + stats)

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')