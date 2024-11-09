# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:32:40 2024

@author: nhmille
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from collections import Counter
from math import ceil

from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from transformers import get_linear_schedule_with_warmup


def readData(filename):  
    '''
    Reads in JSON datafile.
    Converts to a dict of contexts, questions, answer positions.
    '''
    # Load the JSON file
    with open(os.path.join(os.getcwd(), filename)) as f:
        squad = json.load(f)

    contexts = []
    questions = []
    answers = []

    for group in squad['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    
                    # Calculate the end position of the answer
                    answer_data = {'answer_start': answer['answer_start'],
                                   'answer_end': answer['answer_start'] + len(answer['text']),
                                   'text': answer['text']}

                    # Skip blank answers -> Does not get rid of Nonetype answers???
                    if answer_data['answer_start'] == 0 and answer_data['answer_end'] == 0:
                        continue
                    
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer_data)

    # # Return data as dict of lists
    data ={'contexts':contexts,
            'questions':questions,
            'answers':answers,}
    
    return data
    # return contexts, questions, answers

def updateAnswerGroundTruth(tokenized_data, answers, tokenizer):
    '''
    Tokenization adds special characters -> Update answer location. 
    Also adjusts end position -> first char should be after answer.
    '''
    start_positions = []
    end_positions = []
    
    # Iterate over answers
    for i in range(len(answers)):
        # char_to_taken tells us what token location corresponds with the char at answer_start/end
        start_pos = tokenized_data.char_to_token(i, answers[i]['answer_start'])
        
        # We calculated the end position as immediately following the last answer chad -> Update to final char in the answer
        end_char_pos = answers[i]['answer_end'] - 1
        if end_char_pos < 0:
            raise ValueError(f"Negative end position encountered for index {i}: "
                     f"answer_start={answers[i]['answer_start']}, "
                     f"answer_end={answers[i]['answer_end']}")
        end_pos = tokenized_data.char_to_token(i, answers[i]['answer_end'] - 1)
    
        # If start is None -> context is truncated -> set to max length
        if start_pos is None:
            start_pos = tokenizer.model_max_length-1
        
        # If end is None -> context is truncated -> set to max length
        if end_pos is None:
            end_pos = tokenizer.model_max_length-1

        if start_pos is None or end_pos is None:
            print(f"Debug: Encountered NoneType in positions at index {i}")
            print(f"Context: {tokenized_data['input_ids'][i]}")
            print(f"Answer Text: '{answers[i]['text']}'")
            print(f"Answer Start: {answers[i]['answer_start']}, Answer End: {answers[i]['answer_end']}")
            print(f"Start Pos: {start_pos}, End Pos: {end_pos}")
            
        start_positions.append(start_pos)
        end_positions.append(end_pos)
            
        tokenized_data.update({'start_positions': start_positions, 'end_positions': end_positions})


def checkForNones(tokenized_data):
    """
    Check each field in tokenized_data for None values and report any found.
    """
    for key, values in tokenized_data.items():
        none_indices = [i for i, val in enumerate(values) if val is None]
        if none_indices:
            print(f"Field '{key}' has None values at indices: {none_indices}")
            print(f"Total None entries in '{key}': {len(none_indices)}")
        else:
            print(f"Field '{key}' has no None values.")



# Class to facilitate training
class fineTune(nn.Module):
    def __init__(self, model, epochs=1, lr=5e-5):
        super(fineTune, self).__init__()
        self.lr=lr
        self.epochs=epochs
        self.model = model    
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)     

    def train(self, train_dataset, test_loader, tokenizer):
        avg_losses_per_epoch = []
        f1_epoch_scores = []
        
        steps_per_epoch = int(0.10 * len(train_dataset)/32)
        total_steps = self.epochs * steps_per_epoch
        scheduler = get_linear_schedule_with_warmup(self.optimizer,num_warmup_steps=0,num_training_steps=total_steps)
        
        # losses = []
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            # Randomly select 10% of dataset
            indices = np.random.choice(len(train_dataset), size=int(0.10*len(train_dataset)), replace=False)
            subset_train_dataset = Subset(train_dataset, indices)
            train_loader = DataLoader(subset_train_dataset, batch_size=32, shuffle=True)
            num_batches = len(train_loader)
            
            loop = tqdm(train_loader, leave=True)
            for batch in loop:
                self.optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                # Looking inside BertForQuestionAnswering module I can see that it is already calculating and returning cross entropy loss
                # -> Only need to read it in and use it
                outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            avg_epoch_loss = epoch_loss / num_batches
            avg_losses_per_epoch.append(avg_epoch_loss)
            
            f1_epoch_score = self.evaluate(test_loader, tokenizer)
            f1_epoch_scores.append(f1_epoch_score)
            
            
        plt.figure(figsize=(10, 5))
        plt.plot(avg_losses_per_epoch, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Per Epoch During Training")
        plt.legend()
        plt.show()   

        plt.figure(figsize=(10, 5))
        plt.plot(f1_epoch_scores, label="F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("F1 Score During Training")
        plt.legend()
        plt.show()       


            
    def evaluate(self, test_loader, tokenizer):
        self.model.eval()
        
        f1_scores = []
        for batch in tqdm(test_loader):
          with torch.no_grad():
              
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            start_true = batch['start_positions'].to(self.device)
            end_true = batch['end_positions'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)
            
            # Calculate F1 score for each sample
            for i in range(len(input_ids)):
                pred_answer = tokenizer.decode(input_ids[i][start_pred[i]:end_pred[i]+1], skip_special_tokens=True)
                true_answer = tokenizer.decode(input_ids[i][start_true[i]:end_true[i]+1], skip_special_tokens=True)
                f1_scores.append(self.computeF1(pred_answer, true_answer))

        # overall_accuracy = sum(accuracies)/len(accuracies)
        # print(f"\nEvaluation Accuracy: {overall_accuracy:.2f}")
        average_f1 = sum(f1_scores) / len(f1_scores)
        # print(f"Average F1 score over test dataset is {average_f1}")
        return average_f1
    
    def computeF1(self, pred_answer, true_answer):
        pred_tokens = pred_answer.split()
        true_tokens = true_answer.split()
    
        common_tokens = Counter(pred_tokens) & Counter(true_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            return 0.0
    
        precision = num_common/len(pred_tokens)
        recall = num_common/len(true_tokens)
        
        f1 = 2*(precision*recall)/(precision + recall)
        return f1

class QAdataset(torch.utils.data.Dataset):
    '''
    Takes in tokenized data and wraps it in a class compatible with DataLoader.
    DataLoader needs two functions, getitem and len to work.
    '''
    def __init__(self, tokenized_data):
        # Store encodings dict
        self.tokenized_data = tokenized_data
        
    # Retrieve items from takenized data
    def __getitem__(self, idx):
        for key, val in self.tokenized_data.items():
            if val[idx] is None:
                raise ValueError(f"NoneType found in '{key}' at index {idx}.")
        
        return {key: torch.tensor(val[idx]) for key, val in self.tokenized_data.items()}
    
    # Number of samples in dataset
    def __len__(self):
        return len(self.tokenized_data.input_ids)














