# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:31:27 2024

@author: nhmille
"""

from HW3_Supp import readData
from HW3_Supp import updateAnswerGroundTruth
from HW3_Supp import QAdataset
from HW3_Supp import fineTune
# from github_supp import check_for_none_entries
from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.utils.data import DataLoader

# Stored as tuple (passage, question, answer)
# answer is a dict {['answer_start'], ['answer_end']}
train_data = readData('spoken_train-v1.1.json')
test_data = readData('spoken_test-v1.1.json')

# Load the tokenizer for the pre-trained bert model
# Default config: {"do_lower_case": true, "model_max_length": 512}
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize the data
# Returns a dict with ['input_ids', 'token_type_ids', 'attention_mask']
tokenized_train_data = tokenizer(train_data['contexts'], train_data['questions'], truncation=True, padding=True, stride=128)
tokenized_test_data = tokenizer(test_data['contexts'], test_data['questions'], truncation=True, padding=True, stride=128)

# check_for_none_entries(tokenized_train_data)
# check_for_none_entries(tokenized_test_data)

# Adjust positions to reflect new special tokens
updateAnswerGroundTruth(tokenized_train_data, train_data['answers'], tokenizer)
updateAnswerGroundTruth(tokenized_test_data, test_data['answers'], tokenizer)

# check_for_none_entries(tokenized_train_data)
# check_for_none_entries(tokenized_test_data)

# Define datasets -> Class with methods to get item and length -> Comptaible with DataLoader
train_dataset = QAdataset(tokenized_train_data)
test_dataset = QAdataset(tokenized_test_data)

# Pass dataset classes to data loaders
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Read in pre-trained BERT model
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Setup up tuner
tuner = fineTune(model, epochs=75, lr=7.5e-5)

# Fine tune model -> Evaluation is done internally
# Passing training data so I can internally make random dataloaders for each 10% epoch
tuner.train(train_dataset, test_loader, tokenizer)
tuner.evaluate(test_loader, tokenizer)















