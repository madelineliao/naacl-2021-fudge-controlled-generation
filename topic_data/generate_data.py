import os
import scipy
import numpy as np
import pickle
# import getpass
import torch
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import normalize
from datasets import load_dataset
from torch.utils.data import Dataset

# if getpass.getuser() == 'kechoi':
#     HUGGINGFACE_CACHE_DIR = '/atlas/u/kechoi/huggingface_cache'

load_dataset('scientific_papers', 'arxiv', cache_dir='./arxiv')
class Huggingface(Dataset):

    def __init__(
            self,
            name,
            cache_dir,
            max_seq_len=512,
            split='train',
            vocab=None,
            tokenizer=None,
            noise="mask",
            use_metric=False
        ):
        super().__init__()
        self.name = name
        self.cache_dir = cache_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.noise = noise
        self.use_metric = use_metric

        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab() if vocab is None else vocab
        self.i2w = {value:key for key, value in self.vocab.items()}
        # self.data, self.labels = self.load_raw_data()
        self.dataset = load_nlp_dataset(self.name, split=self.split,
                                   cache_dir=self.cache_dir)

    def get_word_counts(self, wc_path):
        word_counter = {idx: 0 for idx in self.i2w}

        for sample in self.dataset:
            key = get_nlp_key(self.name)
            text = sample[key]
            for word in self.tokenizer.tokenize(text.lower()):
                word_i = self.vocab[word]
                word_counter[word_i] += 1

        # save word counts
        with open(wc_path, 'wb') as fp:
            pickle.dump(word_counter, fp)

        return word_counter

    # def load_raw_data(self):
    #     dataset = load_nlp_dataset(self.name, split=self.split,
    #                                cache_dir=self.cache_dir)
    #     key = get_nlp_key(self.name)
    #     raw_data = []
    #     labels = []
    #     for row in dataset:
    #         text = row[key]
    #         label = row['label']
    #         raw_data.append(text)
    #         labels.append(label)
    #     return raw_data, labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        returns a dict with entries: 
            
            {
                'input_ids': [seq_len], 
                'attention_mask': [seq_len], 
                'target_ids': [seq_len],
                'labels': [label]
            } 
        
        for use with pretrained BART
        """
        key = get_nlp_key(self.name)
        data = self.dataset[key][index]
        label = self.dataset['label'][index]
        outputs = self.tokenizer(
            data,
            truncation=True, 
            padding='max_length', 
            max_length=self.max_seq_len,
            pad_to_max_length=True, 
            return_tensors='pt',
        )
        input_ids = outputs['input_ids'].squeeze()
        attention_mask = outputs['attention_mask'].squeeze()

        # we will be changing these live
        label_ids = input_ids.clone()
        outputs = dict(
            input_ids=input_ids,
            label_ids=label_ids,
            attention_mask=attention_mask,
            y=label
        )
        return outputs


def load_nlp_dataset(name, split, cache_dir='./cache'):
    if name == 'wikitext-2':
        return load_dataset('wikitext', 'wikitext-2-v1', split=split, cache_dir=cache_dir)
    elif name == 'wikitext-103':
        return load_dataset('wikitext', 'wikitext-103-v1', split=split, cache_dir=cache_dir)
    elif name == 'bookcorpus': 
        return load_dataset('bookcorpus', 'bookcorpus', split=split, cache_dir=cache_dir)
    elif name == 'yelp':
        return load_dataset('yelp_polarity', split=split, cache_dir=cache_dir)
    elif name == 'arxiv':
        return load_dataset('scientific_papers', 'arxiv', split=split, cache_dir=cache_dir)
    else:
        return load_dataset(name, split=split, cache_dir=cache_dir)


def get_nlp_key(name):
    mapping = {
        'wikitext-2': 'text',
        'wikitext-103': 'text',
        'bookcorpus':  'text',
        'yelp': 'text',
        'arxiv': 'abstract',
    }
    return mapping[name]