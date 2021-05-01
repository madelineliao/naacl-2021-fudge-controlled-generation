import random
import math
import os
import pickle
from collections import defaultdict, namedtuple
import string
from datasets import load_dataset
from looping import LoopingDataset
from torch.utils.data import Dataset as TorchDataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # turn off since we're using multiple threads for loading anyway

from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model
import numpy as np
from tqdm import tqdm
import torch

from util import suppress_stdout
from poetry_util import is_iambic, count_syllables, get_rhymes, get_rhyme_group
from constants import *


DatasetInfo = namedtuple('DatasetInfo', 
                ['index2word', 'word2index', 'total_words', 'vocab', 'glove_embeddings'])
RhymeInfo = namedtuple('RhymeInfo', 
                ['word2rhyme_group', 'rhyme_group_counts', 'rhyme_groups', 'index2rhyme_group', 'rhyme_group2index', 'total_rhyme_groups'])


def collate(batch):
    pad_id = batch[0][4]
    inputs = [b[0] for b in batch]
    lengths = torch.LongTensor([b[1] for b in batch])
    max_length = lengths.max()
    for i in range(len(inputs)):
        if len(inputs[i]) < max_length:
            inputs[i] = torch.cat([inputs[i], torch.zeros(max_length - len(inputs[i])).long()], dim=0) # actually 0 is fine as pad since it's masked out
    inputs = torch.stack(inputs, dim=0)
    future_words = torch.LongTensor([b[2] for b in batch]).unsqueeze(0).expand(len(batch), -1).clone() # batch x N=batch
    labels = torch.zeros_like(future_words).long()
    labels = labels.scatter(1, torch.arange(len(batch)).unsqueeze(1), torch.ones(len(batch)).long().unsqueeze(1)).clone()
    log_probs = torch.Tensor([b[3] for b in batch])
    classification_labels = [b[5] for b in batch] # batch
    if type(classification_labels[0]) == list:
        for i in range(len(classification_labels)):
            assert len(classification_labels[i]) == lengths[i]
            if len(classification_labels[i]) < max_length:
                classification_labels[i] = torch.cat([torch.LongTensor(classification_labels[i]), -1 + torch.zeros(max_length - len(classification_labels[i])).long()], dim=0)
            else:
                classification_labels[i] = torch.LongTensor(classification_labels[i])
        classification_labels = torch.stack(classification_labels, dim=0) # batch x seq
    else:
        assert type(classification_labels[0]) == int
        classification_labels = torch.LongTensor(classification_labels) # they're just int labels
    syllables_to_go = torch.LongTensor([b[6] for b in batch])
    future_word_num_syllables = torch.LongTensor([b[7] for b in batch])
    rhyme_group_index = torch.LongTensor([b[8] for b in batch])
    return (inputs, lengths, future_words, log_probs, labels, classification_labels, syllables_to_go, future_word_num_syllables, rhyme_group_index)


def load_rhyme_info(index2word, vocab):
    word2rhyme_group = defaultdict(lambda: UNKNOWN_RHYME_GROUP)
    rhyme_group_counts = defaultdict(lambda: 0)
    rhyme_groups = set()
    for word in index2word:
        try:
            rhyme_group = get_rhyme_group(word)
            word2rhyme_group[word] = rhyme_group
            rhyme_group_counts[rhyme_group] += (vocab[word] if word in vocab else 1) # for rare words not in vocab, just use 1
            rhyme_groups.add(rhyme_group)
        except:
            rhyme_group_counts[UNKNOWN_RHYME_GROUP] += (vocab[word] if word in vocab else 1)
    index2rhyme_group = [UNKNOWN_RHYME_GROUP] + sorted(list(rhyme_groups))
    rhyme_group2index = {s: i for i, s in enumerate(index2rhyme_group)}
    total_rhyme_groups = sum(rhyme_group_counts.values())

    return RhymeInfo(word2rhyme_group=dict(word2rhyme_group), 
                     rhyme_group_counts=dict(rhyme_group_counts), 
                     rhyme_groups=rhyme_groups, 
                     index2rhyme_group=index2rhyme_group, 
                     rhyme_group2index=rhyme_group2index, 
                     total_rhyme_groups=total_rhyme_groups)


class SplitDataset(TorchDataset):
    def __init__(self, args, split='train', shuffle=True):
        self.args = args
        self.split = split
        self.perc = args.perc
        self.shuffle = shuffle

        self.tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_STRING)
        self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        self.gpt_pad_id = self.tokenizer.encode(PAD_TOKEN)[0] # actually just the vocab size
        self.vocab = defaultdict(lambda: 0)
        self.target_vocab = defaultdict(lambda: 0)

        # self.dataset_name = args.dataset_name
        # self.task = args.task
        full_source_data = self.load_data_path(args.source_dir, split, 'source')
        full_target_data = self.load_data_path(args.target_dir, split, 'target')
        self.source_data, self.target_data = self.initialize_data_splits(full_source_data, full_target_data, split)

        self.source_labels = torch.zeros(len(self.source_data))
        self.target_labels = torch.ones(len(self.target_data))
        print('len(self.vocab): ', len(self.vocab))
    
    def load_data_path(self, path, split, domain):
        if domain == 'source':
            # data = load_dataset('scientific_papers', 'arxiv', cache_dir='./arxiv', split=split)
            data = load_dataset('text', data_files='/atlas/u/madeline/naacl-2021-fudge-controlled-generation/train_data/gpt2_generations/out.txt')
            data = data['train'].train_test_split(test_size=0.2)[split]
            return data
        elif domain == 'target':
            data = load_dataset('yelp_polarity', cache_dir='/atlas/u/madeline/naacl-2021-fudge-controlled-generation/topic_data/yelp', split=split)
        else:
            raise NotImplementedError
        return data

    def initialize_data_splits(self, full_source_data, full_target_data, split):
        min_size = min(len(full_source_data), len(full_target_data))
        if split == 'train':
            num_source_samples = min_size
            num_target_samples = int(self.perc * min_size)
        else:
            num_source_samples = num_target_samples = min_size
            
        source_idxs = random.sample(range(len(full_source_data)), num_source_samples)
        target_idxs = random.sample(range(len(full_target_data)), num_target_samples)
        
        new_source_data = full_source_data.select(source_idxs)
        # print(type(full_target_data['abstract']))
        # new_target_data = np.array(full_target_data['abstract'])[target_idxs]
        new_target_data = full_target_data.select(target_idxs)

        if self.shuffle:
            new_source_data = new_source_data.shuffle()
            new_target_data = new_target_data.shuffle()

        for seq in new_source_data:
            for word in seq['text'].strip().split(' '):
                self.vocab[word] += 1
        for i, seq in enumerate(new_target_data['text']):
            if seq.isspace():
                new_target_data['text'].pop(i)
                continue
            new_seq = seq
            
            for word in seq.strip().split(' '):
                # remove bad or numerical tokens
                if '@' in word:
                    new_seq = new_seq.replace(word, '')
                else:
                    self.vocab[word] += 1
                    self.target_vocab[word] +=1
        new_target_data['text'][i] = new_seq
        sorted_words = sorted(self.target_vocab.keys(), key=lambda x: self.target_vocab[x], reverse=True)
        for i in range(len(sorted_words)):
            sorted_words[i] = sorted_words[i] + '\n'
        with open(f"yelp_sorted_vocab_{split}.txt", 'w+') as f:
            f.writelines(sorted_words)
        return LoopingDataset(new_source_data['text']), LoopingDataset(new_target_data['text'])
        
    def __getitem__(self, index):
        source_sample, target_sample = self.source_data[index], self.target_data[index]
        # source_label, target_label = self.source_labels[index], self.target_labels[index]
        
        while len(source_sample) <= 0 or len(target_sample) <= 0:
            index = np.random.choice(len(self.source_data))
            source_sample, target_sample = self.source_data[index], self.target_data[index]

        source_sample = self.tokenizer.encode(source_sample, return_tensors='pt')[0][:100]
        target_sample = self.tokenizer.encode(target_sample, return_tensors='pt')[0][:100]
        
        source_length = len(source_sample)
        target_length = len(target_sample)   

        if source_length < 100:
            source_sample = torch.cat([source_sample, torch.zeros(100 - source_length).long()], dim=0) 
        if target_length < 100:
            target_sample = torch.cat([target_sample, torch.zeros(100 - target_length).long()], dim=0)

        return (source_sample, source_length), (target_sample, target_length)
        #return (source_sample, source_label), (target_sample, target_label)
    
    def __len__(self):
        return len(self.source_data) + len(self.target_data)

class Dataset:
    def __init__(self, args):
        print('loading data')
        random.seed(args.seed)
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.topic = args.task == 'topic'
        self.formality = args.task == 'formality'
        self.iambic = args.task == 'iambic'
        self.rhyme = args.task == 'rhyme'
        self.newline = args.task == 'newline'

        self.tokenizer = AutoTokenizer.from_pretrained(FORMALITY_MODEL_STRING if self.formality else TOPIC_MODEL_STRING)
        self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        self.gpt_pad_id = self.tokenizer.encode(PAD_TOKEN)[0] # actually just the vocab size
        sentences = []
        self.vocab = defaultdict(lambda: 0)
        if self.formality:
            self.vocab['placeholder'] = 1 # anything so we don't crash
            train, val, test = [], [], []
            for category, label in [('formal', 1), ('informal', 0)]:
                with open(os.path.join(args.data_dir, 'train', category), 'r') as rf:
                    for i, line in enumerate(rf):
                        if len(line) > FORMALITY_MAX_LEN:
                            line = ' '.join(line.strip()[:FORMALITY_MAX_LEN].split()[:-1]) # cutoff words until below max len; chosen so only ~20 examples affected in dataset
                        if i < FORMALITY_VAL_SIZE // 2:
                            val.append((line.strip(), label))
                        else:
                            train.append((line.strip(), label))
                with open(os.path.join(args.data_dir, 'test', category), 'r') as rf:
                    for line in rf:
                        if len(line) > FORMALITY_MAX_LEN:
                            line = ' '.join(line.strip()[:FORMALITY_MAX_LEN].split()[:-1]) # cutoff words until below max len
                        test.append((line.strip(), label))
            self.splits = {}
            self.splits['train'], self.splits['val'], self.splits['test'] = train, val, test
            
        else: # topic / poetry
            for root, _, filenames in os.walk(args.data_dir):
                for fname in filenames:
                    with open(os.path.join(root, fname), 'r') as rf:
                        for line in rf:
                            sentences.append(line.strip())
                            for word in line.strip().split(' '):
                                self.vocab[word] += 1
            random.shuffle(sentences)
            self.splits = {}
            if args.debug:
                self.splits['val'] = sentences
                self.splits['test'] = sentences
                self.splits['train'] = sentences
            else:
                self.splits['val'] = sentences[:TOPIC_VAL_SIZE]
                self.splits['test'] = sentences[TOPIC_VAL_SIZE:2*TOPIC_VAL_SIZE]
                self.splits['train'] = sentences[2*TOPIC_VAL_SIZE:]

        if args.dataset_info is not None:
            print('loading dataset info from file')
            with open(args.dataset_info, 'rb') as rf:
                dataset_info = pickle.load(rf)
            self.vocab, self.total_words, self.index2word, self.word2index, self.glove_embeddings = \
                dataset_info.vocab, dataset_info.total_words, dataset_info.index2word, dataset_info.word2index, dataset_info.glove_embeddings
            self.dataset_info = dataset_info
        else:
            print('generating dataset info from scratch')
            words_values = list(self.vocab.items())
            words_values = sorted(words_values, key=lambda x: x[1], reverse=True)
            if args.glove_file is None:
                print('no glove embeddings given')
                for word, _ in words_values[VOCAB_SIZE:]: # only use somewhat common tokens
                    del self.vocab[word]
                glove_embeddings = None
            else:
                print('loading glove embeddings')
                glove_embeddings = {}
                with open(args.glove_file, 'r') as rf:
                    for i, line in enumerate(rf):
                        if i % GLOVE_PRINT_PROGRESS_FREQ == 0:
                            print(i)
                        line = line.strip().split()
                        if len(line) != GLOVE_DIM + 1:
                            continue # skip multi-word embeddings which are rare anyway
                        glove_embeddings[line[0]] = [float(x) for x in line[1:]]
                for word, _ in words_values:
                    if word not in glove_embeddings:
                        del self.vocab[word]
            self.total_words = sum(self.vocab.values())
            self.index2word = [PAD_TOKEN] + sorted(list(self.vocab.keys()))
            self.word2index = {s: i for i, s in enumerate(self.index2word)}
            self.vocab = dict(self.vocab) # so we can pickle later
            if glove_embeddings is None:
                self.glove_embeddings = None
            else:
                self.glove_embeddings = torch.stack([torch.zeros(GLOVE_DIM)] + [torch.Tensor(glove_embeddings[word]) for word in self.index2word[1:]], dim=0)

            self.dataset_info = DatasetInfo(index2word=self.index2word,
                                            word2index=self.word2index,
                                            total_words=self.total_words,
                                            vocab=self.vocab,
                                            glove_embeddings=self.glove_embeddings)
        
        if self.rhyme:
            if args.rhyme_info is not None:
                print('loading rhyme info from file')
                with open(args.rhyme_info, 'rb') as rf:
                    self.rhyme_info = pickle.load(rf)
            else:
                self.rhyme_info = load_rhyme_info(self.index2word, self.vocab)
            self.word2rhyme_group, self.rhyme_group_counts, self.rhyme_groups, self.index2rhyme_group, self.rhyme_group2index, self.total_rhyme_groups = \
                    defaultdict(lambda: UNKNOWN_RHYME_GROUP, self.rhyme_info.word2rhyme_group), self.rhyme_info.rhyme_group_counts, self.rhyme_info.rhyme_groups, self.rhyme_info.index2rhyme_group, self.rhyme_info.rhyme_group2index, self.rhyme_info.total_rhyme_groups

        print('done loading data')
        print('split sizes:')
        for key in ['train', 'val', 'test']:
            print(key, len(self.splits[key]))
        if not self.formality:
            print('total words', self.total_words)
            print('vocab size', len(self.index2word))


    def shuffle(self, split, seed=None):
        assert split in ['train', 'val', 'test']
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.splits[split])


    def loader(self, split, num_workers=20, indices=None):
        assert split in ['train', 'val', 'test']
        data = self.splits[split] if indices is None else [self.splits[split][i] for i in indices]
        return torch.utils.data.DataLoader(SplitLoader(data, self), batch_size=self.batch_size, pin_memory=True, collate_fn=collate, num_workers=num_workers)


class SplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, data, parent):
        super(SplitLoader).__init__()
        self.data = data
        self.pos = 0
        self.parent = parent


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        return self
    

    def __next__(self):
        increment = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: # # in a worker process
            increment = worker_info.num_workers
            worker_id = worker_info.id
            if self.pos == 0:
                self.pos = worker_id
        valid = False
        while not valid:
            if self.pos >= len(self):
                raise StopIteration
            if self.parent.topic:
                failed = False
                future_word_num_syllables, rhyme_group_index, syllables_to_go = -1, -1, -1
                raw_sentence, classification_label = self.data[self.pos], -1
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
                length = len(sentence)
                min_sentence_length = MIN_SENTENCE_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = random.randint(1, length - 1) # for lm, learn all positions at once
                    inp = sentence[:pos_to_split]
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    if not failed and num_words_in_input < len(original_sentence):
                        future_word_position_max = len(original_sentence) - 1
                        future_word_position = random.randint(num_words_in_input-1, future_word_position_max) # allow the last possibly partial word though
                        future_word = original_sentence[future_word_position]
                        unstripped_future_word = future_word
                        future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                        if not failed and future_word in self.parent.word2index.keys():
                            word_log_prob = math.log(self.parent.vocab[future_word] / self.parent.total_words) # roughly baseline prob of word under noise model
                            future_word = self.parent.word2index[future_word]
                            pad_id = self.parent.gpt_pad_id
                            example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                            valid = not failed
            elif self.parent.formality:
                future_word_num_syllables, rhyme_group_index, syllables_to_go = -1, -1, -1
                raw_sentence, classification_label = self.data[self.pos]
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
                length = len(sentence)
                min_sentence_length = MIN_SENTENCE_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = length # no need to split since we already have the label
                    inp = sentence[:pos_to_split]
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    # only look up to 10 words ahead if we're doing count syllables, since we'll filter out anything more than 10 syllables ahead anyway
                    future_word_position_max = len(original_sentence) - 1
                    future_word_position = 0
                    future_word = 'placeholder'
                    unstripped_future_word = future_word
                    future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                    word_log_prob, future_word = 0, 0
                    pad_id = self.parent.gpt_pad_id
                    example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                    valid = True
            elif self.parent.iambic:
                failed = False
                future_word_num_syllables, rhyme_group_index, syllables_to_go = -1, -1, -1
                raw_sentence, classification_label = self.data[self.pos], -1
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
                length = len(sentence)
                min_sentence_length = MIN_SENTENCE_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = random.randint(0, length - 1)
                    # try to get a subseq of exactly 10 syllables
                    inp = sentence[pos_to_split:]
                    num_syllables = 0
                    checked = False
                    for i in range(1, len(inp)):
                        decoded = self.parent.tokenizer.decode(inp[:i])
                        num_syllables = count_syllables(decoded)
                        if num_syllables > POETRY_LINE_SYLLABLES:
                            inp = inp[:i-1] # might get a few data points where the split is in the middle of a word, but it should be ok for learning. 
                            last_line_length = i-1
                            decoded = self.parent.tokenizer.decode(inp)
                            num_syllables = count_syllables(decoded)
                            checked = True
                            break
                    if not checked or num_syllables != POETRY_LINE_SYLLABLES:
                        failed = True
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    classification_label = [is_iambic(self.parent.tokenizer.decode(inp)) for _ in range(length)] # predict for whole seq including future
                    # only look up to 10 words ahead if we're doing count syllables, since we'll filter out anything more than 10 syllables ahead anyway
                    future_word_position_max = len(original_sentence) - 1
                    future_word_position = 0
                    future_word = 'placeholder'
                    unstripped_future_word = future_word
                    future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                    if not failed:
                        word_log_prob, future_word = 0, 0
                        pad_id = self.parent.gpt_pad_id
                        example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                        valid = not failed
            elif self.parent.rhyme:
                failed = False
                future_word_num_syllables, rhyme_group_index = -1, -1
                raw_sentence, classification_label = self.data[self.pos], -1
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
                length = len(sentence)
                min_sentence_length = MIN_SENTENCE_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = random.randint(1, length - 1) # for lm, learn all positions at once
                    inp = sentence[:pos_to_split]
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    if not failed and num_words_in_input < len(original_sentence):
                        # only look up to 10 words ahead if we're doing count syllables, since we'll filter out anything more than 10 syllables ahead anyway
                        future_word_position_max = min(len(original_sentence) - 1, num_words_in_input + MAX_COUNT_SYLLABLE_DIST)
                        future_word_position = random.randint(num_words_in_input-1, future_word_position_max) # allow the last possibly partial word though
                        future_word = original_sentence[future_word_position]
                        unstripped_future_word = future_word
                        future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                                                        
                        words_in_between = original_sentence[num_words_in_input-1:future_word_position+1]
                        syllables_to_go = count_syllables(' '.join(words_in_between))
                        if syllables_to_go > MAX_COUNT_SYLLABLE_DIST:
                            failed = True
                        future_word_num_syllables = count_syllables(future_word)
                        rhyme_group = self.parent.word2rhyme_group[future_word]
                        rhyme_group_index = self.parent.rhyme_group2index[rhyme_group]
                        # truncate context a bit since we're just doing couplets. random length from 1 to max desired length for this purpose. 
                        desired_length = random.randint(1, MAX_COUNT_SYLLABLE_INPUT_LENGTH)
                        inp = inp[-desired_length:]
                        length = len(inp)

                        if not failed and future_word in self.parent.word2index.keys():
                            word_log_prob = math.log(self.parent.rhyme_group_counts[rhyme_group] / self.parent.total_rhyme_groups)
                            future_word = rhyme_group_index # future conditioning is just the rhyme group in this case
                            pad_id = self.parent.gpt_pad_id
                            example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                            valid = not failed
            elif self.parent.newline:
                failed = False
                future_word_num_syllables, rhyme_group_index = -1, -1
                raw_sentence, classification_label = self.data[self.pos], -1
                original_sentence = raw_sentence.split()
                sentence = self.parent.tokenizer.encode(raw_sentence, return_tensors='pt')[0]
                length = len(sentence)
                min_sentence_length = MIN_SENTENCE_LENGTH
                if len(sentence) > min_sentence_length: # set to 3. well, everything in data is > 3 for the bag of words task
                    pos_to_split = random.randint(1, length - 1) # for lm, learn all positions at once
                    inp = sentence[:pos_to_split]
                    while pos_to_split < len(sentence):
                        if len(self.parent.tokenizer.decode(inp).split()) == len(self.parent.tokenizer.decode(sentence[:pos_to_split + 1]).split()):
                            pos_to_split += 1
                            inp = sentence[:pos_to_split]
                        else:
                            break
                    length = len(inp)
                    num_words_in_input = len(self.parent.tokenizer.decode(inp).split())
                    if not failed and num_words_in_input < len(original_sentence):
                        # only look up to 10 words ahead if we're doing count syllables, since we'll filter out anything more than 10 syllables ahead anyway
                        future_word_position_max = len(original_sentence) - 1
                        future_word_position = random.randint(num_words_in_input-1, future_word_position_max) # allow the last possibly partial word though
                        future_word = original_sentence[future_word_position]
                        unstripped_future_word = future_word
                        future_word = future_word.strip().strip(string.punctuation) # NOTE: we didn't strip punctuation for the topic bag of words paper experiments for our method. it doesn't make much difference, though.
                                                        
                        # future_word = original_sentence[-1] # useful for debugging
                        words_in_between = original_sentence[num_words_in_input-1:future_word_position+1]
                        syllables_to_go = count_syllables(' '.join(words_in_between))
                        if syllables_to_go > MAX_COUNT_SYLLABLE_DIST:
                            failed = True
                        # truncate context a bit since we're just doing couplets. random length from 1 to max desired length for this purpose. 
                        desired_length = random.randint(1, MAX_COUNT_SYLLABLE_INPUT_LENGTH)
                        # desired_length = 10 # useful for debugging
                        inp = inp[-desired_length:]
                        length = len(inp)
                        true_label = 1 if unstripped_future_word.strip()[-1] in PHRASE_ENDS else 0 # common ways to end a phrase
                        classification_label = [-1 for _ in range(length)]
                        classification_label[-1] = true_label # only learn at the last position
                        if not failed and future_word in self.parent.word2index.keys():
                            word_log_prob = math.log(self.parent.vocab[future_word] / self.parent.total_words) # roughly baseline prob of word under noise model
                            future_word = self.parent.word2index[future_word]
                            pad_id = self.parent.gpt_pad_id
                            example = (inp, length, future_word, word_log_prob, pad_id, classification_label, syllables_to_go, future_word_num_syllables, rhyme_group_index)
                            valid = not failed
            else:
                raise NotImplementedError

            self.pos += increment
        return example
