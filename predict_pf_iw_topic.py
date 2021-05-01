import os
import random
import time
import pickle
import math
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *

def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    for cw in args.condition_words.split():
        assert cw in dataset_info.word2index
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_string)
    gpt_tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    gpt_pad_id = gpt_tokenizer.encode(PAD_TOKEN)[0]
    gpt_model = AutoModelWithLMHead.from_pretrained(args.model_string).to(args.device)
    gpt_model.eval()

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    while True:
        results = predict(gpt_model, 
                        gpt_tokenizer, 
                        conditioning_model, 
                        [args.input_text], 
                        args.condition_words, 
                        dataset_info, 
                        args.precondition_topk,
                        args.topk, 
                        args.length_cutoff,
                        condition_lambda=args.condition_lambda,
                        device=args.device)
        print(results)
        import pdb; pdb.set_trace()

def predict(gpt_model, gpt_tokenizer, conditioning_model, input_text, precondition_topk, postcondition_topk, length_cutoff, condition_lambda=1.0, device='cuda'):
    with torch.no_grad():
        batch_size = len(input_text)

        # assumes initially all same length.
        encoded_input = [gpt_tokenizer.encode(it, return_tensors='pt').to(device) for it in input_text] # batch x seq
        encoded_input = torch.cat(encoded_input, dim=0)
        lengths = torch.LongTensor([encoded_input.shape[1]]).to(device)
        
        num_particles = 20
        pf_encoded_inputs = [[gpt_tokenizer.encode(it, return_tensors='pt').to(device) for it in input_text] for _ in range(num_particles)]
        pf_lengths = [torch.LongTensor([pf_encoded_inputs[i].shape[1]]).to(device) for i in range(num_particles)]

        i = 0
        pf_is = [0 for i in range(num_particles)]
        pf_prev_condition_logits = {}
        while lengths.max() < length_cutoff:
            particle_probs = []
            for k in range(num_particles):
                tokens_left = torch.LongTensor([length_cutoff - pf_lengths[k].max() for _ in range(batch_size)]).to(device)
                gpt_logits = gpt_model(pf_encoded_inputs[k])[0][:, -1, :] # batch x vocab
                top_logits, top_indices = gpt_logits.topk(precondition_topk, dim=1) # batch x topk
                new_input_candidates = torch.cat([pf_encoded_inputs[k].unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2) # batch x topk x seq+1
                expanded_lengths = (pf_lengths[k] + 1).unsqueeze(1).expand(batch_size, precondition_topk) # batch x topk
                # expanded_tokens_left = tokens_left.unsqueeze(1).expand(-1, precondition_topk) # batch x topk
            
                condition_logits = conditioning_model(new_input_candidates.flatten(0, 1), # batch*topk x seq+1
                        expanded_lengths.flatten(0, 1))  # batch*topk
                condition_logits = condition_logits.view(batch_size, precondition_topk, -1) # batch x topk x N
                condition_logits = condition_logits[:, :, lengths].squeeze(2)
                condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs
                if pf_is[k] == 0:
                    pf_prev_condition_logits[k] = torch.zeros_like(condition_logits)
                    pf_is[k] += 1
                condition_logits = condition_logits - pf_prev_condition_logits[k]
                prev_condition_logits[k] = torch.clone(condition_logits)
            
                # self-normalize
                #print('condition_logits1:', condition_logits)
                condition_logits = torch.log(F.softmax(condition_logits, dim=1))#condition_logits - torch.log(sum(torch.exp(condition_logits)))#condition_logits / sum(condition_logits)
            #print('condition_logits2:', condition_logits)
            #condition_logits = torch.mean(condition_logits, dim=2)
                full_logits = top_logits + condition_logits * condition_lambda # batch x topk
                post_logits, post_indices = full_logits.topk(postcondition_topk, dim=1)
                post_probs = F.softmax(post_logits, dim=1)
                index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()] # batch
                next_indices = top_indices[torch.arange(batch_size).to(top_indices.device), index_into_top_indices] # batch
                pf_encoded_inputs[k] = torch.cat([pf_encoded_inputs[k], next_indices.unsqueeze(1)], dim=1) # batch x seq+1
                pf_lengths[k] = pf_lengths[k] + 1 # batch
                particle_probs.append(pf_encoded_inputs_k[]
            # resample
            particle_probs = 
            new_pf_encoded_inputs = pf_encoded_inputs[torch.multinomial( , num_particles, replacement=True), :]
        return [gpt_tokenizer.decode(s) for s in encoded_input]
        

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='gpt2-medium')

    parser.add_argument('--input_text', type=str, default=None, required=True, help='initial text')
    parser.add_argument('--condition_words', type=str, default=None, required=True, help='word(s) to optimize for')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--topk', type=int, default=10, help='consider top k outputs from gpt at each step')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=80, help='max length')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
