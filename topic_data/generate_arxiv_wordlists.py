from argparse import ArgumentParser
import string

MIN_WORD_LEN = 5

def main():
    parser = ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--k', type=int, default='25')
    
    args = parser.parse_args()
    k = 0
    top_k_train = []
    with open(f'../arxiv_sorted_vocab_train.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) >= MIN_WORD_LEN and not any(i in string.punctuation for i in line):
                top_k_train.append(line)
                k += 1
            if k == args.k:
                break
    with open(f'arxiv_train_wordlist.txt', 'w') as f:
        for word in top_k_train:
            f.write(word + '\n')
    top_k_test = []
    k = 0
    with open(f'../arxiv_sorted_vocab_test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) >= MIN_WORD_LEN and not any(i in string.punctuation for i in line) and line not in top_k_train:
                top_k_test.append(line)
                k += 1
            if k == args.k:
                break
    with open(f'arxiv_test_wordlist.txt', 'w') as f:
        for word in top_k_test:
            f.write(word + '\n')

if __name__=='__main__':
    main()
