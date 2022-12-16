import argparse
from transformers import AutoTokenizer

def main(args):

    tokenizer_src = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    tokenizer_src.src_lang = args.src_code
    tokenizer_tgt = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    tokenizer_tgt.src_lang = args.tgt_code

    # Open files
    src_file = open(args.dataset_path + '.' + args.src)
    tgt_file = open(args.dataset_path + '.' + args.tgt)

    # Output path
    output_path = "/".join(args.dataset_path.split('/')[:-1])
    output_path = output_path + '/train_pr'

    src_sen_lists = []
    for line in src_file:
        line_clean = line.strip()
        src_sen_lists.append(line_clean)

    tgt_sen_lists = []
    for line in tgt_file:
        line_clean = line.strip()
        tgt_sen_lists.append(line_clean)

    src_sen_lists_pair = []
    tgt_sen_lists_pair = []
    src_lens_list = []
    tgt_lens_list = []
    for i in range(len(src_sen_lists)):
        if not src_sen_lists[i].startswith('<') and not tgt_sen_lists[i].startswith('<'):
            src_tok = tokenizer_src.tokenize(src_sen_lists[i])
            tgt_tok = tokenizer_tgt.tokenize(tgt_sen_lists[i])
            if i < 10:
                print(src_tok)
                print(tgt_tok)
            num_tokens_src = len(src_tok)
            num_tokens_tgt = len(tgt_tok)
            if num_tokens_src < args.max_seq_len and num_tokens_tgt < args.max_seq_len:
                # Append
                src_sen_lists_pair.append(src_sen_lists[i])
                tgt_sen_lists_pair.append(tgt_sen_lists[i])
                src_lens_list.append(num_tokens_src)
                tgt_lens_list.append(num_tokens_tgt)

    print(len(src_sen_lists_pair))
    print(len(tgt_sen_lists_pair))

    avg_len_src = sum(src_lens_list) / float(len(src_sen_lists_pair))
    max_len_src = max(src_lens_list)
    avg_len_tgt = sum(tgt_lens_list) / float(len(tgt_sen_lists_pair))
    max_len_tgt = max(tgt_lens_list)

    print(f'SRC - avg token len: {avg_len_src}')
    print(f'SRC - max token len: {max_len_src}')
    print(f'TGT - avg token len: {avg_len_tgt}')
    print(f'TGT - max token len: {max_len_tgt}')

    assert len(src_sen_lists_pair) == len(tgt_sen_lists_pair), "Different number of sentences!"

    # Write file
    src_file_write = open(output_path + '.' + args.src, 'w')
    tgt_file_write = open(output_path + '.' + args.tgt, 'w')

    for i in range(len(src_sen_lists_pair)):
        src_file_write.write(src_sen_lists_pair[i])
        tgt_file_write.write(tgt_sen_lists_pair[i])
        if i < len(src_sen_lists_pair) - 1:
            src_file_write.write('\n')
            tgt_file_write.write('\n')

    src_file_write.close()
    tgt_file_write.close()


my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('--dataset_path', type=str, required=True,
                       help="Document to push to ES.")
my_parser.add_argument('--src', type=str, required=True,
                       help="Source language prefix.")
my_parser.add_argument('--tgt', type=str, required=True,
                       help="Target language prefix.")
my_parser.add_argument('--src_code', type=str, required=True,
                       help="Source language code for tokenizer.")
my_parser.add_argument('--tgt_code', type=str, required=True,
                       help="Target language code for tokenizer.")
my_parser.add_argument('--tokenizer', type=str, required=True,
                       help="Tokenizer.")
my_parser.add_argument('--max_seq_len', type=int, default=85,
                       help="Maximum sequence length.")

if __name__ == "__main__":
    my_args = my_parser.parse_args()
    main(my_args)
