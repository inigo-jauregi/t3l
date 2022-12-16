import argparse

import torch
import pandas as pd

from transformers import AutoTokenizer, BartForConditionalGeneration, MBartTokenizer, AutoModelForSeq2SeqLM


def main(args):

    # read df
    df = pd.read_csv(args.dev_path, delimiter='\t', header=None)

    # Load models
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer = MBartTokenizer.from_pretrained(args.tokenizer, src_lang="es_XX", tgt_lang="en_XX")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).cuda()
    model.eval()

    list_prem_trans = []
    list_hyp_trans = []
    list_gt_label = []
    for i, row in df.iterrows():

        print(i)
        premise = row[0]
        hypothesis = row[1]
        label = row[2]
        # print(premise)
        # print(hypothesis)

        prem_ids = tokenizer.encode(premise.strip())
        prem_mask = [1] * len(prem_ids)
        hyp_ids = tokenizer.encode(hypothesis.strip())
        hyp_mask = [1] * len(hyp_ids)

        # convert to torch
        prem_ids = torch.tensor(prem_ids).unsqueeze(0).cuda()
        prem_mask = torch.tensor(prem_mask).unsqueeze(0).cuda()
        hyp_ids = torch.tensor(hyp_ids).unsqueeze(0).cuda()
        hyp_mask = torch.tensor(hyp_mask).unsqueeze(0).cuda()

        # Prem translation
        prem_trans_ids = model.generate(input_ids=prem_ids, attention_mask=prem_mask,
                                        decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
                                        max_length=args.max_output_len, num_beams=1, num_beam_groups=1, do_sample=False)
        # Hyp translation
        hyp_trans_ids = model.generate(input_ids=hyp_ids, attention_mask=hyp_mask,
                                       decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
                                       max_length=args.max_output_len, num_beams=1, num_beam_groups=1, do_sample=False)

        prem_trans = tokenizer.batch_decode(prem_trans_ids.tolist(), skip_special_tokens=True)[0]
        hyp_trans = tokenizer.batch_decode(hyp_trans_ids.tolist(), skip_special_tokens=True)[0]

        # print(prem_trans)
        # print(hyp_trans)

        list_prem_trans.append(prem_trans)
        list_hyp_trans.append(hyp_trans)
        list_gt_label.append(label)

    df_out = pd.DataFrame()
    df_out['prem'] = list_prem_trans
    df_out['hyp'] = list_hyp_trans
    df_out['label'] = list_gt_label

    df_out.to_csv(args.out_preds_path, sep='\t', header=False, index=False)


my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('--dev_path', type=str, required=True,
                       help="Dataset path.")
my_parser.add_argument('--tokenizer', type=str, required=True,
                       help="Tokenizer path.")
my_parser.add_argument('--model', type=str, required=True,
                       help="Model path.")
my_parser.add_argument('--out_preds_path', type=str, required=True,
                       help="Model path.")
my_parser.add_argument("--max_input_len", type=int, default=170,
                        help="maximum num of wordpieces/summary. Used for training and testing")
my_parser.add_argument("--max_output_len", type=int, default=170,
                        help="maximum num of wordpieces/summary. Used for training and testing")


if __name__ == "__main__":
    my_args = my_parser.parse_args()
    main(my_args)
