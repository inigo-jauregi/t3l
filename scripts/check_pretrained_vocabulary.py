from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('../pretrained_lm/sshleifer-tiny-mbart')

print('Number of tokens in vocab: ', len(tokenizer.get_vocab()))
print('class token: ', tokenizer.cls_token, ' (', tokenizer.cls_token_id, ')')
print('sep token: ', tokenizer.sep_token, ' (', tokenizer.sep_token_id, ')')
print('pad token: ', tokenizer.pad_token, ' (', tokenizer.pad_token_id, ')')
print('unk token: ', tokenizer.unk_token, ' (', tokenizer.unk_token_id, ')')
print('bos token: ', tokenizer.bos_token, ' (', tokenizer.bos_token_id, ')')
print('eos token: ', tokenizer.eos_token, ' (', tokenizer.eos_token_id, ')')
