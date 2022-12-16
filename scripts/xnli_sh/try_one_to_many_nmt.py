from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm

def main():


    model = MBartForConditionalGeneration.from_pretrained("../../pretrained_lm/facebook-mbart-large-50-one-to-many-mmt",
                                                          )
    model.cuda()
    tokenizer = MBart50TokenizerFast.from_pretrained("../../pretrained_lm/facebook-mbart-large-50-one-to-many-mmt",
                                                     src_lang="en_XX")

    raw_list = []
    with open('../../my_datasets/sequence_classification/xnli/dev-en-hyp.en', 'r') as reader:
        for line in reader:
            raw_list.append(line.strip())

    print('Raw num: ', len(raw_list))

    batch_size = 64
    translated_list = []
    num_batches = int(len(raw_list) / batch_size)
    for i in tqdm(range(num_batches)):

        model_inputs = tokenizer(raw_list[i*batch_size:(i*batch_size)+batch_size], return_tensors="pt",
                                 padding=True, truncation=True, max_length=180)
        model_inputs_gpu = {}
        for key, val in model_inputs.items():
            model_inputs_gpu[key] = val.cuda()

        # print(model_inputs)

        # translate from English to Hindi
        generated_tokens = model.generate(
            **model_inputs_gpu,
            forced_bos_token_id=tokenizer.lang_code_to_id["mk_MK"]
        )
        translated_sen = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_list += translated_sen

    # Last batch
    model_inputs = tokenizer(raw_list[num_batches * batch_size:], return_tensors="pt",
                             padding=True, truncation=True, max_length=180)
    model_inputs_gpu = {}
    for key, val in model_inputs.items():
        model_inputs_gpu[key] = val.cuda()

    # print(model_inputs)

    # translate from English to Hindi
    generated_tokens = model.generate(
        **model_inputs_gpu,
        forced_bos_token_id=tokenizer.lang_code_to_id["mk_MK"]
    )
    translated_sen = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    translated_list += translated_sen

    print('Translated num: ', len(translated_list))
    with open('../../my_datasets/sequence_classification/xnli/dev-en-hyp.el', 'w') as writer:
        for line in translated_list:
            writer.write(line + '\n')



if __name__ == "__main__":
    main()
