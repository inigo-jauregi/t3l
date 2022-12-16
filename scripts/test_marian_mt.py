from transformers import MarianMTModel, MarianTokenizer
src_text = ['Hola mi nombre es Iñigo.',
            'Como estais aqui?',
            'Yo estoy mu contento.'
            ]

model_name = 'Helsinki-NLP/opus-mt-es-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)

model = MarianMTModel.from_pretrained(model_name)
print(tokenizer(src_text, return_tensors="pt", padding=True))
translated = model.generate(num_beams=1, num_beam_groups=1, do_sample=False,
                            **tokenizer(src_text, return_tensors="pt", padding=True))
translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(translations)
