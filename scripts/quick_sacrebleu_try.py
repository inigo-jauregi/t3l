from sacrebleu.metrics import BLEU

refs = [['', 'It was not unexpected.', 'The man bit him first.'],
       ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

bleu = BLEU()

bleu_result = bleu.corpus_score(sys, refs)
print(bleu_result.score)
