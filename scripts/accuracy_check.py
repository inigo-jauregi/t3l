from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

lm_preds_read = open('huggingface_preds.txt')
list_lm_preds = []
for line in lm_preds_read:
    list_lm_preds.append(int(line.strip()))

hg_preds_read = open('huggingface_labels.txt')
list_hg_preds = []
for line in hg_preds_read:
    list_hg_preds.append(int(line.strip()))

acc = accuracy_score(list_hg_preds, list_lm_preds)
print(acc)
