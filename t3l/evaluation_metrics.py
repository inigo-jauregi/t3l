import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def text_classification_metrics(out_logits, labels):

    # Obtain the max score
    # print(out_logits)
    out_logits = out_logits.softmax(dim=-1)
    preds = out_logits.argmax(dim=-1)
    # print(preds)
    # for elem in preds:
    #     # print(elem)
    #     my_writer.write(str(int(elem)) + '\n')
    # for elem in labels:
    #     # print(elem)
    #     my_writer_label.write(str(int(elem)) + '\n')


    preds_np = preds.clone().detach().cpu().tolist()
    # print(len(preds_np))
    labels_np = labels.clone().detach().cpu().tolist()
    # print(len(labels_np))
    acc = torch.tensor(accuracy_score(labels_np, preds_np), device=out_logits.device)  # .cuda()
    prec = torch.tensor(precision_score(labels_np, preds_np, average='macro', zero_division=0),
                        device=out_logits.device)  # .cuda()
    rec = torch.tensor(recall_score(labels_np, preds_np, average='macro', zero_division=0), device=out_logits.device)
    f1 = torch.tensor(f1_score(labels_np, preds_np, average='macro', zero_division=0), device=out_logits.device)

    return acc, prec, rec, f1


def multi_label_text_classification_metrics(out_logits, labels):

    # Convert to probabilities
    out_probs = out_logits.sigmoid().clone().detach().cpu().numpy()

    # Output binary
    output_binary = out_probs.copy()
    # Threshold (> 0.5)
    output_binary = output_binary >= 0.5
    # output_binary[output_binary < 0.5] = 0.0

    # Accuracy
    output_preds_labels = [list(np.nonzero(t)[0]) for t in output_binary]
    # print(len(preds_np))
    labels_np = labels.clone().detach().cpu().numpy()
    gt_labels_np = [list(np.nonzero(t)[0]) for t in labels_np]
    gt_labels_binary = labels_np == 1
    # print(len(labels_np))
    acc = torch.tensor(accuracy_score(gt_labels_binary, output_binary), device=out_logits.device)  # .cuda()

    # mean R-Precision@K
    out_sorted_labels = np.argsort(-out_probs)
    mrp = 0.0
    for i in range(out_probs.shape[0]):
        row_preds = out_sorted_labels[i]
        row_labels = gt_labels_np[i]
        num_gt_labels = len(gt_labels_np[i])
        row_preds = row_preds[0:num_gt_labels]
        tp = len(np.intersect1d(row_labels,row_preds))
        prec_at_k = tp / num_gt_labels
        mrp += prec_at_k
    # mean
    mrp = torch.tensor(mrp / out_probs.shape[0], device=out_logits.device)

    return acc, mrp, torch.tensor(0., device=out_logits.device), torch.tensor(0., device=out_logits.device)


def translation_metrics(out_logits, labels, tokenizer=None):

    # print(out_logits.softmax(dim=-1).argmax(dim=-1).size())
    # print(labels.size())
    out_logits = out_logits.softmax(dim=-1).argmax(dim=-1)
    # print([tokenizer.decode(t, skip_special_tokens=False) for t in out_logits])
    out_logits_reshaped = out_logits.view(-1,1)
    # print([tokenizer.decode(t, skip_special_tokens=False) for t in labels])
    equals_vec = out_logits_reshaped.eq(labels.view(-1,1))
    # print(torch.sum(equals_vec))
    # print(equals_vec.size())
    acc = torch.sum(equals_vec) / equals_vec.size()[0]

    return acc
