import os
from typing import List, Optional, Tuple, Dict
import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from transformers import BertConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

from t3l.custom_data_loaders import XNLI_LABEL2ID, MLDOC_LABEL2ID, MARC_LABEL2ID, MULTIEURLEX_LEVEL_1_LABEL2ID, \
    XNLI_ID2LABEL


class NmtPlusLmForTextClassificationConfig(BertConfig):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks',
                 gradient_checkpointing: bool = False, **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        self.gradient_checkpointing = gradient_checkpointing
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2']


class NmtPlusLmForTextClassification(nn.Module):
    def __init__(self, tokenizer, model_seq2seq, model_lm, max_input_len, max_output_len, freeze_strategy, task,
                 save_int_trans=None):
        super().__init__()
        self.embed_dim = 512  # config.d_model
        self.freeze_strategy = freeze_strategy
        self.tokenizer = tokenizer
        if task == 'XNLI':
            self.num_out_classes = len(XNLI_LABEL2ID)
        elif task == 'MLdoc':
            self.num_out_classes = len(MLDOC_LABEL2ID)
        elif task == 'MARC':
            self.num_out_classes = len(MARC_LABEL2ID)
        elif task == 'MultiEurlex':
            self.num_out_classes = len(MULTIEURLEX_LEVEL_1_LABEL2ID)
        self.nmt = AutoModelForSeq2SeqLM.from_pretrained(model_seq2seq)
        self.lm = AutoModelForSequenceClassification.from_pretrained(model_lm, num_labels=self.num_out_classes)
        self.freeze_params()
        self.lm_emb_matrix = self.lm.get_input_embeddings().weight

        # Max lengths
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        # Task
        self.task = task

    def forward(
        self,
        query,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        need_weights=False,
        output_attentions=False,
        test_trans=None
    ) -> Tuple[Tensor, Optional[Tensor]]:

        bsz, src_len = query.size()
        # assert embed_dim == self.embed_dim
        assert list(query.size()) == [bsz, src_len]
        assert attention_mask is not None

        # Actual batch size
        actual_bsz = int(bsz / 2)

        # Seq2seq model that outputs the translation probabilities for the tokens
        translation_preds = self.nmt.generate_t3l(input_ids=query, attention_mask=attention_mask, output_scores=True,
                                                  return_dict_in_generate=True, num_beams=1, num_beam_groups=1,
                                                  do_sample=False,
                                                  decoder_start_token_id=self.tokenizer.bos_token_id,  # self.tokenizer.lang_code_to_id["en_XX"],
                                                  max_length=self.max_output_len)
                                                   # greedy decoding

        # Extract the probability distributions generated by the translator
        probs = translation_preds.scores  # .mean(dim=2)
        # Extract the corresponding sentences
        trans_ids = translation_preds.sequences

        if self.int_trans_writer is not None:
            self.translation_writer(trans_ids)

        # Compute average embeddings with probability distribution
        e_embs = self.expected_embeddings(probs)

        if self.task == 'XNLI':
            # If NLI task re-organise inputs
            e_embs, trans_ids, lm_attention_mask = self.re_organise_batch_xnli(e_embs, trans_ids, actual_bsz)
        elif self.task == 'MLdoc' or self.task == 'MARC' or self.task == 'MultiEurlex':
            # If NLI task re-organise inputs
            e_embs, trans_ids, lm_attention_mask = self.re_organise_batch(e_embs, trans_ids, actual_bsz)

        # Final LM for text classification
        bos_token_embed = self.lm_emb_matrix[self.tokenizer.bos_token_id, :]
        pad_token_embed = self.lm_emb_matrix[self.tokenizer.pad_token_id]
        outputs = self.lm(inputs_embeds=e_embs, attention_mask=lm_attention_mask, pad_token_embed=pad_token_embed,
                          bos_token_id=self.tokenizer.bos_token_id, bos_token_embed=bos_token_embed,
                          pred_ids=trans_ids)
        # Prediction with trans ids
        # outputs = self.lm(trans_ids[:, 1:].int(), attention_mask=lm_attention_mask,
        #                   bos_token_id=self.tokenizer.bos_token_id)

        if self.int_results_writer is not None:
            self.results_writer(outputs.logits)
        return outputs.logits

    def expected_embeddings(self, probabilities):
        """
        Calculate the expected input embeddings for the LM model with the generated probability distributions.
        :param probabilities: The probability distributions generated by the NMT model.
        :return: Expected embeddings (torch.tensor)
        """

        # Obtain sizes
        batch_size, max_seq_len, vocab_size = probabilities.size()
        probabilities = torch.softmax(probabilities, dim=-1)

        # Obtain embeding_dimensions
        emb_vocab_size, emb_dim = self.lm_emb_matrix.size()

        # Calculate the expectation
        probabilities = probabilities.view(-1, vocab_size)
        expected_embs = probabilities.mm(self.lm_emb_matrix)
        expected_embs = expected_embs.view(-1, max_seq_len, emb_dim)

        return expected_embs

    def re_organise_batch_xnli(self, embeddings, ids, actual_bsz):
        """
        Re-organise the batch of sentences for the classification task.
        E.g. Natural Language Inference -> Concatenate Hypothesis and premise with sepparation token before
        feeding into the language model for classification
        :param embeddings: sequence of embeddings
        :param ids: sequence of ids
        :return: re_organised objects
        """

        # Embeddings
        _, seq_size, _ = embeddings.size()
        emb_premise = embeddings[:actual_bsz, :, :]
        emb_hypothesis = embeddings[actual_bsz:, :, :]
        # Add class_token_embedding and sep_token_embedding
        new_batch_size = emb_premise.size()[0]
        emb_matrix = self.lm_emb_matrix.to(embeddings.device)
        class_emb = emb_matrix[self.tokenizer.cls_token_id, :]
        bos_emb = emb_matrix[self.tokenizer.bos_token_id, :]
        # lang_emb = emb_matrix[self.tokenizer.lang_code_to_id["en_XX"], :]
        sep_emb = emb_matrix[self.tokenizer.sep_token_id, :]
        # bos_embs = bos_emb.repeat(new_batch_size, 1).unsqueeze(1)
        class_embs = class_emb.repeat(new_batch_size, 1).unsqueeze(1)
        # lang_embs = lang_emb.repeat(new_batch_size, 1).unsqueeze(1)
        # sep_embs = sep_emb.repeat(new_batch_size, 1).unsqueeze(1)
        embeddings_out = torch.cat((class_embs, emb_premise, emb_hypothesis), dim=1)
        # embeddings_out = torch.cat((class_embs, emb_premise, sep_embs, emb_hypothesis, sep_embs), dim=1)
        # embeddings_out = torch.cat((class_embs, bos_embs, emb_premise, lang_embs, sep_embs, bos_embs, emb_hypothesis,
        #                             lang_embs, sep_embs), dim=1)
        # embeddings_out = torch.cat((emb_premise, emb_hypothesis), dim=1)
        # print(embeddings_out.size())
        # embeddings_out = torch.ones((actual_bsz, seq_size*2 + 3, 768), device=embeddings.device)

        # Ids
        ids_premise = ids[:actual_bsz, :]
        # ids_bos = ids_premise[:, 0].unsqueeze(1)
        ids_premise_cp = ids_premise[:, 1:]
        ids_hypothesis = ids[actual_bsz:, 1:]  # Remove the eos token
        # Add class_token_id and sep_token_id
        id_class_vec = [self.tokenizer.cls_token_id] * new_batch_size
        # ids_class = torch.tensor(id_class_vec, device=embeddings.device).unsqueeze(1)  # .cuda()
        ids_class = embeddings.new_tensor(id_class_vec).unsqueeze(1)
        # id_lang_vec = [self.tokenizer.lang_code_to_id["en_XX"]] * new_batch_size
        # ids_class = torch.tensor(id_class_vec, device=embeddings.device).unsqueeze(1)  # .cuda()
        # ids_lang = embeddings.new_tensor(id_lang_vec).unsqueeze(1)
        id_sep_vec = [self.tokenizer.sep_token_id] * new_batch_size
        # ids_sep = torch.tensor(id_sep_vec, device=embeddings.device).unsqueeze(1)  # .cuda()
        # ids_sep = embeddings.new_tensor(id_sep_vec).unsqueeze(1)
        ids_out = torch.cat((ids_class, ids_premise_cp, ids_hypothesis), dim=1)
        # ids_out = torch.cat((ids_bos, ids_class, ids_premise_cp, ids_sep, ids_hypothesis, ids_sep), dim=1)  # .cuda()
        # print(self.tokenizer.decode(ids_out[0]))
        # ids_out = torch.cat((ids_bos, ids_class, ids_premise_cp, ids_lang, ids_sep,
        #                      ids_hypothesis, ids_lang, ids_sep), dim=1)  # .cuda()
        # ids_out = torch.cat((ids_bos, ids_premise_cp, ids_hypothesis), dim=1)  # .cuda()

        # Create attention mask
        att_bsz, att_seq_len, _ = embeddings_out.size()
        attention_mask = torch.ones((att_bsz, att_seq_len), device=embeddings.device)  # .cuda()
        attention_mask.masked_fill_(ids_out == self.tokenizer.pad_token_id, 0)

        return (embeddings_out, ids_out, attention_mask)

    def re_organise_batch(self, embeddings, ids, actual_bsz):
        """
        Re-organise the batch of sentences for the classification task.
        Only 1 sentence, no sentence pair.
        :param embeddings: sequence of embeddings
        :param ids: sequence of ids
        :return: re_organised objects
        """

        # Embeddings
        batch_size, seq_size, _ = embeddings.size()
        emb_sen = embeddings
        # Add class_token_embedding and sep_token_embedding
        emb_matrix = self.lm_emb_matrix.to(embeddings.device)
        class_emb = emb_matrix[self.tokenizer.cls_token_id, :]
        # bos_embs = bos_emb.repeat(new_batch_size, 1).unsqueeze(1)
        class_embs = class_emb.repeat(batch_size, 1).unsqueeze(1)
        # lang_embs = lang_emb.repeat(new_batch_size, 1).unsqueeze(1)
        # sep_embs = sep_emb.repeat(new_batch_size, 1).unsqueeze(1)
        embeddings_out = torch.cat((class_embs, emb_sen), dim=1)
        # embeddings_out = torch.cat((class_embs, emb_premise, sep_embs, emb_hypothesis, sep_embs), dim=1)
        # embeddings_out = torch.cat((class_embs, bos_embs, emb_premise, lang_embs, sep_embs, bos_embs, emb_hypothesis,
        #                             lang_embs, sep_embs), dim=1)
        # embeddings_out = torch.cat((emb_premise, emb_hypothesis), dim=1)
        # print(embeddings_out.size())
        # embeddings_out = torch.ones((actual_bsz, seq_size*2 + 3, 768), device=embeddings.device)

        # Ids
        ids_sen = ids[:,1:]
        # ids_bos = ids_premise[:, 0].unsqueeze(1)
        # Add class_token_id and sep_token_id
        id_class_vec = [self.tokenizer.cls_token_id] * batch_size
        # ids_class = torch.tensor(id_class_vec, device=embeddings.device).unsqueeze(1)  # .cuda()
        ids_class = embeddings.new_tensor(id_class_vec).unsqueeze(1)
        # id_lang_vec = [self.tokenizer.lang_code_to_id["en_XX"]] * new_batch_size
        # ids_class = torch.tensor(id_class_vec, device=embeddings.device).unsqueeze(1)  # .cuda()
        # ids_lang = embeddings.new_tensor(id_lang_vec).unsqueeze(1)
        # ids_sep = torch.tensor(id_sep_vec, device=embeddings.device).unsqueeze(1)  # .cuda()
        # ids_sep = embeddings.new_tensor(id_sep_vec).unsqueeze(1)
        ids_out = torch.cat((ids_class, ids_sen), dim=1)
        # ids_out = torch.cat((ids_bos, ids_class, ids_premise_cp, ids_sep, ids_hypothesis, ids_sep), dim=1)  # .cuda()
        # print(self.tokenizer.decode(ids_out[0]))
        # ids_out = torch.cat((ids_bos, ids_class, ids_premise_cp, ids_lang, ids_sep,
        #                      ids_hypothesis, ids_lang, ids_sep), dim=1)  # .cuda()
        # ids_out = torch.cat((ids_bos, ids_premise_cp, ids_hypothesis), dim=1)  # .cuda()

        # if seq_size == self.max_input_len -1:
        #     embeddings_out = embeddings_out[:, :-1, :]
        #     ids_out = ids_out[:, :-1]
        # Create attention mask
        att_bsz, att_seq_len, _ = embeddings_out.size()
        attention_mask = torch.ones((att_bsz, att_seq_len), device=embeddings.device)  # .cuda()
        attention_mask.masked_fill_(ids_out == self.tokenizer.pad_token_id, 0)

        return (embeddings_out, ids_out, attention_mask)

    def freeze_params(self):

        if self.freeze_strategy == 'fix_nmt':
            print('Fixed NMT!')
            for param in self.nmt.parameters():
                param.requires_grad = False
        elif self.freeze_strategy == 'fix_lm':
            print('Fixed LM!')
            for param in self.lm.parameters():
                param.requires_grad = False
        elif self.freeze_strategy == 'fix_nmt_dec_lm_enc':
            print('Fixed NMT decoder and LM encoder!')
            for name, param in self.nmt.named_parameters():
                if 'decoder' not in name:
                    param.requires_grad = False
            for name, param in self.lm.named_parameters():
                if 'encoder' not in name:
                    param.requires_grad = False
        elif self.freeze_strategy == 'train_all':
            print('Train alL!')

    def create_translation_writer(self, int_trans_path):

        if int_trans_path:
            dir_path = "/".join(int_trans_path.split('/')[:-1])
            os.makedirs(dir_path, exist_ok=True)
            self.int_trans_writer = open(int_trans_path, 'w', encoding='utf-8')
            self.int_results_writer = open(f"{dir_path}/pred_labels.txt", "w", encoding="utf-8")
        else:
            self.int_trans_writer = None
            self.int_results_writer = None


    def translation_writer(self, trans_ids):

        batch_size, max_seq_len = trans_ids.size()
        real_batch_size = int(batch_size / 2)
        trans_str = self.tokenizer.batch_decode(trans_ids, skip_special_tokens=True)
        trans_str_prem = trans_str[:real_batch_size]
        trans_str_hyp = trans_str[real_batch_size:]
        for i in range(len(trans_str_prem)):
            self.int_trans_writer.write(f"PREMISE: {trans_str_prem[i]}\tHYPOTHESIS: {trans_str_hyp[i]}\n")

    def results_writer(self, logits):

        preds = logits.argmax(dim=-1).clone().detach().cpu().tolist()
        for pred in preds:
            value_pred = XNLI_ID2LABEL[pred]
            self.int_results_writer.write(value_pred+'\n')
