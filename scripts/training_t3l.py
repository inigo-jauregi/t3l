# Script to train the joint model
import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, MBartTokenizer
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

from t3l.nmt_plus_lm import NmtPlusLmForTextClassification, NmtPlusLmForTextClassificationConfig
from t3l.language_code_mapping import LANGUAGE_CODE_MAPPING, LANG2LANG
from t3l.evaluation_metrics import text_classification_metrics, multi_label_text_classification_metrics
from t3l.custom_data_loaders import XNLIDataset_tt, MLDocCorpus_tt, MultiEurlexCorpus_tt, \
    XNLI_LABEL2ID, MLDOC_LABEL2ID, MULTIEURLEX_LEVEL_1_LABEL2ID


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class JoinTranslationTransferLearning(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params
        self.hparams['params'] = params
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer, use_fast=False)
        if isinstance(self.tokenizer, MBartTokenizer):
            # print(self.tokenizer._src_lang)
            # print(self.tokenizer.tgt_lang)
            if self.args.src in LANG2LANG:
                source_lang = LANG2LANG[self.args.src]
            else:
                source_lang = self.args.src
            self.src_lan_code = LANGUAGE_CODE_MAPPING[source_lang]
            if self.args.tgt in LANG2LANG:
                target_lang = LANG2LANG[self.args.tgt]
            else:
                target_lang = self.args.tgt
            self.tgt_lan_code = LANGUAGE_CODE_MAPPING[target_lang]
            self.tokenizer._src_lang = self.src_lan_code
            self.tokenizer.tgt_lang = self.tgt_lan_code
            # print(self.tokenizer._src_lang)
            # print(self.tokenizer.tgt_lang)

        if not hasattr(self.args, 'task'):
            self.args.task = 'XNLI'
        # Load the model NMT plus LM
        self.model = NmtPlusLmForTextClassification(self.tokenizer, self.args.model_seq2seq_path,
                                                    self.args.model_lm_path,
                                                    self.args.max_input_len, self.args.max_output_len,
                                                    self.args.freeze_strategy, self.args.task)

        # # Test
        # if self.args.test:
        #     path_save = os.path.join(self.args.save_dir, self.args.save_prefix, "test_translations.txt")
        #     self.test_trans_writer = open(path_save, 'w')

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

    def forward(self, input_ids, attention_mask, output_label):
        labels = output_label.clone()
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            test_trans=self.test_trans_writer if False else None
        )
        lm_logits = outputs

        if self.args.label_smoothing == 0:
            assert lm_logits.shape[-1] == self.model.num_out_classes  # It has to be the same as the output class
            if self.args.task != "MultiEurlex":
                # Same behavior as modeling_bart.py, besides ignoring pad_token_id
                ce_loss_fct = torch.nn.CrossEntropyLoss()
                loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            else:
                bces_loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = bces_loss_fct(lm_logits, labels)
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=self.tokenizer.pad_token_id
            )

        # Metrics
        if args.task != 'MultiEurlex':
            acc, prec, rec, f1 = text_classification_metrics(lm_logits.detach(), labels)
            return [loss, acc.detach(), prec.detach(), rec.detach(), f1.detach()]
        else:
            acc, mrp, rec, f1 = multi_label_text_classification_metrics(lm_logits.detach(), labels)
            return [loss, acc.detach(), mrp.detach(), rec.detach(), f1.detach()]

    def training_step(self, batch, batch_nb):
        output = self.forward(*batch)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        if self.args.task != 'MultiEurlex':
            tensorboard_logs = {'train_loss': loss, 'lr': lr,
                                'input_size': batch[0].numel(),
                                'output_size': batch[1].numel(),
                                'mem': torch.cuda.memory_allocated(
                                    loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0,
                                'accuracy': output[1],
                                'precision': output[2],
                                'recall': output[3],
                                'f1_score': output[4]}
            self.log("my_lr", lr, prog_bar=True, on_step=True)
            self.log("accuracy", output[1].clone().detach(), prog_bar=True, on_step=True)
            self.log("loss", loss.clone().detach())
            return {'loss': loss, 'accuracy': output[1], 'precision': output[2], 'recall': output[3], 'f1_score': output[4],
                    'log': tensorboard_logs}
        else:
            tensorboard_logs = {'train_loss': loss.detach(), 'lr': lr,
                                'input_size': batch[0].numel(),
                                'output_size': batch[1].numel(),
                                'mem': torch.cuda.memory_allocated(
                                    loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0,
                                'accuracy': output[1],
                                'mrp': output[2],
                                'recall': output[3],
                                'f1_score': output[4]}
            self.log("my_lr", lr, prog_bar=True, on_step=True)
            self.log("accuracy", output[1], prog_bar=True, on_step=True)
            self.log("mRP", output[2], prog_bar=True, on_step=True)
            return {'loss': loss, 'accuracy': output[1], 'mrp': output[2], 'recall': output[3],
                    'f1_score': output[4],
                    'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        if self.args.task != 'MultiEurlex':
            return {'vloss': vloss, 'vaccuracy': outputs[1], 'vprecision': outputs[2], 'vrecall': outputs[3],
                    'vf1_score': outputs[4]}
        else:
            return {'vloss': vloss, 'vaccuracy': outputs[1], 'vmrp': outputs[2], 'vrecall': outputs[3],
                    'vf1_score': outputs[4]}

    def validation_epoch_end(self, outputs):
        for name, p in self.model.named_parameters():
            p.requires_grad = True
        # Freeze params according to freeze strategy
        self.model.freeze_params()

        if self.args.task != 'MultiEurlex':
            names = ['vloss', 'vaccuracy', 'vprecision', 'vrecall', 'vf1_score']
        else:
            names = ['vloss', 'vaccuracy', 'vmrp', 'vrecall', 'vf1_score']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            # if self.trainer.accelerator_connector.use_ddp:
            #     torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
            #     metric /= self.trainer.world_size
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        self.log("avg_val_accuracy", logs["vaccuracy"])
        self.log("avg_val_loss", logs['vloss'])
        if self.args.task == 'MultiEurlex':
            self.log("avg_val_mrp", logs['vmrp'])
        print(logs)
        if self.args.task != 'MultiEurlex':
            return {'avg_val_loss': logs['vloss'], 'avg_accuracy': logs['vaccuracy'], 'avg_precision': logs['vprecision'],
                    'avg_recall': logs['vrecall'], 'avg_f1_score': logs['vf1_score'], 'log': logs, 'progress_bar': logs}
        else:
            return {'avg_val_loss': logs['vloss'], 'avg_accuracy': logs['vaccuracy'],
                    'avg_mrp': logs['vmrp'],
                    'avg_recall': logs['vrecall'], 'avg_f1_score': logs['vf1_score'], 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        print(result)

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(self.model.parameters(), lr=self.args.lr, scale_parameter=False, relative_step=False)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        if self.args.debug:
            return optimizer  # const LR
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_steps = self.args.dataset_size * self.args.epochs / num_gpus / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader

        if self.args.task == 'XNLI':
            dataset = XNLIDataset_tt(hf_dataset=self.hf_datasets[split_name], tokenizer=self.tokenizer,
                                     max_input_len=self.args.max_input_len, max_output_len=self.args.max_output_len,
                                     label_dict=self.label_dict)
            selec_collate_fn = XNLIDataset_tt.collate_fn
        elif self.args.task == 'MLdoc':
            dataset = MLDocCorpus_tt(hf_dataset=self.hf_datasets[split_name], tokenizer=self.tokenizer,
                                  max_input_len=self.args.max_input_len, label_dict=self.label_dict)
            selec_collate_fn = MLDocCorpus_tt.collate_fn
        elif self.args.task == 'MultiEurlex':
            dataset = MultiEurlexCorpus_tt(hf_dataset=self.hf_datasets[split_name], tokenizer=self.tokenizer,
                                           max_input_len=self.args.max_input_len, label_dict=self.label_dict)
            selec_collate_fn = MultiEurlexCorpus_tt.collate_fn

        # sampler = torch.utils.data.distributed.DistributedSampler(dataset,
        #                                                           shuffle=is_train) if \
        #     self.trainer.accelerator_connector.use_ddp else None
        sampler = None
        # Shuffle or not
        if is_train and (sampler is None):
            is_shuffle = True
        else:
            is_shuffle = False
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=is_shuffle,
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=selec_collate_fn)

    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'validation', is_train=False)
        return self.val_dataloader_object

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--task", type=str, required=True, choices=['XNLI', 'MARC', 'MLdoc', 'MultiEurlex'],
                            help='Task (and dataset) to learn.')
        parser.add_argument("--train_data", type=str, required=True, help='Path to training data')
        parser.add_argument("--validation_data", type=str, required=True, help='Path to validation data')
        parser.add_argument("--test_data", type=str, required=True, help='Path to testing data')
        parser.add_argument("--src", type=str, required=True, help='Source language')
        parser.add_argument("--tgt", type=str, required=True, help='Target language')
        parser.add_argument("--save_dir", type=str, default='summarization')
        parser.add_argument("--save_prefix", type=str, default='test')
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--gpus", type=int, default=0,
                            help="Number of gpus. 0 for CPU")
        parser.add_argument("--warmup", type=int, default=0, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--max_output_len", type=int, default=170,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=170,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--int_trans_name", type=str, default=None, help="Test only, no training")
        parser.add_argument("--model_seq2seq_path", type=str, default='../pretrained_lm/sshleifer-tiny-mbart',
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--model_lm_path", type=str, default='../pretrained_lm/sshleifer-tiny-mbart',
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--freeze_strategy", type=str,
                            choices=['fix_nmt', 'fix_lm', 'fix_nmt_dec_lm_enc', 'train_all'], default='fix_nmt',
                            help="Modules fixing strategy.")
        parser.add_argument("--tokenizer", type=str, default='../pretrained_lm/mrm8488-mbart-large-finetuned'
                                                             '-opus-es-en-translation')
        parser.add_argument("--progress_bar", type=int, default=1, help="Progress bar. Good for printing")
        parser.add_argument("--precision", type=int, default=32, help="Double precision (64), full precision (32) "
                                                                      "or half precision (16). Can be used on CPU, "
                                                                      "GPU or TPUs.")
        parser.add_argument("--amp_backend", type=str, default='apex', help="The mixed precision backend to "
                                                                              "use ('native' or 'apex')")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--from_pretrained", type=str, default=None,
                            help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
        parser.add_argument("--attention_mode", type=str, default='sliding_chunks', help="Longformer attention mode")
        parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--adafactor", action='store_true', help="Use adafactor optimizer")

        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.from_pretrained is not None:
        model = JoinTranslationTransferLearning.load_from_checkpoint(args.from_pretrained)
    else:
        model = JoinTranslationTransferLearning(args)

    # Assign labels manually -> For XNLI
    if args.task == 'XNLI':
        label_dict = XNLI_LABEL2ID
    elif args.task == 'MLdoc':
        label_dict = MLDOC_LABEL2ID
    elif args.task == 'MultiEurlex':
        label_dict = MULTIEURLEX_LEVEL_1_LABEL2ID

    model.hf_datasets = {'train': args.train_data,
                         'validation': args.validation_data,
                         'test': args.test_data}
    print(model.hf_datasets)
    model.label_dict = label_dict

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    if args.task != 'MultiEurlex':
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
            filename='check-{epoch:02d}-{avg_val_accuracy:.4f}',
            save_top_k=1,
            verbose=True,
            monitor='avg_val_accuracy',
            mode='max',
            every_n_epochs=1
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
            filename='check-{epoch:02d}-{avg_val_mrp:.4f}',
            save_top_k=1,
            verbose=True,
            monitor='avg_val_mrp',
            mode='max',
            every_n_epochs=1
        )

    print(args)

    args.dataset_size = 2500  # hardcode dataset size. Needed to compute number of steps for the lr scheduler

    trainer = pl.Trainer(gpus=args.gpus, accelerator=None,  # 'ddp' if torch.cuda.is_available() else None,
                         track_grad_norm=-1,
                         max_epochs=args.epochs if not args.debug else 100,
                         max_steps=None if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.grad_accum,
                         val_check_interval=args.val_every if not args.debug else 1,
                         num_sanity_val_steps=2 if not args.debug else 0,
                         check_val_every_n_epoch=1 if not args.debug else 1,
                         logger=logger,
                         callbacks=checkpoint_callback if not args.disable_checkpointing else False,
                         progress_bar_refresh_rate=args.progress_bar,
                         precision=args.precision,
                         amp_backend=args.amp_backend, amp_level='apex',
                         resume_from_checkpoint=args.resume_ckpt,
                         )
    model.model.create_translation_writer(args.int_trans_name)
    if not args.test:
        trainer.fit(model)
    else:
        trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = JoinTranslationTransferLearning.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
