# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os
from collections import OrderedDict
import torch

import argparse

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    iterators,
    iterators_dtn,
    FairseqDataset
)

from . import FairseqTask, register_task

class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        # print("values: {}".format(values))
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

@register_task('translation_dtn')
class TranslationDtnTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

        # dtn
        parser.add_argument('--restore-from-trans', action='store_true',
                            help='restore from transformer')
        parser.add_argument('--random-select', action='store_true',
                            help='random select')
        parser.add_argument('--random-select-factor', default=0.7, type=float,
                            help='random select factor')
        parser.add_argument('--kd', action='store_true',
                            help='knowledge distill')
        parser.add_argument('--kd-lambda', default=0.1, type=float,
                            help='knowledge distill coefficient')

        # fmt: on

    @staticmethod
    def load_pretrained_model(path, src_dict_path, tgt_dict_path, arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        args = utils.override_model_args(args, arg_overrides)
        src_dict = Dictionary.load(src_dict_path)
        tgt_dict = Dictionary.load(tgt_dict_path)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        task = TranslationDtnTask(args, src_dict, tgt_dict)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)), domains=args.domains)
        tgt_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)), domains=args.domains)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data
        train_domains = self.args.train_domains
        valid_domains = self.args.valid_domains
        test_domains = self.args.test_domains

        for dk, data_path in enumerate(data_paths):
            src, tgt = self.args.source_lang, self.args.target_lang
            for k in itertools.count():
                if split=='train':
                    for domain in train_domains:
                        split_k = split
                        data_path_exp = os.path.join(data_path, domain)
                        # infer langcode
                        if split_exists(split_k, src, tgt, src, data_path_exp):
                            prefix = os.path.join(data_path_exp, '{}.{}-{}.'.format(split_k, src, tgt))
                        elif split_exists(split_k, tgt, src, src, data_path_exp):
                            prefix = os.path.join(data_path_exp, '{}.{}-{}.'.format(split_k, tgt, src))
                        else:
                            if k > 0 or dk > 0:
                                break
                            else:
                                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path_exp))

                        src_datasets.append((domain, indexed_dataset(prefix + src, self.src_dict)))
                        tgt_datasets.append((domain, indexed_dataset(prefix + tgt, self.tgt_dict)))

                        print('| {} {} {} examples'.format(data_path_exp, domain, len(src_datasets[-1][-1])))

                if split=='valid':
                    for domain in valid_domains:
                        split_k = split
                        data_path_exp = os.path.join(data_path, domain)
                        # infer langcode
                        if split_exists(split_k, src, tgt, src, data_path_exp):
                            prefix = os.path.join(data_path_exp, '{}.{}-{}.'.format(split_k, src, tgt))
                        elif split_exists(split_k, tgt, src, src, data_path_exp):
                            prefix = os.path.join(data_path_exp, '{}.{}-{}.'.format(split_k, tgt, src))
                        else:
                            if k > 0 or dk > 0:
                                break
                            else:
                                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path_exp))

                        src_datasets.append((domain, indexed_dataset(prefix + src, self.src_dict)))
                        tgt_datasets.append((domain, indexed_dataset(prefix + tgt, self.tgt_dict)))

                        print('| {} {} {} examples'.format(data_path_exp, domain, len(src_datasets[-1][-1])))

                if split=='test':
                    for domain in test_domains:
                        split_k = split
                        data_path_exp = os.path.join(data_path, domain)
                        # infer langcode
                        if split_exists(split_k, src, tgt, src, data_path_exp):
                            prefix = os.path.join(data_path_exp, '{}.{}-{}.'.format(split_k, src, tgt))
                        elif split_exists(split_k, tgt, src, src, data_path_exp):
                            prefix = os.path.join(data_path_exp, '{}.{}-{}.'.format(split_k, tgt, src))
                        else:
                            if k > 0 or dk > 0:
                                break
                            else:
                                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path_exp))

                        src_datasets.append((domain, indexed_dataset(prefix + src, self.src_dict)))
                        tgt_datasets.append((domain, indexed_dataset(prefix + tgt, self.tgt_dict)))

                        print('| {} {} {} examples'.format(data_path_exp, domain, len(src_datasets[-1][-1])))

                break
        assert len(src_datasets) == len(tgt_datasets)
        if hasattr(self.args, 'gen_subset'):
            assert len(src_datasets)==1, 'Need one domain in test'
            self.datasets[split] = LanguagePairDataset(
                src_datasets[0][1], src_datasets[0][1].sizes, self.src_dict,
                tgt_datasets[0][1], tgt_datasets[0][1].sizes, self.tgt_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions, )
        else:
            src_datasets = OrderedDict(src_datasets)
            tgt_datasets = OrderedDict(tgt_datasets)

            self.datasets[split] = OrderedDict((key, LanguagePairDataset(
                src_dataset, src_dataset.sizes, self.src_dict,
                tgt_datasets[key], tgt_datasets[key].sizes, self.tgt_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,)) for (key, src_dataset) in src_datasets.items())

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset
        if split not in self.datasets:
            raise KeyError('Dataset not loaded: ' + split)

        return self.datasets[split]

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, OrderedDict)

        # get indices ordered by example size
        indices = OrderedDict()
        for domain in dataset.keys():
            with data_utils.numpy_seed(seed):
                indices[domain]=dataset[domain].ordered_indices()

        batch_sampler = OrderedDict()
        collater = OrderedDict()
        # filter examples that are too large
        for domain in dataset.keys():

            indices[domain] = data_utils.filter_by_size(
                indices[domain], dataset[domain].size, max_positions, raise_exception=(not ignore_invalid_inputs),
            )

            # create mini-batches with given size constraints
            batch_sampler[domain] = data_utils.batch_by_size(
                indices[domain], dataset[domain].num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
            collater[domain] = dataset[domain].collater
        # return a reusable, sharded iterator
        return iterators_dtn.EpochBatchIteratorDtn(
            dataset=dataset,
            collate_fn=collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,)

    def get_batch_iterator_valid(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        indices = data_utils.filter_by_size(
            indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
        )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
        )


    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions
        return criterions.build_criterion(args, self)


    def build_generator(self, args):
        if args.score_reference:
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_generator_dtn import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False, domain=None, num_updates=None):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        loss, sample_size, logging_output = criterion(model, sample, domain=domain, num_updates=num_updates)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, domain=None, num_updates=None):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, domain=domain, num_updates=num_updates)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None, domain=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens, domain=domain)

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
