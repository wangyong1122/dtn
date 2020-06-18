# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import math

import numpy as np
import torch

from . import data_utils
import copy

from collections import OrderedDict
from . import iterators

class CountingIteratorDtn(object):
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap

    Attributes:
        count (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable):
        self.iterable = iterable
        self.count = 0
        self.itr = iter(self)
        self.len = len(iterable)

    def __len__(self):
        return self.len

    def __iter__(self):
        for x in self.iterable:
            self.count += 1
            yield x

    def __next__(self):
        try:
            data = next(self.itr)
        except StopIteration:
            self.count = 0
            self.itr = iter(self)
            data = next(self.itr)
        return data

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.count < len(self)

    def skip(self, num_to_skip):
        """Fast-forward the iterator by skipping *num_to_skip* elements."""
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        self.len -= num_to_skip
        return self

class EpochBatchIteratorDtn(object):
    """A multi-epoch iterator over a :class:`torch.utils.data.Dataset`.

    Compared to :class:`torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the *num_shards* and *shard_id* arguments

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        collate_fn (callable): merges a list of samples to form a mini-batch
        batch_sampler (~torch.utils.data.Sampler): an iterator over batches of
            indices
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
    """

    def __init__(
        self, dataset, collate_fn, batch_sampler, seed=1, num_shards=1, shard_id=0,
        num_workers=0,
    ):
        assert isinstance(dataset, OrderedDict)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.frozen_batches = OrderedDict()
        for domain in batch_sampler.keys():
            self.frozen_batches[domain] = tuple(batch_sampler[domain])
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers

        self.epoch = 0
        self.count_sum = 0
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        self._supports_prefetch = getattr(dataset[list(batch_sampler.keys())[0]], \
                                                  'supports_prefetch', False)

    def __len__(self):

        return sum(len(frozen_batch) for (domain, frozen_batch) in self.frozen_batches.items())

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
        """
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, fix_batches_to_gpus=fix_batches_to_gpus,
            )
        return self._cur_epoch_itr

    def end_of_epoch(self):
        """Returns whether the most recent epoch iterator has been exhausted"""
        count = 0
        leng = 0
        for domain in self.dataset.keys():
            count += self._cur_epoch_itr[domain].count
            leng += len(self._cur_epoch_itr[domain])
        return not count <leng

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return sum(self._cur_epoch_itr[domain].count for domain in self._cur_epoch_itr.keys())
        elif self._next_epoch_itr is not None:
            return sum(self._next_epoch_itr[domain].count for domain in self._next_epoch_itr.keys())
        return 0

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        domain_count = OrderedDict()
        if self._cur_epoch_itr is not None:
            for domain in self._cur_epoch_itr.keys():
                domain_count[domain] = self._cur_epoch_itr[domain].count
        elif self._next_epoch_itr is not None:
            for domain in self._next_epoch_itr.keys():
                domain_count[domain] = self._next_epoch_itr[domain].count
        else:
            for domain in self.dataset.keys():
                domain_count[domain] = 0
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
            'domain_count': domain_count
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        domain_count = OrderedDict()
        for domain in self.dataset.keys():
            domain_count[domain] = 0

        if itr_pos > 0:
            # fast-forward epoch iterator
            itr = self._get_iterator_for_epoch(self.epoch, state_dict.get('shuffle', True))
            self._next_epoch_itr = OrderedDict()
            for domain in self.dataset.keys():
                if domain_count[domain] < len(itr[domain]):
                    self._next_epoch_itr[domain] = itr[domain].skip(domain_count[domain])

    def _get_iterator_for_epoch(self, epoch, shuffle, fix_batches_to_gpus=False):

        def shuffle_batches(batches, seed):
            # set seed based on the seed and epoch number so that we get
            # reproducible results when resuming from checkpoints
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
            return batches

        # if self._supports_prefetch:
        if self._supports_prefetch:
            batches = copy.deepcopy(self.frozen_batches)

            if shuffle and not fix_batches_to_gpus:
                for domain in self.frozen_batches.keys():
                    batches[domain] = shuffle_batches(list(batches[domain]), self.seed + epoch)
            for domain in self.frozen_batches.keys():
                batches[domain] = list(iterators.ShardedIterator(
                batches[domain], self.num_shards, self.shard_id, fill_value=[]))
                self.dataset[domain].prefetch([i for s in batches[domain] for i in s])

            if shuffle and fix_batches_to_gpus:
                for domain in self.frozen_batches.keys():
                    batches[domain] = shuffle_batches(batches[domain], self.seed + epoch + self.shard_id)

        else:
            batches = OrderedDict()
            if shuffle:
                for domain in self.frozen_batches.keys():
                    batches[domain] = shuffle_batches(list(copy.deepcopy(self.frozen_batches[domain])), self.seed + epoch)
            else:
                for domain in self.frozen_batches.keys():
                    batches[domain] = copy.deepcopy(self.frozen_batches[domain])
            for domain in self.frozen_batches.keys():
                batches[domain] = iterators.ShardedIterator(batches[domain], self.num_shards, self.shard_id, fill_value=[])
        countingiterator = OrderedDict()

        for domain in self.frozen_batches.keys():
            countingiterator[domain] = CountingIteratorDtn(torch.utils.data.DataLoader(
                                                        self.dataset[domain],
                                                        collate_fn=self.collate_fn[domain],
                                                        batch_sampler=batches[domain],
                                                        num_workers=self.num_workers,))
        return countingiterator

class GroupedIteratorDtn(object):
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk
    """

    def __init__(self, iterable, chunk_size):
        self._len = int(math.ceil(len(iterable) / float(chunk_size)))
        self.itr = iterable
        self.chunk_size = chunk_size

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        chunk = []
        try:
            for _ in range(self.chunk_size):
                chunk.append(next(self.itr))
        except StopIteration as e:
            if len(chunk) == 0:
                raise e
        return chunk
