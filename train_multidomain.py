#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import itertools
import os
import math
import random
from collections import OrderedDict
import numpy as np

import torch

from fairseq import distributed_utils, options, progress_bar, tasks, utils, bleu
from fairseq.data import iterators
from fairseq.trainer_multidomain import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.utils import import_user_module

import subprocess

def main(args, init_distributed=False):
    import_user_module(args)

    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)
    assert args.task == 'translation_multidomain'

    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    # Initialize distributed training (after data loading)
    if init_distributed:
        import socket
        args.distributed_rank = distributed_utils.distributed_init(args)
        print('| initialized host {} as rank {}'.format(socket.gethostname(), args.distributed_rank))

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )

    assert len(args.train_domains)==1
    TRAIN_TAG = args.train_domains[0]
    dummy_batch = task.dataset('train')[TRAIN_TAG].get_dummy_batch(args.max_tokens, max_positions)
    oom_batch = task.dataset('train')[TRAIN_TAG].get_dummy_batch(1, max_positions)

    trainer = Trainer(args, task, model, criterion, dummy_batch, oom_batch)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset)[TRAIN_TAG],
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    )

    # Load the latest checkpoint if one is available
    if not load_checkpoint(args, trainer, epoch_itr):
        trainer.dummy_train_step([dummy_batch])
    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()

    valid_losses = [None]
    valid_bleus = [None]

    valid_subsets = args.valid_subset.split(',')
    valid_select = args.valid_select[0]

    with open(os.path.join(args.tensorboard_logdir, 'args_log.txt'), 'w') as f:
        for k in args.__dict__.keys():
            f.write("'%s':'%s', \n" % (k, args.__dict__[k]))

    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses, valid_bleus = validate(args, trainer, task, epoch_itr, valid_subsets)
            save_checkpoint(args, trainer, epoch_itr, valid_losses, valid_bleus, valid_select)

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[valid_select])

    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))

def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
            if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    valid_select = args.valid_select[0]
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.validate_interval_updates > 0 and num_updates % args.validate_interval_updates == 0 and num_updates > 0:
            valid_losses, valid_bleus = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses, valid_bleus, valid_select)

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats

def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = OrderedDict()
    valid_bleus = OrderedDict()
    valid_select = task.args.valid_select[0]
    assert len(subsets) == 1

    for subset in subsets:
        # Initialize data iterator
        valid_loss_all = []
        valid_nll_loss_all =[]
        valid_bleu_all =[]

        for k in ['valid_loss', 'valid_nll_loss', 'valid_bleu']:
            meter = trainer.get_meter(k + '_all')
            meter.reset()

        for domain, data_valid in task.dataset(subset).items():

            itr = task.get_batch_iterator(
                dataset=data_valid,
                max_tokens=args.max_tokens,
                max_sentences=args.max_sentences_valid,
                max_positions=utils.resolve_max_positions(
                    task.max_positions(),
                    trainer.get_model().max_positions(),
                ),
                ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=8,
                seed=args.seed,
                num_shards=args.distributed_world_size,
                shard_id=args.distributed_rank,
                num_workers=args.num_workers,
            ).next_epoch_itr(shuffle=False)

            progress = progress_bar.build_progress_bar(
                args, itr, epoch_itr.epoch,
                prefix='valid on \'{}\' subset \'{}\' domain'.format(subset, domain),
                no_progress_bar='simple'
            )
            # reset validation loss meters
            for k in ['valid_loss', 'valid_nll_loss','valid_bleu']:
                meter = trainer.get_meter(k + '_' + domain)
                meter.reset()

            extra_meters = collections.defaultdict(lambda: AverageMeter())

            src_target_hypo_strs = []
            for sample in progress:
                log_output, src_target_hypo_str = trainer.valid_step(sample, domain=domain, trainer_scorer=None)
                src_target_hypo_strs.extend(src_target_hypo_str)

                for k, v in log_output.items():
                    if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                        continue
                    extra_meters[k].update(v)
            src_target_hypo_strs_filter = []
            for sents in src_target_hypo_strs:
                for sent in sents:
                    if sent is None or len(sent)==0:
                        continue
                    src_target_hypo_strs_filter.append(sent)

            src_target_hypo_strs_filter=sorted(src_target_hypo_strs_filter, key=lambda elem: int(elem[0]), reverse=False)
            if args.valid_decoding_path is not None:
                with open(os.path.join(args.valid_decoding_path, domain, 'decoding_{}.txt'.format(args.distributed_rank)), 'w') as f:
                    for sent in src_target_hypo_strs_filter:
                        if len(sent) == 0:
                            continue
                        f.write(sent[-1] + '\n')

            num_ref = args.num_ref[domain]
            ref_path = []
            for i in range(int(num_ref)):
                ref_path.append(os.path.join(args.valid_decoding_path, domain, 'valid.tok.'+args.target_lang+str(i)))

            valid_decoding_path = os.path.join(args.valid_decoding_path, domain, 'decoding_{}.txt'.format(args.distributed_rank))

            with open(valid_decoding_path) as out_file:
                out_file.seek(0)
                res = subprocess.check_output(
                    'perl %s/multi-bleu.perl %s' % (args.multi_bleu_path, ' '.join(ref_path)),
                    stdin=out_file, shell=True
                ).decode("utf-8")

            trainer.get_meter('valid_bleu_' +domain).update(float(res.split(',')[0].split('=')[1]), 1.0)

            # log validation stats
            stats = get_valid_stats(trainer, domain=domain, valid_select = valid_select)

            for k in ['loss', 'nll_loss', 'bleu']:
                stats[k] = stats[k].avg
            for k, meter in extra_meters.items():
                stats[k] = meter.avg

            progress.print(stats, tag=os.path.join(subset, domain), step=trainer.get_num_updates())
            valid_losses.update({domain: stats['loss']})
            valid_bleus.update({domain: stats['bleu']})

            valid_loss_all.append(stats['loss'])
            valid_nll_loss_all.append(stats['nll_loss'])
            valid_bleu_all.append(stats['bleu'])

        trainer.get_meter('valid_loss_all').update(np.mean(valid_loss_all), 1.0)
        trainer.get_meter('valid_nll_loss_all').update(np.mean(valid_nll_loss_all), 1.0)
        trainer.get_meter('valid_bleu_all').update(np.mean(valid_bleu_all), 1.0)

        stats = get_valid_stats(trainer, domain='all', valid_select=valid_select)

        for k in ['loss', 'nll_loss', 'bleu']:
            stats[k] = stats[k].avg

        progress = progress_bar.build_progress_bar(
            args, [0], epoch_itr.epoch,
            prefix='valid on \'{}\' subset \'{}\' domain'.format(subset, 'all'),
            no_progress_bar='simple'
        )

        progress.print(stats, tag=os.path.join(subset, 'all'), step=trainer.get_num_updates())
        valid_losses.update({'all': stats['loss']})
        valid_bleus.update({'all': stats['bleu']})

    return valid_losses, valid_bleus

def get_valid_stats(trainer, domain=None, valid_select=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss_' + domain)
    stats['bleu'] = trainer.get_meter('valid_bleu_' + domain)

    if trainer.get_meter('valid_nll_loss_' + domain).count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss_' + domain)
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']

    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()

    if domain == valid_select:
        if hasattr(save_checkpoint_bleu, 'best_bleu'):
            stats['best_bleu'] = max(save_checkpoint_bleu.best_bleu, stats['bleu'].avg)

    return stats

def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')

def save_checkpoint(args, trainer, epoch_itr, valid_losses, valid_bleus, valid_select, begin=False):
    save_checkpoint_bleu(args, trainer, epoch_itr, valid_losses, valid_bleus, valid_select, begin=begin)

def save_checkpoint_bleu(args, trainer, epoch_itr, valid_losses, valid_bleus, valid_select, begin):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    if begin:
        end_of_epoch = True
    else:
        end_of_epoch = epoch_itr.end_of_epoch()

    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best_bleu.pt'] = (
            valid_select in valid_bleus.keys() and
            (not hasattr(save_checkpoint_bleu, 'best_bleu') or valid_bleus[valid_select] > save_checkpoint_bleu.best_bleu)
    )

    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best_bleu = getattr(save_checkpoint_bleu, 'best_bleu', valid_bleus[valid_select])

    if valid_select in valid_bleus.keys():
        save_checkpoint_bleu.best_bleu = max(valid_bleus[valid_select], prev_best_bleu)

    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
    }
    for domain, bleu_domain in valid_bleus.items():
        extra_state.update({'valid_loss_'+ domain: valid_losses[domain]})
        extra_state.update({'valid_bleu_'+ domain: valid_bleus[domain]})

    if hasattr(save_checkpoint_bleu, 'best_bleu'):
        extra_state.update({'best_bleu': save_checkpoint_bleu.best_bleu})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint(\d+)\.pt')
        for old_chk in checkpoints[args.keep_last_epochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    if os.path.isabs(args.restore_file):
        checkpoint_path = args.restore_file
    else:
        checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path, args.reset_optimizer, args.reset_lr_scheduler,
                                              eval(args.optimizer_overrides))
        if extra_state is not None:
            # replay train iterator to match checkpoint
            if eval(args.finetune):
                extra_state['train_iterator']['iterations_in_epoch'] = 0
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best_bleu' in extra_state:
                del extra_state['best_bleu']
        return True
    else:
        print('| no existing checkpoint found {}'.format(checkpoint_path))
    return False


def load_dataset_splits(task, splits):
    for split in splits:
        task.load_dataset(split, combine=True)

def distributed_main(i, args):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
