# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict

from fairseq import utils
from fairseq.tasks.translation_dtn import TranslationDtnTask
import torch.nn as nn
import os
import torch
from torch.serialization import default_restore_location

from . import FairseqModel, register_model, register_model_architecture
from fairseq.modules import DtnBlock

from .transformer import (
    base_architecture,
    Embedding,
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)

@register_model('transformer_dtn')
class TransformerDtnModel(FairseqModel):
    """Train Transformer models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from TransformerModel and assume that all language
    pairs use a single Transformer architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args:

    """

    def __init__(self, encoder, decoder, dtn_blocks, args, distill_models):
        super().__init__(encoder, decoder)
        self.dtn_blocks = dtn_blocks
        self.args = args
        self.distill_models = distill_models

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        assert isinstance(task, TranslationDtnTask)

        # make sure all arguments are present in older models
        base_dtn_architecture(args)
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        domains = args.domains

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)
        # encoders/decoders for each language
        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

        dtn_blocks = nn.ModuleDict()
        for domain in domains:
            dtn_blocks[domain] = DtnBlock(args)

        distill_models = None
        if args.kd:
            distill_models = dict()
            for domain in domains:
                distill_models[domain] = cls.build_distill_model(args, task, domain)

        return TransformerDtnModel(encoder, decoder, dtn_blocks, args, distill_models)

    @staticmethod
    def build_distill_model(args, task, domain):
        """Build a new model instance."""
        assert isinstance(task, TranslationDtnTask)

        # make sure all arguments are present in older models
        base_dtn_architecture(args)
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        def load_distill_checkpoint(args, model, domain):

            filename = os.path.join(args.save_dir, 'finetune-checkpoints', \
                    'checkpoint_best_bleu_finetune_{}.pt'.format(domain))
            state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            model.upgrade_state_dict(state['model'])
            # load model parameters
            model.load_state_dict(state['model'], strict=True)
            print('| loaded kd checkpoint for {}'.format(domain))

        # build shared embeddings (if applicable)
        encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)
        # encoders/decoders for each language
        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        model = TransformerModel(encoder, decoder)
        load_distill_checkpoint(args, model, domain)

        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, domain=None):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            the decoder's output, typically of shape `(batch, tgt_len, vocab)`
        """
        encoder_out = self.encoder(src_tokens, src_lengths, domain=domain)
        encoder_out = self.dtn_blocks[domain](encoder_out)

        decoder_out = self.decoder(prev_output_tokens, encoder_out, domain=domain)
        return decoder_out

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        if self.args.restore_from_trans:
            super().load_state_dict(state_dict, False)
        else:
            super().load_state_dict(state_dict, True)

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

@register_model_architecture('transformer_dtn', 'transformer_dtn')
def base_dtn_architecture(args):
    base_architecture(args)

