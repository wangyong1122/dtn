# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn.functional as F

from fairseq.models.transformer import Linear

class DtnDiscr(torch.nn.Module):
    """
    """

    def __init__(self, args, domain_adv):
        super().__init__()

        self.embed_dim = args.encoder_embed_dim
        self.domain_adv = domain_adv
        self.label = dict()
        for i, domain in enumerate(args.domains):
            self.label[domain] = i
        self.fc1 = Linear(self.embed_dim, self.embed_dim, bias=False)
        self.fc2 = Linear(self.embed_dim, 1, bias=False)
        self.fc3 = Linear(self.embed_dim, len(args.domains), bias=False)

    def forward(self, x, padding_mask, domain):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        x = torch.transpose(x, 0, 1)

        weight = self.fc2(torch.tanh(self.fc1(x)))

        if padding_mask is None:
            padding_mask=torch.ones_like(x)[:,:,0:1]
        else:
            padding_mask = torch.unsqueeze(1.0 - padding_mask.float(), -1)

        weight = weight * padding_mask + (1 - padding_mask) * float('-1.0e10')
        weight = F.softmax(weight, 1)
        aggr = torch.sum(weight * x, 1)

        prob = F.softmax(self.fc3(torch.relu(aggr)), -1)
        log_prob = torch.log(prob+1.0e-8)

        selected_prob = log_prob[:, self.label[domain]]
        self.loss = -torch.mean(selected_prob)*torch.sum(padding_mask[:,:,0])

        if self.domain_adv:
            self.loss_entropy = torch.mean(torch.sum(log_prob*prob, -1))*torch.sum(padding_mask[:,:,0])

        return