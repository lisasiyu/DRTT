#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

'''
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")
'''

Batch = namedtuple("Batch", "ids src_tokens src_lengths")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, arg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)
   
    constraints_tensor = None

    tokens, lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=arg.max_tokens,
        max_sentences=arg.batch_size,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )


class Generator():
    def __init__(self, data_path, checkpoint_path="checkpoint_best.pt"):
        self.parser = options.get_generation_parser(interactive=True)
        self.parser.set_defaults(path=checkpoint_path,remove_bpe="none", dataset_impl="lazy", num_wokers=5)
        self.args = options.parse_args_and_arch(self.parser,input_args=[data_path])

        utils.import_user_module(self.args)

        if self.args.buffer_size < 1:
            self.args.buffer_size = 1
        if self.args.max_tokens is None and self.args.batch_size is None:
            self.args.batch_size = 1

        assert (
            not self.args.sampling or self.args.nbest == self.args.beam
        ), "--sampling requires --nbest to be equal to --beam"
        assert (
            not self.args.batch_size
            or self.args.batch_size <= self.args.buffer_size
        ), "--batch-size cannot be larger than --buffer-size"



        self.use_cuda = torch.cuda.is_available() and not self.args.cpu

        # Setup task, e.g., translation
        self.task = tasks.setup_task(self.args)

        # Load ensemble
        overrides = ast.literal_eval(self.args.model_overrides)
        self.models, self._model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(self.args.path),
        arg_overrides=overrides,
        task=self.task,
        )   

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary
        #self.src_dict = src_dict
        #self.tgt_dict = tgt_dict

        # Optimize ensemble for generation
        for model in self.models:
            if model is None:
                continue
            if self.args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        # Initialize generator
        self.generator = self.task.build_generator(self.models,self.args)

        # Handle tokenization and BPE
        self.tokenizer = self.task.build_tokenizer(self.args.tokenizer)
        self.bpe = self.task.build_bpe(self.args.bpe)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )



    def generate(self, string):

        def encode_fn(x):
            if self.tokenizer is not None:
                x = self.tokenizer.encode(x)
            if self.bpe is not None:
                x = self.bpe.encode(x)
            return x

        def decode_fn(x):
            if self.bpe is not None:
                x = self.bpe.decode(x)
            if self.tokenizer is not None:
                x = self.tokenizer.decode(x)
            return x
        start_id = 0
        inputs = [string]
        results = []
        for batch in make_batches(inputs, self.args, self.task, self.max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            translations = self.task.inference_step(
                self.generator, self.models, sample
            )
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append(
                    (
                        start_id + id,
                        src_tokens_i,
                        hypos,
                    )
                )

        # sort output to match input order
        for id_, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            src_str = ''
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.post_process)

            # Process top predictions
            #self.args.nbest = 5
            out = []
            for hypo in hypos[: min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=None,
                )
                detok_hypo_str = decode_fn(hypo_str)
                #score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                out.append(hypo_str)
                # detokenized hypothesis
                #print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str))
        start_id += len(inputs)
        return out


if __name__ == "__main__":
    gen = Generator("/apdcephfs/share_47076/lisalai/code/fairseq-robust-nmt/data-bin/tlm_ch-en",
                    "/apdcephfs/share_47076/lisalai/code/fairseq-robust-nmt/checkpoints/tlm_ch-en/checkpoint_best.pt")
    gen.generate('欧盟 执委会 主席 普罗迪 认为 , 周六 将 是 柏@@ 林@@ 围@@ 墙 倒塌 而 使 苏联 集团 >成为 过@@ 眼@@ 云@@ 烟 十五 年 来 , 欧洲 大@@ 一@@ 统 的 " 惊人 " 高潮 。 [SEP] european commission chairman prodi hailed saturday as an " ast@@ on@@ ishing " cli@@ max to the process of re - uniting europe 15 years after the berlin wall came crashing down , [MASK] with it the soviet bloc .')
