"""Metrics."""

from __future__ import print_function
from collections import Counter
import math
import numpy as np
import torch


class Metric(object):
    """ Base class for encapsulation of the metrics functions.
    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions
    Note:
        Do not use this class directly, use one of the sub classes.
    Attributes:
        name (str): name of the metric used by logging messages.
        target (str): dictionary key to fetch the target from dictionary that stores
                      different variables computed during the forward pass of the model
        metric_total (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the value of the metric of multiple batches.
            sub-classes.
    """

    def __init__(self, name, log_name, input_var):
        self.name = name
        self.log_name = log_name
        self.input = input_var

    def reset(self):
        """ Reset accumulated metric values"""
        raise NotImplementedError("Implement in subclass")

    def get_val(self):
        """ Get the value for the metric given the accumulated loss
        and the normalisation term
        Returns:
            loss (float): value of the metric.
        """
        raise NotImplementedError("Implement in subclass")

    def eval_batch(self, outputs, target):
        """ Compute the metric for the batch given results and target results.
        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError("Implement in subclass")


class WordAccuracy(Metric):
    """
    Batch average of word accuracy.
    Args:
        ignore_index (int, optional): index of masked token
    """

    _NAME = "Word Accuracy"
    _SHORTNAME = "acc"
    _INPUT = "sequence"

    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index
        self.word_match = 0
        self.word_total = 0

        super(WordAccuracy, self).__init__(
            self._NAME, self._SHORTNAME, self._INPUT)

    def get_val(self):
        if self.word_total != 0:
            return float(self.word_match) / self.word_total
        else:
            return 0

    def reset(self):
        self.word_match = 0
        self.word_total = 0

    def eval_batch(self, outputs, targets):
        # batch_size x seq_len
        non_padding = targets.ne(self.ignore_index)
        correct = outputs.eq(targets).long().sum().item()
        self.word_match += correct
        self.word_total += non_padding.long().sum().item()


class SequenceAccuracy(Metric):
    """
    Batch average of word accuracy.
    Args:
        ignore_index (int, optional): index of masked token
    """

    _NAME = "Sequence Accuracy"
    _SHORTNAME = "seq_acc"
    _INPUT = "seqlist"

    def __init__(self, ignore_index=None, device=None):
        self.ignore_index = ignore_index
        self.seq_match = 0
        self.seq_total = 0

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        super(SequenceAccuracy, self).__init__(
            self._NAME, self._SHORTNAME, self._INPUT)

    def get_val(self):
        if self.seq_total != 0:
            return float(self.seq_match) / self.seq_total
        else:
            return 0

    def reset(self):
        self.seq_match = 0
        self.seq_total = 0

    def eval_batch(self, outputs, targets):
        batch_size = outputs.size(0)
        truth = torch.ones(batch_size, dtype=torch.long, device=self.device) * 23
        correct = outputs.eq(targets).long().sum(dim=1)
        self.seq_match += correct.eq(truth).sum()
        self.seq_total += batch_size

        # batch_size = targets.size(1)
        #
        # # compute sequence accuracy over batch
        # match_per_seq = torch.zeros(
        #     batch_size, dtype=torch.float, device=self.device)
        # total_per_seq = torch.zeros(
        #     batch_size, dtype=torch.float, device=self.device)
        #
        # for step, step_output in enumerate(outputs):
        #     target = targets[step]
        #
        #     non_padding = target.ne(self.ignore_index)
        #
        #     correct_per_seq = (outputs[step].view(-1).eq(target) * non_padding)
        #     print(correct_per_seq)
        #     # correct_per_seq = (outputs[step].view(-1).eq(target)*non_padding).data
        #     match_per_seq += correct_per_seq.float()
        #     total_per_seq += non_padding.float()

        # self.seq_match += match_per_seq.eq(total_per_seq).long().sum()
        # self.seq_total += total_per_seq.shape[0]