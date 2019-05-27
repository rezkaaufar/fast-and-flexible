from __future__ import print_function, division
import copy

import torch
import torchtext

from src.loss import NLLLoss
from src.metrics import WordAccuracy, SequenceAccuracy

class Evaluator(object):
    """ Class to evaluate models with given datasets.
    Args:
        loss (machine.loss, optional): loss for evaluator (default: machine.loss.NLLLoss)
        metrics (machine.metrics, optional): metrics for evaluator (default
            machine.metrics.WordAccuracy and SequenceAccuracy )
    """

    def __init__(self, loss=[NLLLoss()], metrics=[
                 WordAccuracy(), SequenceAccuracy()]):
        self.losses = loss
        self.metrics = metrics

    @staticmethod
    def update_batch_metrics(metrics, predicted, target_variable):
        """
        Update a list with metrics for current batch.
        Args:
            metrics (list): list with of machine.metric.Metric objects
            predicted (Tensor): prediction made by the model
            target_variable (Tensor): gold label
        Returns:
            metrics (list): list with updated metrics
        """

        outputs = predicted.argmax(dim=2)

        for metric in metrics:
            metric.eval_batch(outputs, target_variable)

        return metrics

    def compute_batch_loss(self, decoder_outputs, target_variable):
        """
        Compute the loss for the current batch.
        Args:
            decoder_outputs (torch.Tensor): decoder outputs of a batch
            decoder_hidden (torch.Tensor): (batch first) decoder hidden states for a batch
            target_variable (dict): map of keys to different targets
        Returns:
           losses (list): a list with machine.loss.Loss objects
        """

        losses = self.losses
        for loss in losses:
            loss.reset()

        losses = self.update_loss(losses, decoder_outputs, target_variable)

        return losses

    @staticmethod
    def update_loss(losses, decoder_outputs, target_variable):
        """
        Update a list with losses for current batch
        Args:
            losses (list): a list with machine.loss.Loss objects
            decoder_outputs (torch.Tensor): decoder outputs of a batch
            target_variable (dict): map of keys to different targets
        Returns:
           losses (list): a list with machine.loss.Loss objects
        """

        for loss in losses:
            loss.eval_batch(decoder_outputs, target_variable)

        return losses

    def evaluate(self, model, data_iterator, gen_position_ids):
        """ Evaluate a model on given dataset and return performance.
        Args:
            model (machine.models): model to evaluate
            data_iterator (torchtext.data.Iterator): data iterator to evaluate against
        Returns:
            loss (float): loss of the given model on the given dataset
            accuracy (float): accuracy of the given model on the given dataset
        """
        # If the model was in train mode before this method was called, we make sure it still is
        # after this method.

        # Since we are passing data_iterator
        # We evaluate on whole batches - so exhaust all batches first
        # and store the initial point
        # data_iterator_reset = False
        initial_iteration = data_iterator.iterations
        if initial_iteration > 1 and initial_iteration != len(data_iterator):
            raise Warning("Passed in data_iterator in middle of iterations")

        previous_train_mode = model.training
        model.eval()

        for loss in self.losses:
            loss.reset()
        losses = copy.deepcopy(self.losses)

        for metric in self.metrics:
            metric.reset()
        metrics = copy.deepcopy(self.metrics)

        # loop over batches
        with torch.no_grad():
            for batch in data_iterator:

                ins_variables = getattr(batch, 'ins')
                src_variables = getattr(batch, 'src')
                tgt_variables = getattr(batch, 'tgt')[0]
                ins_position_variables = gen_position_ids(ins_variables[1])
                position_variables = gen_position_ids(src_variables[1])

                predicted_softmax, attn = model(ins_variables[0], src_variables[0],
                                                ins_variables[1], ins_position_variables, position_variables)

                # Compute metric(s) over one batch
                metrics = self.update_batch_metrics(metrics, predicted_softmax, tgt_variables)

                # Compute loss(es) over one batch
                losses = self.update_loss(losses, predicted_softmax, tgt_variables)

        model.train(previous_train_mode)

        return losses, metrics