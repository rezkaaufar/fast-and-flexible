import os
import argparse
import logging

import torch
import torchtext
import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch import optim

from src.data.get_standard_iter import get_standard_iter
from src.data.fields import MechField
from src.models.EncoderRNN import EncoderRNN
from src.models.DecoderRNN import DecoderRNN
from src.models.DecoderConv import DecoderConv
from src.models.EncoderConv import EncoderConv
from src.models.enc2dec import Enc2dec
from src.loss.loss import NLLLoss
from src.metrics.metrics import WordAccuracy, SequenceAccuracy
from src.evaluator.evaluator import Evaluator
from src.trainer import SupervisedTrainer
from src.utils import Checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# CONSTANTS
IGNORE_INDEX = -1
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

def init_argparser():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('--train', help='Training data')
    parser.add_argument('--dev', help='Development data')
    parser.add_argument('--monitor', nargs='+', default=[],
                        help='Data to monitor during training')
    parser.add_argument('--output_dir', default='models/',
                        help='Path to model directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs', default=6)
    parser.add_argument('--optim', type=str, help='Choose optimizer',
                        choices=['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'sgd'])
    parser.add_argument('--max_len', type=int,
                        help='Maximum sequence length', default=50)
    parser.add_argument('--encoder', type=str,
                        help='Which encoder to use', choices=['rnn','conv'])
    parser.add_argument('--decoder', type=str,
                        help='Which decoder to use', choices=['rnn','conv'])
    parser.add_argument(
        '--rnn_cell', help="Chose type of rnn cell", default='lstm')
    parser.add_argument('--bidirectional', action='store_true',
                        help="Flag for bidirectional encoder")
    parser.add_argument('--embedding_size', type=int,
                        help='Embedding size', default=128)
    parser.add_argument('--hidden_size', type=int,
                        help='Hidden layer size', default=128)
    parser.add_argument('--n_layers', type=int,
                        help='Number of RNN layers in both encoder and decoder', default=2)
    parser.add_argument('--n_conv_layers', type=int,
                        help='Number of Convolutional layers in both encoder and decoder', default=3)
    parser.add_argument('--blck_vocab', type=int,
                        help='blocks vocabulary size', default=50000)
    parser.add_argument('--ins_vocab', type=int,
                        help='instructions vocabulary size', default=50000)
    parser.add_argument('--dropout_p_encoder', type=float,
                        help='Dropout probability for the encoder', default=0.2)
    parser.add_argument('--dropout_p_decoder', type=float,
                        help='Dropout probability for the decoder', default=0.2)
    parser.add_argument(
        '--attention', choices=['pre-rnn', 'post-rnn'], default=False)
    parser.add_argument('--attention_method',
                        choices=['dot', 'mlp', 'concat', 'general'], default=None)
    parser.add_argument('--metrics', nargs='+', default=['word_acc','seq_acc'], choices=[
                        'word_acc', 'seq_acc'], help='Metrics to use')
    parser.add_argument('--full_focus', action='store_true')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=32)
    parser.add_argument('--eval_batch_size', type=int,
                        help='Batch size', default=128)
    parser.add_argument(
        '--lr', type=float, help='Learning rate, recommended settings.'
                                 '\nrecommended settings: adam=0.001 '
                                 'adadelta=1.0 adamax=0.002 rmsprop=0.01 sgd=0.1', default=0.001)
    # Data management
    parser.add_argument('--load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')

    parser.add_argument('--save_every', type=int,
                        help='Every how many batches the model should be saved', default=100)
    parser.add_argument('--print_every', type=int,
                        help='Every how many batches to print results', default=100)

    parser.add_argument('--resume-training', action='store_true',
                        help='Indicates if training has to be resumed from the latest checkpoint')

    parser.add_argument('--log-level', default='info', help='Logging level.')
    parser.add_argument(
        '--write-logs', help='Specify file to write logs to after training')
    parser.add_argument('--cuda_device', default=0,
                        type=int, help='set cuda device to use')

    return parser

def init_logging(opt):
    logging.basicConfig(format=LOG_FORMAT, level=getattr(
        logging, opt.log_level.upper()))
    logging.info(opt)

def prepare_iters(opt):
    # use_output_eos = not opt.ignore_output_eos
    src = MechField(batch_first=True)
    ins = MechField(batch_first=True)
    tgt = MechField(batch_first=True)
    tabular_data_fields = [('src', src), ('ins', ins), ('tgt', tgt)]

    max_len = opt.max_len

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    # generate training and testing data
    train = get_standard_iter(torchtext.data.TabularDataset(
        path=opt.train, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    ), batch_size=opt.batch_size)

    if opt.dev:
        dev = get_standard_iter(torchtext.data.TabularDataset(
            path=opt.dev, format='tsv',
            fields=tabular_data_fields,
            filter_pred=len_filter), batch_size=opt.eval_batch_size)
    else:
        dev = None

    monitor_data = OrderedDict()
    for dataset in opt.monitor:
        m = get_standard_iter(torchtext.data.TabularDataset(
            path=dataset, format='tsv',
            fields=tabular_data_fields,
            filter_pred=len_filter), batch_size=opt.eval_batch_size)
        monitor_data[dataset] = m

    return src, ins, tgt, train, dev, monitor_data

def prepare_losses_and_metrics(opt, pad):
    # Prepare loss and metrics
    losses = [NLLLoss(ignore_index=pad)]
    loss_weights = [1.]

    for loss in losses:
        loss.to(device)

    metrics = []

    if 'word_acc' in opt.metrics:
        metrics.append(WordAccuracy(ignore_index=pad))
    if 'seq_acc' in opt.metrics:
        metrics.append(SequenceAccuracy(ignore_index=pad))

    return losses, loss_weights, metrics

def load_model_from_checkpoint(opt, src, tgt):
    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.output_dir, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.output_dir, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model

    input_vocab = checkpoint.input_vocab
    src.vocab = input_vocab

    output_vocab = checkpoint.output_vocab
    tgt.vocab = output_vocab

    ins_vocab = checkpoint.ins_vocab
    ins.vocab = ins_vocab

    return seq2seq, input_vocab, ins_vocab, output_vocab


def initialize_model(opt, src, ins, tgt, train):
    # build vocabulary
    src.build_vocab(train.dataset, max_size=opt.blck_vocab)
    ins.build_vocab(train.dataset, max_size=opt.ins_vocab)
    tgt.build_vocab(train.dataset, max_size=opt.blck_vocab)
    input_vocab = src.vocab
    ins_vocab = ins.vocab
    output_vocab = tgt.vocab

    # Initialize model
    hidden_size = opt.hidden_size
    # decoder_hidden_size = hidden_size * 2 if opt.bidirectional else hidden_size

    if opt.encoder == "rnn":
        encoder = EncoderRNN(len(ins.vocab), opt.max_len, opt.embedding_size, opt.hidden_size,
                             dropout_p=opt.dropout_p_encoder,
                             n_layers=opt.n_layers,
                             bidirectional=opt.bidirectional,
                             rnn_cell=opt.rnn_cell,
                             variable_lengths=True)
        encoder.to(device)
    else:
        encoder = EncoderConv(len(ins.vocab), 3, opt.hidden_size, opt.hidden_size,
                              n_layers=opt.n_conv_layers,
                              dropout_p=opt.dropout_p_decoder,
                              max_len_blocks=5)
        encoder.to(device)

    if opt.decoder == "rnn":
        decoder = DecoderRNN(len(src.vocab), opt.max_len, opt.hidden_size,
                             dropout_p=opt.dropout_p_decoder,
                             n_layers=opt.n_layers,
                             use_attention=opt.attention,
                             attention_method=opt.attention_method,
                             full_focus=opt.full_focus,
                             bidirectional=opt.bidirectional,
                             rnn_cell=opt.rnn_cell)
        decoder.to(device)
    else:
        decoder = DecoderConv(len(src.vocab), 3, opt.hidden_size, opt.hidden_size,
                              n_layers=opt.n_conv_layers,
                              dropout_p=opt.dropout_p_decoder,
                              max_len_blocks=23,
                              use_attention=opt.attention,
                              attention_method=opt.attention_method,
                              full_focus=opt.full_focus)
        decoder.to(device)

    enc2dec = Enc2dec(encoder, decoder, which_enc=opt.encoder, which_dec=opt.decoder)

    # This enables using all GPUs available
    if torch.cuda.device_count() > 1:
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        enc2dec = torch.nn.DataParallel(enc2dec)

    enc2dec.to(device)

    return enc2dec, input_vocab, ins_vocab, output_vocab

# def generate_position_ids(length_list):
#     batch_size = len(length_list)
#     length = max(length_list).item()
#     pos_tensor = torch.zeros(batch_size, length).long()
#     for i in range(batch_size):
#         pos_tensor[i] = torch.LongTensor(range(0, length))
#     return Variable(pos_tensor)
#
# def get_optim(optim_name):
#     optims = {'adam': optim.Adam, 'adagrad': optim.Adagrad,
#               'adadelta': optim.Adadelta, 'adamax': optim.Adamax,
#               'rmsprop': optim.RMSprop, 'sgd': optim.SGD,
#               None: optim.Adam}
#     return optims[optim_name]

parser = init_argparser()
opt = parser.parse_args()
init_logging(opt)
src, ins, tgt, train, dev, monitor_data = prepare_iters(opt)
src.build_vocab(train.dataset, max_size=opt.blck_vocab)
ins.build_vocab(train.dataset, max_size=opt.ins_vocab)
tgt.build_vocab(train.dataset, max_size=opt.blck_vocab)

# print(len(src.vocab), len(ins.vocab), len(tgt.vocab))
#
# for elem in range(len(ins.vocab)):
#     print(ins.vocab.itos[elem])

encoder = None
decoder = None
if opt.encoder == "rnn":
    encoder = EncoderRNN(len(ins.vocab), opt.max_len, opt.embedding_size, opt.hidden_size,
                         dropout_p=opt.dropout_p_encoder,
                         n_layers=opt.n_layers,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         variable_lengths=True)
    encoder.to(device)
else:
    encoder = EncoderConv(len(ins.vocab), 3, opt.hidden_size, opt.hidden_size,
                         n_layers=opt.n_conv_layers,
                         dropout_p=opt.dropout_p_decoder,
                         max_len_blocks=5)
    encoder.to(device)

if opt.decoder == "rnn":
    decoder = DecoderRNN(len(src.vocab), opt.max_len, opt.hidden_size,
                         dropout_p=opt.dropout_p_decoder,
                         n_layers=opt.n_layers,
                         use_attention=opt.attention,
                         attention_method=opt.attention_method,
                         full_focus=opt.full_focus,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell)
    decoder.to(device)
else:
    decoder = DecoderConv(len(src.vocab), 3, opt.hidden_size, opt.hidden_size,
                         n_layers=opt.n_conv_layers,
                         dropout_p=opt.dropout_p_decoder,
                         max_len_blocks=23,
                         use_attention=opt.attention,
                         attention_method=opt.attention_method,
                         full_focus=opt.full_focus)
    decoder.to(device)

enc2dec = Enc2dec(encoder, decoder, which_enc=opt.encoder, which_dec=opt.decoder)

##########################

### prepare training ###
pad = tgt.vocab.stoi[tgt.pad_token]
losses, loss_weights, metrics = prepare_losses_and_metrics(opt, pad)
checkpoint_path = os.path.join(
        opt.output_dir, opt.load_checkpoint) if opt.resume_training else None
trainer = SupervisedTrainer(expt_dir=opt.output_dir)
### Train ###
enc2dec, logs = trainer.train(enc2dec, train,
                              num_epochs=opt.epochs, dev_data=dev, monitor_data=monitor_data, optimizer=opt.optim,
                              learning_rate=opt.lr, resume_training=opt.resume_training, checkpoint_path=checkpoint_path,
                              losses=losses, metrics=metrics, loss_weights=loss_weights,
                              checkpoint_every=opt.save_every, print_every=opt.print_every)

if opt.write_logs:
    output_path = os.path.join(opt.output_dir, opt.write_logs)
    logs.write_to_file(output_path)

##########################

# pad = tgt.vocab.stoi[tgt.pad_token]
# evaluator = Evaluator(loss=losses, metrics=metrics)
# optim_name = opt.optim
# optimizer = Optimizer(get_optim(optim_name)(enc2dec.parameters(),lr=opt.lr),max_grad_norm=5)
# enc2dec.train()

# for batch in tqdm.tqdm(train):
#     ins_variables = getattr(batch, 'ins')
#     src_variables = getattr(batch, 'src')
#     print(batch)
#     # tmp_tgt = getattr(batch, 'tgt')[0]
#     # for i in tmp_tgt[0]:
#     #     print(tgt.vocab.itos[int(i)])
#     tgt_variables = getattr(batch, 'tgt')[0]
#     output = None
#     hidden = None
#     predicted_softmax = None
#     attn = None
#     ins_position_variables = generate_position_ids(ins_variables[1])
#     position_variables = generate_position_ids(src_variables[1])
#
#     predicted_softmax, attn = enc2dec(ins_variables[0], src_variables[0],
#                                       ins_variables[1], ins_position_variables, position_variables)
#
#     # for step, _ in enumerate(predicted_softmax):
#     #     print(step)
#
#     losses = evaluator.compute_batch_loss(predicted_softmax, tgt_variables)
#
#     for i, loss in enumerate(losses, 0):
#         loss.scale_loss(loss_weights[i])
#         loss.backward(retain_graph=True)
#         # print(loss.acc_loss)
#
#     optimizer.step()
#     enc2dec.zero_grad()
#
#     # if opt.encoder == "conv":
#     #     ins_position_variables = generate_position_ids(ins_variables[1])
#     #     output = encoder(ins_variables[0], ins_position_variables)
#     # else:
#     #     output, hidden = encoder(ins_variables[0], input_lengths=ins_variables[1])
#     # if opt.decoder == "conv":
#     #     position_variables = generate_position_ids(src_variables[1])
#     #     predicted_softmax, attn = decoder(src_variables[0], position_variables, output, F.log_softmax)
#     # else:
#     #     if opt.encoder == "conv":
#     #         hidden = None
#     #     print(hidden)
#     #     predicted_softmax, hidden, attn = decoder(src_variables[0], hidden, output, F.log_softmax)
#     # output, _ = encoder(ins_variables[0], input_lengths=ins_variables[1])
#
#     # output, hidden = encoder(ins_variables[0], input_lengths=ins_variables[1])
#
#     # print(predicted_softmax)
#     break
#
# enc2dec.eval()
# losses, metrics = evaluator.evaluate(enc2dec, dev, generate_position_ids)
# for metric in metrics:
#     print(metric.get_val())
# for loss in losses:
#     print(loss.get_loss())