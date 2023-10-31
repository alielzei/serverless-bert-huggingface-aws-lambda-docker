from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace, model_helper, utils, brew
from caffe2.python.rnn_cell import LSTM
from caffe2.proto import caffe2_pb2
from caffe2.python.optimizer import build_sgd
import argparse
import logging
import numpy as np
from datetime import datetime
'\nThis script takes a text file as input and uses a recurrent neural network\nto learn to predict next character in a sequence.\n'
logging.basicConfig()
log = logging.getLogger('char_rnn')
log.setLevel(logging.DEBUG)

def CreateNetOnce(net, created_names=set()):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.examples.char_rnn.CreateNetOnce', 'CreateNetOnce(net, created_names=set())', {'workspace': workspace, 'net': net, 'created_names': created_names}, 0)


class CharRNN(object):
    
    def __init__(self, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        self.iters_to_report = args.iters_to_report
        self.hidden_size = args.hidden_size
        with open(args.train_data) as f:
            self.text = f.read()
        self.vocab = list(set(self.text))
        self.char_to_idx = {ch: idx for (idx, ch) in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for (idx, ch) in enumerate(self.vocab)}
        self.D = len(self.char_to_idx)
        print('Input has {} characters. Total input size: {}'.format(len(self.vocab), len(self.text)))
    
    def CreateModel(self):
        log.debug('Start training')
        model = model_helper.ModelHelper(name='char_rnn')
        (input_blob, seq_lengths, hidden_init, cell_init, target) = model.net.AddExternalInputs('input_blob', 'seq_lengths', 'hidden_init', 'cell_init', 'target')
        (hidden_output_all, self.hidden_output, _, self.cell_state) = LSTM(model, input_blob, seq_lengths, (hidden_init, cell_init), self.D, self.hidden_size, scope='LSTM')
        output = brew.fc(model, hidden_output_all, None, dim_in=self.hidden_size, dim_out=self.D, axis=2)
        softmax = model.net.Softmax(output, 'softmax', axis=2)
        (softmax_reshaped, _) = model.net.Reshape(softmax, ['softmax_reshaped', '_'], shape=[-1, self.D])
        self.forward_net = core.Net(model.net.Proto())
        xent = model.net.LabelCrossEntropy([softmax_reshaped, target], 'xent')
        loss = model.net.AveragedLoss(xent, 'loss')
        model.AddGradientOperators([loss])
        build_sgd(model, base_learning_rate=0.1 * self.seq_length, policy='step', stepsize=1, gamma=0.9999)
        self.model = model
        self.predictions = softmax
        self.loss = loss
        self.prepare_state = core.Net('prepare_state')
        self.prepare_state.Copy(self.hidden_output, hidden_init)
        self.prepare_state.Copy(self.cell_state, cell_init)
    
    def _idx_at_pos(self, pos):
        return self.char_to_idx[self.text[pos]]
    
    def TrainModel(self):
        log.debug('Training model')
        workspace.RunNetOnce(self.model.param_init_net)
        smooth_loss = -np.log(1.0 / self.D) * self.seq_length
        last_n_iter = 0
        last_n_loss = 0.0
        num_iter = 0
        N = len(self.text)
        text_block_positions = np.zeros(self.batch_size, dtype=np.int32)
        text_block_size = N // self.batch_size
        text_block_starts = list(range(0, N, text_block_size))
        text_block_sizes = [text_block_size] * self.batch_size
        text_block_sizes[self.batch_size - 1] += N % self.batch_size
        assert sum(text_block_sizes) == N
        workspace.FeedBlob(self.hidden_output, np.zeros([1, self.batch_size, self.hidden_size], dtype=np.float32))
        workspace.FeedBlob(self.cell_state, np.zeros([1, self.batch_size, self.hidden_size], dtype=np.float32))
        workspace.CreateNet(self.prepare_state)
        last_time = datetime.now()
        progress = 0
        while True:
            workspace.FeedBlob('seq_lengths', np.array([self.seq_length] * self.batch_size, dtype=np.int32))
            workspace.RunNet(self.prepare_state.Name())
            input = np.zeros([self.seq_length, self.batch_size, self.D]).astype(np.float32)
            target = np.zeros([self.seq_length * self.batch_size]).astype(np.int32)
            for e in range(self.batch_size):
                for i in range(self.seq_length):
                    pos = text_block_starts[e] + text_block_positions[e]
                    input[i][e][self._idx_at_pos(pos)] = 1
                    target[i * self.batch_size + e] = self._idx_at_pos((pos + 1) % N)
                    text_block_positions[e] = (text_block_positions[e] + 1) % text_block_sizes[e]
                    progress += 1
            workspace.FeedBlob('input_blob', input)
            workspace.FeedBlob('target', target)
            CreateNetOnce(self.model.net)
            workspace.RunNet(self.model.net.Name())
            num_iter += 1
            last_n_iter += 1
            if num_iter % self.iters_to_report == 0:
                new_time = datetime.now()
                print('Characters Per Second: {}'.format(int(progress / (new_time - last_time).total_seconds())))
                print('Iterations Per Second: {}'.format(int(self.iters_to_report / (new_time - last_time).total_seconds())))
                last_time = new_time
                progress = 0
                print('{} Iteration {} {}'.format('-' * 10, num_iter, '-' * 10))
            loss = workspace.FetchBlob(self.loss) * self.seq_length
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            last_n_loss += loss
            if num_iter % self.iters_to_report == 0:
                self.GenerateText(500, np.random.choice(self.vocab))
                log.debug('Loss since last report: {}'.format(last_n_loss / last_n_iter))
                log.debug('Smooth loss: {}'.format(smooth_loss))
                last_n_loss = 0.0
                last_n_iter = 0
    
    def GenerateText(self, num_characters, ch):
        CreateNetOnce(self.forward_net)
        text = '' + ch
        for _i in range(num_characters):
            workspace.FeedBlob('seq_lengths', np.array([1] * self.batch_size, dtype=np.int32))
            workspace.RunNet(self.prepare_state.Name())
            input = np.zeros([1, self.batch_size, self.D]).astype(np.float32)
            input[0][0][self.char_to_idx[ch]] = 1
            workspace.FeedBlob('input_blob', input)
            workspace.RunNet(self.forward_net.Name())
            p = workspace.FetchBlob(self.predictions)
            next = np.random.choice(self.D, p=p[0][0])
            ch = self.idx_to_char[next]
            text += ch
        print(text)


@utils.debug
def main():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.examples.char_rnn.main', 'main()', {'argparse': argparse, 'core': core, 'workspace': workspace, 'caffe2_pb2': caffe2_pb2, 'CharRNN': CharRNN, 'utils': utils}, 0)
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()

