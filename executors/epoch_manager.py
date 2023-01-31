import os.path
from typing import Iterator, Union, List

import Levenshtein as Levenshtein
import torch
from torch import tensor
from torch.utils.data import DataLoader


class EpochManager:
    def __init__(self,
                 model, optimizer, criterion, cfg,
                 alphabet,
                 char2idx,
                 writer=None,
                 scheduler=None,
                 class_names=None,
                 dataloaders_dict=None,
                 metrics: dict = None,
                 device=None):
        self.cfg = cfg
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = criterion
        self.metrics = metrics
        self.device = self.cfg.device if device is None else device
        self.writer = writer
        self.alphabet = alphabet
        self.dataloaders = dataloaders_dict if dataloaders_dict is not None else dict()
        self.losses = dict()

        self._global_step = dict()
        self.char2idx = char2idx

    def train(self, stage_key, i_epoch):
        self.model.train()
        for batch_info in self._epoch_generator(stage=stage_key, epoch=i_epoch):
            for param in self.model.parameters():
                param.grad = None
            batch_info['loss'].backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    @torch.no_grad()
    def validation(self, stage_key, i_epoch):
        self.model.eval()
        for batch_info in self._epoch_generator(stage=stage_key, epoch=i_epoch):
            ...

    @torch.no_grad()
    def test(self, stage_key):
        self.model.eval()
        accuracies = []
        for batch_info in self._epoch_generator(stage=stage_key):
            targets = batch_info['targets']
            prediction_words = [self._seq2word(seq) for seq in batch_info['predictions'].argmax(1)]
            accuracy = 0
            for p_word, t_word in zip(prediction_words, targets):
                accuracy += Levenshtein.distance(p_word, t_word) / max(len(p_word), len(t_word))

            accuracies.append(1 - accuracy / len(targets))
        return accuracies

    def save_model(self, epoch, path=None):
        path = self.cfg.SAVE_PATH if path is None else path

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, f'{epoch}.pth')

        checkpoint = dict(epoch=self._global_step,
                          model=self.model.state_dict(),
                          optimizer=self.optimizer.state_dict(),
                          losses=self.losses,
                          metrics=self.metrics)

        torch.save(checkpoint, path)
        print('model saved, epoch:', epoch)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        self._global_step = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.losses = checkpoint['losses']
        self.metrics = checkpoint['metrics']
        print('model loaded')

    def _epoch_generator(self, stage, epoch=None) -> Iterator[tensor]:

        if stage not in self._global_step:
            self._get_global_step(stage)

        print('\n_______', stage, f'epoch{epoch}' if epoch is not None else '',
              'len:', len(self.dataloaders[stage]), '_______')

        for i, (inputs_, targets, lengths) in enumerate(self.dataloaders[stage]):

            self._global_step[stage] += 1
            t_indexes = [[self.char2idx[char] for char in label] for label in targets]
            predictions, logits = self.model(inputs_.to(self.device))
            predictions = predictions.detach().cpu()
            loss = self.criterion(predictions, t_indexes, lengths)
            self.losses[stage].append(loss.cpu().detach().item())

            if self.cfg.debug and i % self.cfg.show_each == 0:
                print('\n___', f'Iteration {i + 1}', '___')
                print(f'Loss: {loss.item()}')

            yield dict(loss=loss, predictions=predictions, logits=logits, targets=targets)

    def _get_global_step(self, data_type):
        self._global_step[data_type] = -1
        self.losses[data_type] = []

    def _seq2word(self, seq):
        word = ''
        prev_idx = 0
        for idx in seq:
            if idx != prev_idx and idx > 0:
                word += self.alphabet[idx - 1]
            prev_idx = idx
        return word
