import os
import time
from multiprocessing import freeze_support
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

import sys
sys.path.append(r'C:\AdvCV\WasGAN')


def batch_collate(batch):
    inputs, targets, lengths = [], [], []

    max_length = 0
    images = []
    for img, label in batch:
        length = img.shape[-1]
        out_len = length // 4
        images.append(img)
        targets.append(label)
        lengths.append(out_len)

        if length > max_length:
            max_length = length

    for img in images:
        padded = torch.zeros((*img.shape[:2], max_length))
        padded[..., :img.shape[-1]] = img
        inputs.append(padded)
    return torch.stack(inputs), targets, lengths


def main():
    from datasets import MJSynth
    from executors import EpochManager
    from models import DeepTextSpotter
    from transforms import GaussianNoise, VerticalResize, RandomHorizontalResize, RandomHorizontalCrop, HorizontalLimit
    from configs import Config
    from utils import Timer

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_ROOT = r'D:\datasets\mjsynth'

    alphabet = ["A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f", "G", "g",
                "H", "h", "I", "i", "J", "j", "K", "k", "L", "l", "M", "m", "N", "n", "O", "o",
                "P", "p", "Q", "q", "R", "r", "S", "s", "T", "t", "U", "u", "V", "v", "W", "w",
                "X", "x", "Y", "y", "Z", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    char2idx = {char: i + 1 for i, char in enumerate(alphabet)}

    cfg = Config(ROOT_DIR=ROOT, DATASET_DIR=DATASET_ROOT,
                 dataset_name='MJSynth',
                 model_name='text_spotter', out_features=len(alphabet) + 1, device='cuda',
                 batch_size=4, lr=1e-4, momentum=0.9, weight_decay=0.00005,
                 debug=True, show_each=1000,
                 overfit=False, shuffle=False,
                 seed=10)

    keys = train_key, valid_key, test_key = 'train', 'val', 'test'

    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

    norm = transforms.Normalize(mean=0.4, std=0.225)
    eval_transforms = transforms.Compose([VerticalResize(height=32, save_ratio=True,
                                                         interpolation=Image.Resampling.BILINEAR),
                                          HorizontalLimit(min_=4, max_=500),
                                          transforms.ToTensor(),
                                          norm])

    image_transforms = {train_key: transforms.Compose([VerticalResize(height=32, save_ratio=True,
                                                                      interpolation=Image.Resampling.BILINEAR),
                                                       RandomHorizontalCrop(max_indent=1),
                                                       RandomHorizontalResize(min_scale=0.8, max_scale=1.2,
                                                                              interpolation=Image.Resampling.BILINEAR),
                                                       HorizontalLimit(min_=4, max_=500),
                                                       transforms.ToTensor(),
                                                       GaussianNoise(mean=0, std=.2),
                                                       norm]),

                        valid_key: eval_transforms,
                        test_key: eval_transforms}

    # dataset
    datasets_dict = {key: MJSynth(path=DATASET_ROOT, stage=key, num_data=125_000 if key == train_key else 12_500,
                                  transforms=image_transforms[key])
                     for key in keys}

    dataloaders_dict = {key: DataLoader(datasets_dict[key], batch_size=cfg.batch_size,
                                        shuffle=cfg.shuffle and key == train_key,
                                        num_workers=1, pin_memory=True,
                                        collate_fn=batch_collate)
                        for key in keys}

    model = DeepTextSpotter(1, cfg.out_features).to(cfg.device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=True)
    criterion = nn.CTCLoss()

    epoch_manager = EpochManager(dataloaders_dict=dataloaders_dict, model=model,
                                 optimizer=optimizer, criterion=criterion,
                                 alphabet=alphabet, cfg=cfg, char2idx=char2idx)

    epochs = 17
    save_each = 1

    final_train = True
    if final_train:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

    epoch_manager.load_model(
        r'C:\AdvCV\checkpoints\text_spotter\MJSynth\model_text_spotter_batch_size4_lr_1e-05_1669811033.8471413\15.pth')

    is_test = False
    if is_test:
        writer = SummaryWriter(log_dir=cfg.LOG_PATH)
        accs = epoch_manager.test(test_key)

        losses = np.cumsum(epoch_manager.losses[test_key]) / np.arange(1, 1 + len(epoch_manager.losses[test_key]))
        accs = np.cumsum(accs) / np.arange(1, 1 + len(accs))
        for i, (loss, acc) in enumerate(zip(losses, accs)):
            writer.add_scalar(f'{test_key}/Loss', loss, i)
            writer.add_scalar(f'{test_key}/Accuracy', acc, i)
    else:
        for epoch in range(16, epochs):
            with Timer(f'Train epoch #{epoch}'):
                epoch_manager.train(train_key, epoch)

            with Timer(f'Train epoch #{epoch}'):
                epoch_manager.validation(valid_key, epoch)
            epoch_manager.save_model(epoch)

    writer = SummaryWriter(log_dir=cfg.LOG_PATH)
    for stage, losses_list in epoch_manager.losses.items():
        losses = np.cumsum(losses_list) / np.arange(1, 1 + len(losses_list))
        for i, loss in enumerate(losses):
            writer.add_scalar(f'{stage}/Loss', loss, i)


if __name__ == '__main__':
    main()
