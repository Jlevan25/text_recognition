import os

from PIL import Image


class MJSynth:
    def __init__(self, path, stage: str, transforms=None, num_data: int = None):
        self.paths, self.labels = [], []
        self.transforms = transforms
        self.stage = stage
        self._num_data = num_data
        self._read_data(path)

    def _read_data(self, path):
        path = os.path.join(path, r'mnt\ramdisk\max\90kDICT32px')
        with open(os.path.join(path, f'annotation_{self.stage}.txt')) as anns:
            for i, ann in enumerate(anns):
                file_path = ann.split(' ')[0][2:]
                if i == self._num_data:
                    break
                # if i % 100_000 == 0:
                #     print(len(self.labels))
                self.paths.append(os.path.join(path, file_path))
                self.labels.append(file_path.split('_')[1])

        # self.paths = self.paths * 10_000
        # self.labels = self.labels * 10_000

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        label = self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return self._num_data


if __name__ == '__main__':
    dataset = MJSynth(r'D:\datasets\mjsynth_small')
