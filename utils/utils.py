import time


class Timer:
    def __init__(self, timer_name=''):
        self.start_time = 0
        self.timer_name = timer_name

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, *args):
        print(f'{self.timer_name} time {time.perf_counter() - self.start_time:.4} sec')