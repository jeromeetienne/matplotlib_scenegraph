import time


class FpsMonitor:
    """
    A simple FPS monitor to track and print frames per second.
    """

    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self.last_time = None

    def reset(self):
        self.fps = 0.0
        self.last_time = None

    def print_fps(self):
        self.update(time.time())
        print(f"Frame per second: {self.fps:.2f}")

    def update(self, current_time):
        if self.last_time is not None:
            delta = current_time - self.last_time
            if delta > 0:
                current_fps = 1.0 / delta
                self.fps = (self.smoothing * self.fps) + ((1 - self.smoothing) * current_fps)
        self.last_time = current_time

    def get_fps(self):
        return self.fps


if __name__ == "__main__":
    fps_monitor = FpsMonitor()
    for _ in range(10):
        time.sleep(0.1)  # Simulate work
        fps_monitor.print_fps()
