import signal
import sys

import time


def signal_handler(signal, frame):
    print(frame.f_locals)
    print(frame.f_globals)
    print(frame)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

while True:
    time.sleep(0.5)
