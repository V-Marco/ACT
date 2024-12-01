from datetime import datetime
import os
import psutil
from multiprocessing import current_process

class Logger:

    def __init__(self, path: str = None):
        if path is None:
            self.path = None
        else:
            self.path = os.path.join(path, "log.txt")

    def log(self, msg: str):
        if self.path is None:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[INFO]: {msg}")
        else:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[INFO]: {msg}", file = open(self.path, "a"))

    def log_warining(self, msg: str):
        if self.path is None:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[WARNING]: {msg}")
        else:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[WARNING]: {msg}", file = open(self.path, "a"))

    def log_step(self, step: int):
        if self.path is None:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[STEP]: {step}")
        else:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[STEP]: {step}", file = open(self.path, "a"))

    def log_memory(self):
        memory = psutil.virtual_memory()
        if self.path is None:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[MEMORY]: available {round(memory.available * 1e-9, 2)}, used: {memory.percent}% of total.")
        else:
            print(f"({datetime.now()})-[PID: {current_process().pid}]–[MEMORY]: available {round(memory.available * 1e-9, 2)}, used: {memory.percent}% of total.", 
                  file = open(self.path, "a"))
            
class EmptyLogger(Logger):
    def log(self, msg: str): pass
    def log_warining(self, msg: str): pass
    def log_step(self, step: int): pass
    def log_memory(self): pass



