import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import torch
import pandas as pd
from io import StringIO
import json
import multiprocessing
import queue
import os
import time
import signal
import sys

from legacy import utils

from simulation_configs import pospischilsPYr_passive

import warnings

warnings.filterwarnings("ignore")

config = pospischilsPYr_passive

output_file = "output/arima_stats.json"

parent_pid = os.getpid()


def arima_coefs(traces, num_processes=64):
    class ARIMAProcess(multiprocessing.Process):
        def __init__(self, queue=None, out_queue=None, cancel_event=None):
            super(ARIMAProcess, self).__init__()
            self.queue = queue
            self.out_queue = out_queue
            self.cancel_event = cancel_event

        def run(self):
            while not self.queue.empty() and not self.cancel_event.is_set():
                trace_dict = self.queue.get()
                print(
                    f"processing next item... approximate # of traces remaining: {self.queue.qsize()}"
                )
                self.process(trace_dict)
            if self.cancel_event.is_set():
                print("process cancelled")

        def process(self, trace_dict):
            cell_id = trace_dict["cell_id"]
            trace = trace_dict["trace"]
            model = ARIMA(endog=trace, order=(10, 0, 10)).fit()
            stats_df = pd.read_csv(
                StringIO(model.summary().tables[1].as_csv()),
                index_col=0,
                skiprows=1,
                names=["coef", "std_err", "z", "significance", "0.025", "0.975"],
            )
            stats_df.loc[stats_df["significance"].astype(float) > 0.05, "coef"] = 0
            coefs = stats_df.coef.tolist()
            self.out_queue.put({"cell_id": cell_id, "coefs": coefs})
            print(f"processed cell: {cell_id} | {coefs}")

    MAX_QUEUE_SIZE = 32767

    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue(MAX_QUEUE_SIZE)
    total_traces = len(traces)

    if total_traces > MAX_QUEUE_SIZE:
        print("WARNING: total traces exceeds the max queue size. TRACES WILL BE LOST.")

    for i, trace in enumerate(traces):
        input_queue.put({"cell_id": i, "trace": trace.cpu().detach().numpy()})

    processes = []
    cancel_event = multiprocessing.Event()
    print(f"spawning {num_processes} ARIMA process(es)")
    for _ in range(num_processes):
        process = ARIMAProcess(input_queue, output_queue, cancel_event)
        processes.append(process)

    def write_output():
        print(f"writing results to {output_file}")
        coefs_list = []
        coefs_dict = {}
        if not output_queue.empty():
            values_captured = 0
            while not output_queue.empty() or values_captured < total_traces:
                try:
                    output = output_queue.get(True, 60)
                except:
                    print("no values on the queue for 60 seconds, breaking")
                    break
                coefs_dict[output["cell_id"]] = output["coefs"]
                values_captured = values_captured + 1

            for i in range(len(traces)):
                if coefs_dict.get(i):
                    coefs_list.append(coefs_dict[i])
                else:
                    coefs_list.append([0 for i in range(20)])

            output_dict = {}
            output_dict["arima_coefs"] = coefs_list

            with open(output_file, "w") as fp:
                json.dump(output_dict, fp, indent=4)

            print(
                f"done writing output. [{values_captured}/{total_traces}] runs successfully captured."
            )
        else:
            print("output queue empty, writing nothing")

    def signal_handler(signal, frame):
        print("Inturrupt caught, exiting. Closing processes.")
        if os.getpid() == parent_pid:
            if not cancel_event.set():  # only want to write once
                write_output()  # want to write the output no matter how far we get. Sometimes it hangs on several.
            cancel_event.set()
            # for process in processes:
            # process.terminate()
            # process.join(0)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("starting processes")
    for process in processes:
        process.start()

    print("waiting for processes to complete")
    for process in processes:
        # process.terminate()
        process.join()

    if not cancel_event.set():
        write_output()


if __name__ == "__main__":
    traces, params, amps = utils.load_parametric_traces(config)
    # n_spikes, spike_min, spike_max = utils.spike_stats(traces)
    arima_coefs(traces)
    print("done")
