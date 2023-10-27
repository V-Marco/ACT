from act import utils
from simulation_configs import selected_config
import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from io import StringIO
import numpy as np
from statsmodels.graphics.tsaplots import plot_predict 
import time

warnings.filterwarnings("ignore")

traces = None
params = None

def plot_arima(cell_id, pq=10):
    trace = traces[cell_id]
    param = params[cell_id]
    start_time = time.time()
    model = ARIMA(endog=trace, order=(pq, 0, pq)).fit()
    print("--- took %s seconds to generate fit ---" % (time.time() - start_time))
    stats_df = pd.read_csv(
        StringIO(model.summary().tables[1].as_csv()),
        index_col=0,
        skiprows=1,
        names=["coef", "std_err", "z", "significance", "0.025", "0.975"],
    )
    print(f"Cell id: {cell_id}")
    print(f"Params: {params[cell_id]}")
    print(f"{stats_df}")
    _, ax = plt.subplots(1, 1, figsize=(10, 10))

    times = np.arange(len(trace))
    ax.plot(times, trace, label="Original Trace")
    #plot_predict(model, 0, 2000, dynamic=True, ax=ax, plot_insample=False)
    pred = model.get_prediction(start=0, dynamic=False)
    ax.plot(times, pred.predicted_mean, label='ARIMA Fit', alpha=.7)
    ax.set_title(f"ARIMA model fit for cell {cell_id}")
    ax.set_xlabel("Timestamp (ms)")
    ax.set_ylabel("V (mV)")
    ax.legend()
    ax.grid()

    plt.show()


if __name__ == "__main__":
    traces, params, amps = utils.load_parametric_traces(selected_config)

    traces = traces.cpu().detach().tolist()
    params = params.cpu().detach().tolist()
    cell_id = 40000
    plot_arima(cell_id)

    print("pausing for PDB, run `plot_arima(100)` where 100 is the desired cell id, to plot an ARIMA fit for a trace")
    import pdb;pdb.set_trace()
