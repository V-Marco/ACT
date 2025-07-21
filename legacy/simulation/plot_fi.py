import argparse
import os

from simulation_configs import selected_config
import sys
sys.path.append("../")
from act import analysis, utils
import meta_sweep


def main(config, extra_trace, extra_trace_label, title=None):

    inj_dur = config["simulation_parameters"]["h_i_dur"]
    random_seed = f"{config['optimization_parameters']['random_seed']}"
    output_folder = utils.get_output_folder_name(config)
    target_params = config["optimization_parameters"].get("target_params")
    model_data_dir = utils.get_last_model_data_folder_name(config)
    fi_file = model_data_dir + f"FI.png"

    traces_file = model_data_dir + "traces.h5"
    simulated_traces, target_traces, amps = utils.load_final_traces(traces_file)

    simulated_curve = utils.get_fi_curve(simulated_traces, amps, inj_dur=inj_dur)
    target_curve = utils.get_fi_curve(target_traces, amps, inj_dur=inj_dur)

    simulated_label = config["output"]["simulated_label"]
    target_label = config["output"]["target_label"]

    err1 = utils.get_fi_curve_error(simulated_traces, target_traces, amps, inj_dur=inj_dur)
    simulated_label = simulated_label + f" (err: {err1})"

    curves_list = [
        simulated_curve.cpu().detach().numpy(),
        target_curve.cpu().detach().numpy(),
    ]
    labels = [simulated_label, target_label]

    if extra_trace is not None:
        extra_trace_fi = utils.get_fi_curve(extra_trace, amps, inj_dur=inj_dur)
        extra_trace_fi = extra_trace_fi.cpu().detach().numpy()
        curves_list.append(extra_trace_fi)
        err2 = utils.get_fi_curve_error(extra_trace, target_traces, amps, inj_dur=inj_dur)
        extra_trace_label = extra_trace_label + f" (err: {err2})"
        labels.append(extra_trace_label)

    print(fi_file)
    analysis.plot_fi_curves(
        curves_list, amps.cpu().detach().numpy(), labels=labels, title=title, output_file=fi_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-trace-file", required=False)
    parser.add_argument("--extra-trace-label", required=False)
    parser.add_argument("--title", required=False)
    args = parser.parse_args()

    extra_label = (None,)
    extra_trace = None
    title = None
    if args.extra_trace_file:  # assumes that the amps are the same
        extra_trace, _, amps = utils.load_final_traces(args.extra_trace_file)
        extra_label = args.extra_trace_label

    if args.title:
        title = args.title

    if '--sweep' in sys.argv:
        selected_config = meta_sweep.get_meta_params_for_sweep()
    main(selected_config, extra_trace, extra_label, title=title)
