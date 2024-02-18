import argparse
import os

from simulation_configs import selected_config
import sys
sys.path.append("../")
from act import analysis, utils


def main(extra_trace, extra_trace_label, title=None):
    config = selected_config

    inj_dur = config["simulation_parameters"]["h_i_dur"]

    output_folder = utils.get_output_folder_name(config)
    target_params = config["optimization_parameters"].get("target_params")
    if(config["run_mode"] == "segregated"):
        segregation_index = utils.get_segregation_index(config)
        segregation_dir = f"seg_module_{segregation_index+1}/"
        model_data_dir = os.path.join(output_folder, segregation_dir)
    else:
        model_data_dir = output_folder

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

    fi_file = f"{output_folder[2:-1]}_FI"
    analysis.plot_fi_curves(
        curves_list, amps.cpu().detach().numpy(), labels=labels, title=title, output_file=model_data_dir + fi_file
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

    main(extra_trace, extra_label, title=title)
