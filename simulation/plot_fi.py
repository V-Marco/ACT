import argparse
import os 

from simulation_configs import selected_config

from act import analysis, utils


def main(extra_trace, extra_trace_label):
    config = selected_config

    traces_file = os.path.join(config['output']['folder'], config['run_mode'], 'traces.h5')
    simulated_traces, target_traces, amps = utils.load_final_traces(traces_file)

    simulated_curve = utils.get_fi_curve(simulated_traces, amps)
    target_curve = utils.get_fi_curve(target_traces, amps)

    simulated_label = config['output']['simulated_label']
    target_label = config['output']['target_label']

    curves_list = [target_curve.cpu().detach().numpy(), simulated_curve.cpu().detach().numpy()]
    labels = [target_label, simulated_label]

    if extra_trace is not None:
        curves_list.append(extra_trace)
        labels.append(extra_trace_label)

    analysis.plot_fi_curves(curves_list, amps.cpu().detach().numpy(), labels=labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra-trace-file', required=False)
    parser.add_argument('--extra-trace-label', required=False)

    args = parser.parse_args() 

    extra_label = None,
    extra_trace = None
    if args.extra_trace_file: # assumes that the amps are the same
        extra_trace, _, amps = utils.load_final_traces(args.extra_trace_file)
        extra_trace = utils.get_fi_curve(extra_trace, amps)
        extra_trace = extra_trace.cpu().detach().numpy()
        extra_label = args.extra_trace_label


    main(extra_trace, extra_label)

