from bmtool.singlecell import Passive,run_and_plot
import matplotlib.pyplot as plt
sim = Passive('Cell_Cf', inj_amp=-100., inj_delay=1500., inj_dur=1000., tstop=2500., method='exp2')
title = 'Passive Cell Current Injection'
xlabel = 'Time (ms)'
ylabel = 'Membrane Potential (mV)'
X, Y = run_and_plot(sim, title, xlabel, ylabel, plot_injection_only=True)
plt.gca().plot(*sim.double_exponential_fit(), 'r:', label='double exponential fit')
plt.legend()
plt.show()