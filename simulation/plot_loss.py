import json
import matplotlib.pyplot as plt

with open('train_stats_repeat_0.json','r') as fp:
    d = json.load(fp)

plt.plot(d['loss'])
plt.show()

