import json
import matplotlib.pyplot as plt

with open('target_v.json','r') as f:
    t = json.load(f)

for trace in t['traces']:
    plt.figure()
    plt.plot(trace)

plt.show()
