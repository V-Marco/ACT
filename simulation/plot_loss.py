import json
import matplotlib.pyplot as plt

with open('train_stats_repeat_1.json','r') as fp:
    d = json.load(fp)

plt.plot(d['train_loss'], label='Train Loss')
plt.plot(d['test_loss'], label='Test Loss')
plt.legend()
plt.show()

