import re
import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = "prompt_weight_tune_test.txt"

# Format like `pred_len: [losses]`
losses = {
    12: [],
    24: [],
    36: [],
    48: []
}

with open(f"results/{FILE_NAME}", "r") as file:
    nums = re.findall(r"\nmse:*([-+]?[0-9]*\.?[0-9]+)", file.read())
    print(nums)
    
    for i in range(len(nums)):
        losses[((i % 4) + 1) * 12].append(float(nums[i])) # Sort into correct pred_len list

x = np.arange(0.1, 0.85, 0.05)
for k, v in losses.items():
    plt.plot(x, v, label=str(k))
    min_val = min(v)
    plt.plot(x[v.index(min_val)], min_val, marker='o', color='red')

plt.title("MSE Loss vs prompt_weight (red dot is minimum MSE)")
plt.xlabel("prompt_weight")
plt.ylabel("MSE Loss")
plt.legend(title="Forecast horizon")
plt.show()
        