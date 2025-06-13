import re
import matplotlib.pyplot as plt
import numpy as np

def get_loss(filename: str) -> list[float]:
    """Parse loss from file.

    Args:
        filename (str): filename with extension

    Returns:
        list[float]: list of losses in order
    """
    nums = None
    with open(f"results/{filename}", "r") as file:
        nums = re.findall(r"\nmse:*([-+]?[0-9]*\.?[0-9]+)", file.read())
        
    return [float(n) for n in nums]

def grid_heads_layers():
    FILE_NAME = "grid_heads_layers.txt"
    loss = get_loss(FILE_NAME)
    
    xs = np.arange(4, 13)
    plt.figure()
    plt.plot(xs, loss[:9], label="12", alpha=0.2)
    plt.plot(xs, loss[18:27], label="24", alpha=0.2)
    plt.plot(xs, loss[36:45], label="36", alpha=0.2)
    plt.plot(xs, loss[54:63], label="48", alpha=0.2)
    avg = np.array(loss[:9] + loss[18:27] 
                   + loss[36:45] + loss[54:63]).reshape(4, 9).mean(axis=0)
    plt.plot(xs, avg, label="Avg")
    plt.plot(np.argmin(avg) + 4, np.min(avg), marker='o')
    
    plt.xlabel("Num heads")
    plt.ylabel("MSE Loss")
    plt.legend(title="Pred_len", loc="lower left")
    plt.title("Num heads vs MSE Loss (2 layers)")
    
    plt.figure()
    plt.plot(xs, loss[9:18], label="12", alpha=0.2)
    plt.plot(xs, loss[27:36], label="24", alpha=0.2)
    plt.plot(xs, loss[45:54], label="36", alpha=0.2)
    plt.plot(xs, loss[63:], label="48", alpha=0.2)
    avg = np.array(loss[9:18] + loss[27:36] 
                   +loss[45:54] + loss[63:]).reshape(4, 9).mean(axis=0)
    plt.plot(xs, avg, label="Avg")
    plt.plot(np.argmin(avg) + 4, np.min(avg), marker='o')
    
    plt.xlabel("Num heads")
    plt.ylabel("MSE Loss")
    plt.legend(title="Pred_len", loc="lower left")
    plt.title("Num heads vs MSE Loss (4 layers)")
    
grid_heads_layers()

def grid_search():
    FILE_NAME = "grid_search_iTrans.txt"
    
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

def iTrans_encoder_search():
    FILE_NAME = "grid_search_iTrans.txt"
    
    nums = []
    with open(f"results/{FILE_NAME}", "r") as file:
        nums = re.findall(r"\nmse:*([-+]?[0-9]*\.?[0-9]+)", file.read())
        nums = [float(n) for n in nums]

    x = np.arange(2, 9)
    plt.figure()
    plt.plot(x, nums[:7], alpha=0.2, label="12")
    plt.plot(x, nums[7:14], alpha=0.2, label="24")
    plt.plot(x, nums[14:21], alpha=0.2, label="36")
    plt.plot(x, nums[21:29], alpha=0.2, label="48")
    plt.legend(title="Step Size")

    nums = list(np.array(nums).reshape(4, 7).mean(axis=0))
    min_val = min(nums)
    min_idx = nums.index(min_val)
    plt.xlabel("Num iTransformer Encoder Layers")
    plt.ylabel("MSE Loss")
    plt.title("Num iTransformer Encoder Layers vs MSE Loss")
    plt.plot(x, nums)
    plt.plot(min_idx+2, min_val, marker='o')

iTrans_encoder_search()
plt.show()