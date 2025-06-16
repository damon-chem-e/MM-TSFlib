import torch

gates = torch.load('results/long_term_forecast_ChimeraTransformer_Public_Health_12_raw_skip_dual_gate/gates.pt')

# first_30 = gates[-1][0, 0, :30].tolist()
# last_30 = gates[-1][0, 0, 482:].tolist()
# print("Avg of first 30 gates: " + str(sum(first_30)/30))
# print("Avg of final 30 gates: " + str(sum(last_30)/30))

first_30 = gates[-1][0][0, 0, :30].tolist()
last_30 = gates[-1][0][0, 0, 482:].tolist()
print("Avg of first 30 gates: " + str(sum(first_30)/30))
print("Avg of final 30 gates: " + str(sum(last_30)/30))