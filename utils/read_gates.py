import torch

gates = torch.load('results/long_term_forecast_ChimeraTransformer_Public_Health_12/gates.pt')

print(gates[-1].shape)
print(gates[-1][0,:,:5])
print(gates[0][0,:,:5])
