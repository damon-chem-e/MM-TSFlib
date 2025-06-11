import torch

gates = torch.load('results/long_term_forecast_Public_Health_12_ChimeraTransformer_custom_ftS_sl24_ll12_pl12_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/gates.pt')

print(gates[-1].shape)
print(gates[-1][0,:,:])
print(gates[0][0,:,:])
