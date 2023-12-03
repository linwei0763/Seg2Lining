from network.lfa_lin_v1 import LFA

config = 1

net = LFA(10, 20, config)
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())