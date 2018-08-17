# The config file for training and evaluating pvanet

"""
Choose a network structure to use:
    'regular': The ordinary PVANet with scale implemented independently, in 'pvanet.py/pvanet'
    'lite': The PVANet-lite network, it seems no CReLU is used, in 'pvanet.py/pvanet_lite'
    'trial': The trial version where no scale is used, other parts remains same with regular PVANet, in 'pvanet_try.py/pvanet'
    'noincep': No inception block version of PVANet, in 'pvanet_try.py/pvanet_crelu_only'
"""
net_type = 'noincep'
fatness = 2
concat = True
dropout = 0.0


# Adam optimizer parameters
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-2
