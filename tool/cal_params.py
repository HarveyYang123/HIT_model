import sys
import torch

param = torch.load(sys.argv[1])

sums = 0

for item in param.items():
    shape = item[1].shape
    num = 1
    for v in shape:
        num *= v
    if len(shape) > 0:
        sums += num
print(sums)
