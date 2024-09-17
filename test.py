import pickle
import torch

a = torch.load('/public/home/zhuyuchen530/projects/cvpr24/2sM2F_cont_distillation/output/ps/test/step2/fake_query.pkl', map_location = 'cuda')
print(a.keys())
