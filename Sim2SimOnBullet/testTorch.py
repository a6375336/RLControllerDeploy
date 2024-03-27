import torch 

a = torch.Tensor([1,2,3])
print(a)

b = torch.Tensor()
b = torch.cat((b , a), dim=-1)
b = torch.cat((b , a), dim=-1)
print(b)
b = torch.reshape(b, (2,-1))
print(b)
print(b[:,:3])