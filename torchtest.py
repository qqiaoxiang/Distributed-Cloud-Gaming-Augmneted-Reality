import torch
import torchvision
print(torch.__version__,torch.cuda.is_available())
print(torchvision.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.rand(3,3).cuda())