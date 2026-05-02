import torchvision.transforms as transforms
device='mps'
resize=transforms.Resize((244, 244))
resize=resize.to(device)