import torchvision.transforms as transforms

def train_transform():
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomCrop(28, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),  
    ])

def val_transform():
    return transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
    ])

