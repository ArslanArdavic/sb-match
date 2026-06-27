import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import v2

def load_afhq_image(
        path="/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/val/cat/flickr_cat_000526.jpg",
        downsize=64,
        ):
    image_tensor = torchvision.io.read_image(path)          # (3, H, W) uint8 [0,255]
    image_tensor = image_tensor.float() / 255.0             # -> float [0, 1]
    image_tensor = TF.resize(image_tensor, [downsize, downsize], antialias=True)
    image_tensor = image_tensor * 2.0 - 1.0                 # [0,1] -> [-1, 1]
    return image_tensor.unsqueeze(0)                        # (1, 3, H, W)

def load_afhq_train(
        datadir="/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/train/",    
        cls="cat",
        downsize=64,
        batch_size = 32,
    ):
    transform = v2.Compose(
        ([v2.Resize([downsize, downsize], antialias=True)] if downsize != 512 else [])  # AFHQ is native 512
        + [
            v2.ToImage(),                              # PIL -> tensor image
            v2.ToDtype(torch.float32, scale=True),     # uint8 [0,255] -> float [0,1]
            v2.Normalize([0.5] * 3, [0.5] * 3),        # -> [-1, 1]
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root=datadir, transform=transform)

    idx = dataset.class_to_idx[cls]                # keep only the requested class
    dataset.samples = [s for s in dataset.samples if s[1] == idx]

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return dataloader


def load_afhq_val(
        datadir="/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/val/",    
        cls="cat",
        downsize=64,
        batch_size = 32,
    ):
    transform = v2.Compose(
        ([v2.Resize([downsize, downsize], antialias=True)] if downsize != 512 else [])  # AFHQ is native 512
        + [
            v2.ToImage(),                              # PIL -> tensor image
            v2.ToDtype(torch.float32, scale=True),     # uint8 [0,255] -> float [0,1]
            v2.Normalize([0.5] * 3, [0.5] * 3),        # -> [-1, 1]
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root=datadir, transform=transform)

    idx = dataset.class_to_idx[cls]                # keep only the requested class
    dataset.samples = [s for s in dataset.samples if s[1] == idx]

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
    )
    return dataloader