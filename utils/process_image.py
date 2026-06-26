import torchvision
import torchvision.transforms.functional as TF

def load_afhq_image(
        path="/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/val/cat/flickr_cat_000526.jpg",
        downsize=64,
        ):
    image_tensor = torchvision.io.read_image(path)          # (3, H, W) uint8 [0,255]
    image_tensor = image_tensor.float() / 255.0             # -> float [0, 1]
    image_tensor = TF.resize(image_tensor, [downsize, downsize], antialias=True)
    image_tensor = image_tensor * 2.0 - 1.0                 # [0,1] -> [-1, 1]
    return image_tensor.unsqueeze(0)                        # (1, 3, H, W)
