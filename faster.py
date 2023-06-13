import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Optimizer

import click
import wandb
from tqdm import tqdm
import torchvision
from torchvision import transforms as T
from einops import rearrange, reduce, repeat, pack
from flash_attn.flash_attention import FlashMHA
from accelerate import Accelerator

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attention = FlashMHA(dim, heads)

    def forward(self, x):
        x, _ = self.attention(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.attention = Attention(dim=dim, heads=heads)
        self.prenorm = nn.LayerNorm(dim)
        self.postnorm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        x = self.attention(self.prenorm(x)) + x
        x = self.ffn(self.postnorm(x)) + x
        return x

class ViT(nn.Module):
    def __init__(self, dim=768, patch=4, size=32):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        self.layers = nn.Sequential(*[Transformer() for _ in range(12)])
        self.wpe = nn.Parameter(torch.zeros((size // patch) ** 2 + 1, dim))
        self.head = nn.Linear(dim, 10)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        x = self.patch(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        cls = repeat(self.cls, "1 1 c -> b 1 c", b=len(x))
        x, _ = pack((cls, x), "b * c")
        x = self.layers(x + self.wpe)
        x = self.head(x[:,0,:])

        return x
    
@click.command()
@click.option("--lr", default=3e-4)
@click.option("--batch_size", default=256)
@click.option("--epochs", default=10)
@click.option("--patch", default=4)
@click.option("--size", default=32)
@click.option("--heads", default=12)
@click.option("--dim", default=768)
def train(**cfg):
    accelerator = Accelerator(mixed_precision="bf16")
    trainds = torchvision.datasets.CIFAR10(root="data", download=True, train=True, transform=T.ToTensor())
    valds = torchvision.datasets.CIFAR10(root="data", download=True, train=False, transform=T.ToTensor())
    trainloader = DataLoader(trainds, batch_size=cfg["batch_size"], shuffle=True)
    valloader = DataLoader(valds, batch_size=cfg["batch_size"], shuffle=False)

    model = ViT(cfg["dim"], cfg["patch"], cfg["size"])
    opt = AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    model, opt, trainloader, valloader = accelerator.prepare(model, opt, trainloader, valloader)

    for epoch in tqdm(range(10)):
        model.train()
        for x, y in tqdm(trainloader, total=len(trainloader)):
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            accelerator.backward(loss)
            opt.step()
        
        with torch.no_grad():
            model.eval()
            correct, total = 0, 0
            for x, y in tqdm(valloader, total=len(valloader)):
                logits = model(x)
                pred = torch.argmax(logits, dim=-1)
                correct += (pred == y).sum()
                total += len(y)
            acc = correct / total
            print(f"Epoch {epoch} | Acc: {acc}")
            

if __name__ == "__main__":
    train()