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
from einops import rearrange, reduce
from flash_attn.flash_attention import FlashMHA



# class Attention(nn.Module):
#     def __init__(self, dim, heads=8):
#         super().__init__()
#         self.heads = heads
#         self.dim = dim
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.out = nn.Linear(dim, dim)

#     def forward(self, x):
#         # Thanks Copilot!
#         q, k, v = self.qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
#         dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) / self.dim ** 0.5
#         attn = F.softmax(dots, dim=-1)
#         out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         return self.out(out)

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
            nn.SiLU(),
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
        self.wpe = nn.Parameter(torch.zeros((size // patch) ** 2, dim))
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.layers(x + self.wpe)
        x = reduce(x, "b n c -> b c", "mean")
        x = self.head(x)

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
    wandb.init(project="vit", config=cfg)
    trainds = torchvision.datasets.CIFAR10(root="data", download=True, train=True, transform=T.ToTensor())
    valds = torchvision.datasets.CIFAR10(root="data", download=True, train=False, transform=T.ToTensor())
    trainloader = DataLoader(trainds, batch_size=cfg["batch_size"], shuffle=True)
    valloader = DataLoader(valds, batch_size=cfg["batch_size"], shuffle=False)

    dtype = torch.bfloat16
    model = ViT(cfg["dim"], cfg["patch"], cfg["size"])
    model.to("cuda").to(dtype)
    opt = AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()


    for epoch in tqdm(range(10)):
        model.train()
        for x, y in tqdm(trainloader, total=len(trainloader)):
            x = x.to("cuda").to(dtype)
            y = y.to("cuda")
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        
        with torch.no_grad():
            model.eval()
            correct, total = 0, 0
            for x, y in tqdm(valloader, total=len(valloader)):
                x = x.to("cuda").to(dtype)
                y = y.to("cuda")
                logits = model(x)
                pred = torch.argmax(logits, dim=-1)
                correct += (pred == y).sum()
                total += len(y)

            acc = correct / total
            wandb.log({"acc": acc.item()})
            

if __name__ == "__main__":
    train()