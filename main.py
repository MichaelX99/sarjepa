import typer
from lomar_vit import build_lomar_vit_tiny
import torch

app = typer.Typer()

@app.command()
def pretrain():
    model = build_lomar_vit_tiny()
    x = torch.zeros(4, 1, 224, 224)

    out = model(x)



if __name__ == "__main__":
    app()