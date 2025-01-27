import hydra
import torch
import wandb
import math
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
import torch.optim as optim
from omegaconf import DictConfig
from model.gpt import GPT, GPTConfig


import distributed
import dataset
from utils import setup_logger


def get_lr(iteration, args):
    if iteration < args.training.warmup_iters:
        return args.training.lr * iteration / args.training.warmup_iters

    if iteration > args.training.lr_decay_iters:
        return args.training.min_lr

    decay_ratio = (iteration - args.training.warmup_iters) / (
        args.training.lr_decay_iters - args.training.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.training.min_lr + coeff * (args.training.lr - args.training.min_lr)


@torch.no_grad()
def estimate_loss(model, args, ctx, device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(args.training.eval_iters)
        for k in range(args.training.eval_iters):
            X, Y = dataset.get_batch(args, split, device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@hydra.main(version_base=None, config_path="conf", config_name="configs")
def main(args: DictConfig):
    torch.manual_seed(args.settings.seed)
    device = (
        "cuda" if (not args.settings.no_cuda and torch.cuda.is_available()) else "cpu"
    )
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )
    logger = setup_logger("gpt-enwiki")
    vocab_sz = dataset.get_data(args, logger)

    model_args = dict(
        n_layer=args.model.n_layer,
        n_head=args.model.n_head,
        n_embd=args.model.n_embd,
        block_size=args.model.block_size,
        bias=args.model.bias,
        vocab_size=vocab_sz,
        dropout=args.model.dropout,
        block_type=args.model.block_type,
    )
    config = GPTConfig(**model_args)
    model = GPT(config)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer, n_params = model.configure_optimizers(
        args.training.weight_decay,
        args.training.lr,
        (args.training.beta1, args.training.beta2),
        device,
    )

    if args.settings.compile:
        unopt_model = model
        model = torch.compile(model)

    wandb.init(
        project="sakana-gpt",
        name=f"{args.wandb.name}",
        config={
            "block_type": args.model.block_type,
            "n_layers": args.model.n_layer,
            "n_embd": args.model.n_embd,
            "n_head": args.model.n_head,
            "n_params": n_params,
        },
    )
    progress_bar = tqdm(range(0, args.training.n_iters), desc="Steps...")
    for it in range(args.training.n_iters):
        X, Y = dataset.get_batch(args, "train", device)
        lr = get_lr(it, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if it % args.training.eval_interval == 0:
            losses = estimate_loss(model, args, ctx, device)
            logger.info(
                f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            wandb.log(
                {
                    "iter": it,
                    "train/loss": losses["train"],
                    "train/bpc": losses["train"] / torch.log(torch.tensor(2)),
                    "val/loss": losses["val"],
                    "val/bpc": losses["val"] / torch.log(torch.tensor(2)),
                    "lr": lr,
                }
            )

        for micro_step in range(args.training.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / args.training.gradient_accumulation_steps

            X, Y = dataset.get_batch(args, "train", device)

            scaler.scale(loss).backward()

        if args.training.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(model.parameters(), args.training.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        progress_bar.update(1)

    # Final Test
    if args.settings.test:
        results = dataset.evaluate(model, args, ctx, device)
        wandb.log(
            {
                "test/loss": results,
                "test/bpc": results / torch.log(torch.tensor(2)),
            }
        )
    if args.settings.heatmap:
        from extractor import ActivationExtractor

        extractor = ActivationExtractor(model, ctx, args)
        extractor.extract_activations(device)


if __name__ == "__main__":
    main()
