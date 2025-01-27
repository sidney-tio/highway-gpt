import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

from model.gpt import AttentionTranspose


class ActivationExtractor:
    """
    Extractor Class to collect and visualize the outputs of sigmoid and softmax operations during inference
    """

    def __init__(self, model, ctx, args):
        self.model = model
        self.ctx = ctx
        self.args = args.copy()
        self.corpus = self.load_corpus()

        self.activations = {}

    def load_corpus(self):
        corpus_path = os.path.join(self.args.training.data, "corpus.pt")
        return torch.load(corpus_path)

    def get_activation(self, name):
        def hook(module, input, output):
            if isinstance(module, nn.Sigmoid):
                self.activations[f"sigmoid_{name}"] = output.detach()
            elif isinstance(module, AttentionTranspose):
                self.activations[f"softmax_{name}"] = output.detach()

        return hook

    @torch.no_grad()
    def extract_activations(self, device):
        self.activations = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Sigmoid, AttentionTranspose)):
                module.register_forward_hook(self.get_activation(name))

        X = self.get_sample(self.args, device)
        with self.ctx:
            outputs = self.model(X, None)

        names = self.get_names(self.activations.keys())
        for sigmoid_names, softmax_names, layer_id in names:
            sigmoid_activations = None
            softmax_activations = None
            if sigmoid_names:
                sigmoid_activations = (
                    self.activations[sigmoid_names].to(torch.float32).cpu().numpy()
                )
            if softmax_names:
                softmax_activations = (
                    self.activations[softmax_names].to(torch.float32).cpu().numpy()
                )
            for seq in range(len(softmax_activations)):
                layer_name = (
                    f"{self.args.model.block_type}_attn_block_{layer_id}_seq_{seq}"
                )
                self.plot_activations(
                    sigmoid_activations, softmax_activations, layer_name, seq
                )

    def get_sample(self, args, device):
        self.model.eval()
        data = np.memmap(
            os.path.join(args.training.data, f"train.bin"), dtype=np.uint16, mode="r"
        )

        idx = [0 + i * args.model.block_size for i in range(5)]
        X = torch.stack(
            [
                torch.from_numpy((data[i : i + args.model.block_size]).astype(np.int64))
                for i in idx
            ]
        )
        if device == "cuda":
            X = X.pin_memory().to(device, non_blocking=True)
        else:
            X = X.to(device)

        return X

    def plot_activations(
        self, sigmoid_activations, softmax_activations, layer_name, seq
    ):
        available_activations = []
        if sigmoid_activations is not None:
            available_activations.append(
                ("Sigmoid", sigmoid_activations[seq], "viridis")
            )
        if softmax_activations is not None:
            available_activations.append(
                ("Softmax", softmax_activations[seq], "plasma")
            )

        num_plots = len(available_activations)

        if num_plots == 0:
            print(f"No activation data available for {layer_name}")
            return

        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 8 * num_plots))
        fig.suptitle(
            f"Activation{'s' if num_plots > 1 else ''} ({layer_name})", fontsize=16
        )

        if num_plots == 1:
            axes = [axes]

        for idx, (title, activations, cmap) in enumerate(available_activations):
            im = axes[idx].imshow(activations, aspect="auto", cmap=cmap)
            axes[idx].set_title(title)
            axes[idx].set_xlabel("Hidden Dimension")
            axes[idx].set_ylabel("Sequence Index")
            plt.colorbar(im, ax=axes[idx], label="Activation Value")

        plt.tight_layout()
        figname = os.path.join(self.args.logging.log_dir, f"{layer_name}.png")
        plt.savefig(figname)
        plt.close(fig)

    def get_names(self, keys):
        layer_dict = {}

        for key in keys:
            if "sigmoid" in key:
                layer_num = int(key.split(".h.")[1].split(".")[0])
                if layer_num not in layer_dict:
                    layer_dict[layer_num] = [None, None, None]
                layer_dict[layer_num][0] = key
                layer_dict[layer_num][2] = layer_num
            elif "softmax" in key:
                layer_num = int(key.split(".h.")[1].split(".")[0])
                if layer_num not in layer_dict:
                    layer_dict[layer_num] = [None, None, None]
                layer_dict[layer_num][1] = key
                layer_dict[layer_num][2] = layer_num

        return [
            (layer_dict[i][0], layer_dict[i][1], layer_dict[i][2])
            for i in sorted(layer_dict.keys())
        ]
