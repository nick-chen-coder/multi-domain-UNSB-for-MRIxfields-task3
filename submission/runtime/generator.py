from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import (
    AdaIN2d,
    ConditionalUpBlock,
    ResidualBlock,
    ResidualDownBlock,
    crop_model,
    pad_model,
    position_fourier,
)


class GeneratorCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = nn.Embedding(5, 64)
        self.modality = nn.Embedding(3, 64)
        self.source_field = nn.Embedding(5, 64)
        self.target_field = nn.Embedding(5, 64)
        self.position = nn.Sequential(nn.Linear(16, 64), nn.SiLU(), nn.Linear(64, 64))
        self.fusion = nn.Sequential(nn.Linear(320, 256), nn.SiLU(), nn.Linear(256, 64))

    def forward(self, step, modality, source_field, target_field, position):
        if not torch.is_tensor(step):
            step = torch.full_like(modality, int(step))
        values = (
            self.step(step),
            self.modality(modality),
            self.source_field(source_field),
            self.target_field(target_field),
            self.position(position_fourier(position)),
        )
        return self.fusion(torch.cat(values, dim=1))


class Generator(nn.Module):
    feature_channels = (64, 128, 256, 512, 512)

    def __init__(self, bottleneck_blocks: int = 2):
        super().__init__()
        if int(bottleneck_blocks) < 1:
            raise ValueError("Generator needs at least one bottleneck residual block")
        self.bottleneck_blocks = int(bottleneck_blocks)
        self.condition = GeneratorCondition()
        self.stem = nn.Conv2d(1, 32, 3, padding=1)
        self.encoder = nn.ModuleList(
            [
                ResidualDownBlock(32, 64),
                ResidualDownBlock(64, 128),
                ResidualDownBlock(128, 256),
                ResidualDownBlock(256, 512),
                ResidualDownBlock(512, 512),
            ]
        )
        shared_bottleneck_blocks = min(self.bottleneck_blocks, 2)
        self.bottleneck = nn.Sequential(
            *(ResidualBlock(512) for _ in range(shared_bottleneck_blocks))
        )
        self.decoder = nn.ModuleList(
            [
                ConditionalUpBlock(512, 512, False),
                ConditionalUpBlock(512, 512, False),
                ConditionalUpBlock(512, 512, True, skip_channels=512),
                ConditionalUpBlock(512, 256, True, skip_channels=256),
                ConditionalUpBlock(256, 128, True, skip_channels=128),
                ConditionalUpBlock(128, 64, True, skip_channels=64),
                ConditionalUpBlock(64, 32, True),
            ]
        )
        self.output_norm = AdaIN2d(32)
        self.output = nn.Conv2d(32, 1, 1)
        if self.bottleneck_blocks > shared_bottleneck_blocks:
            # Keep every shared parameter and subsequent model initialization
            # identical between capacity-ablation runs with the same seed.
            extra_seed = torch.initial_seed() + 104729
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(extra_seed)
                self.bottleneck.extend(
                    ResidualBlock(512)
                    for _ in range(self.bottleneck_blocks - shared_bottleneck_blocks)
                )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.stem(pad_model(x))
        features = []
        for block in self.encoder:
            x = block(x)
            features.append(x)
        expected = [(192, 224), (96, 112), (48, 56), (24, 28), (12, 14)]
        if [tuple(item.shape[-2:]) for item in features] != expected:
            raise RuntimeError("generator encoder feature geometry changed")
        return x, features

    def forward(
        self,
        x: torch.Tensor,
        step,
        modality: torch.Tensor,
        source_field: torch.Tensor,
        target_field: torch.Tensor,
        position: torch.Tensor,
        return_features: bool = False,
    ):
        condition = self.condition(step, modality, source_field, target_field, position)
        target_field_condition = self.condition.target_field(target_field)
        h, features = self.encode(x)
        h = self.bottleneck(h)
        decoder_skips = {2: features[3], 3: features[2], 4: features[1], 5: features[0]}
        for index, block in enumerate(self.decoder):
            h = block(h, condition, decoder_skips.get(index))
        h = self.output(F.leaky_relu(self.output_norm(h, target_field_condition), 0.2)).tanh()
        output = crop_model(h)
        return (output, features) if return_features else output


def generator_from_checkpoint_state(state: dict) -> Generator:
    resolved = state.get("resolved_config", {})
    generator_config = resolved.get("model", {}).get("generator", {})
    blocks = generator_config.get("bottleneck_blocks", 2) if isinstance(generator_config, dict) else 2
    return Generator(bottleneck_blocks=int(blocks))

