from __future__ import annotations


def build_model(name: str = "cnn", num_classes: int = 5):
    import torch
    from torch import nn

    class MultiTaskHead(nn.Module):
        def __init__(self, in_features: int) -> None:
            super().__init__()
            self.score = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
            self.classes = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.score(x), self.classes(x)

    class ActivityCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, 5, padding=2),
                nn.BatchNorm2d(24),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1),
                nn.BatchNorm2d(48),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1),
                nn.BatchNorm2d(96),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.head = MultiTaskHead(96)

        def forward(self, x):
            return self.head(self.features(x))

    if name == "cnn":
        return ActivityCNN()

    if name == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        features = model.fc.in_features
        model.fc = MultiTaskHead(features)
        return model

    raise ValueError(f"Unknown PyTorch model: {name!r}. Expected 'cnn' or 'resnet18'.")
