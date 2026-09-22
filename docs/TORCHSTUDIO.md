# TorchStudio

TorchStudio is optional and is not required by RF CrowdSense.

There are two different things named `torchstudio` online:

1. TorchStudio, the standalone PyTorch GUI/IDE from torchstudio.ai.
2. A separate package named `torchstudio` on PyPI.

This repository does not add the PyPI package to `pyproject.toml` because it is not needed by the RF pipeline and can be confused with the standalone GUI.

If you want the standalone TorchStudio GUI, install it separately from:

https://www.torchstudio.ai/download/

The project models are ordinary PyTorch modules, so the custom CNN and ResNet-18 model code can be reused from TorchStudio or another IDE without making TorchStudio a runtime dependency.
