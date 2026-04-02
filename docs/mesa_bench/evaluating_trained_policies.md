# Evaluating Trained Policies

Our evaluation pipeline involves launching separate environment and policy servers, which we describe in detail in the [running the inference server](../using_mesa/running_inference.md) documentation.

## OpenPI models

Our `openpi` fork which supports MESA inference is available [here](https://github.com/pairlab/openpi-mesa). To run inference with our trained models, clone this repository and install it as described in the README. Then, download the checkpoints from [Hugging Face](https://huggingface.co/collections/albertwilcox/mesa). Finally, from the `openpi-mesa` repository, run

```bash
uv run scripts/serve_policy.py \
  --port <port> \
  policy:checkpoint \
  --policy.config=<config_name, one of {pi0_mesa, pi05_mesa}> \
  --policy.dir=<path_to_checkpoint>
```

We open source the following models:
- [MESA-pi0](https://huggingface.co/albertwilcox/mesa-pi0)
- [MESA-pi0-fast](https://huggingface.co/albertwilcox/mesa-pi0-fast)
- [MESA-pi05](https://huggingface.co/albertwilcox/mesa-pi05)


## GR00T models

Coming soon.