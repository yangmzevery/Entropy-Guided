# EfficientViT for Image Classification

The codebase implements the image classification with EfficientViT.

## Get Started


### Install requirements

Run the following command to install the dependences:

```bash
pip install -r requirements.txt
```

### Data preparation


- ImageNet-1k

ImageNet-1k contains 1.28 M images for training and 50 K images for validation.
The images shall be stored as individual files:


## Evaluation

Run the following command to evaluate a pre-trained EfficientViT-M5 on ImageNet val with a single GPU:
```bash
python main.py --eval --model EfficientViT_M5 --resume ./efficientvit_m5.pth --data-path $PATH_TO_IMAGENET
```

## Training


```bash
python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py --model EfficientViT_M5 --data-path $PATH_TO_IMAGENET --dist-eval
```


# # Transfer Entropy Calculation Example

This section demonstrates how to compute **Transfer Entropy (TE)** during model evaluation using the `evaluate` function.

```python
# Evaluate without skipping any layers
test_stats, init_entropy = evaluate(data_loader_val, model, device, skip_blocks=[])

# Example: Skip block/attention/mlp 2 and block/attention/mlp 5, then observe entropy changes
test_stats_pruned, entropy_pruned = evaluate(data_loader_val, model, device, skip_blocks=[2, 5])

# Compute Transfer Entropy difference
transfer_entropy = torch.abs(init_entropy - entropy_pruned)
print(f"ΔTransfer Entropy after pruning: {transfer_entropy:.4f}")
```