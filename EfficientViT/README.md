# EfficientViT for Object Detection

The codebase implements the object detection with EfficientViT.

## Get Started


### Install requirements

Run the following command to install the dependences:

```bash
pip install -r requirements.txt
```

### Data preparation


- ImageNet-1k
- MS-COCO

## Training


```bash
python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py --model EfficientViT_M5 --data-path $PATH_TO_IMAGENET --dist-eval
```

### After pre-training on ImageNet-1K, the pruned model should be fine-tuned on MS-COCO.

```bash
 torchrun --nproc_per_node=8 tools/train.py \
 configs/retinanet/retinanet_efficientvit_m5_fpn_1x_coco.py \
 --launcher pytorch \
 --work-dir
```

## Transfer Entropy Calculation Example

This section demonstrates how to compute **Transfer Entropy (TE)** during model evaluation using the `evaluate` function.

```python
# Evaluate without skipping any layers or blocks
test_stats, init_entropy = evaluate(data_loader_val, model, device, skip_blocks=[])

# Example: Skip block/attention/mlp 2 and block/attention/mlp 5, then observe entropy changes
test_stats_pruned, entropy_pruned = evaluate(data_loader_val, model, device, skip_blocks=[2, 5])

# Compute Transfer Entropy difference
transfer_entropy = torch.abs(init_entropy - entropy_pruned)
print(f"ΔTransfer Entropy after pruning: {transfer_entropy:.4f}")
```
