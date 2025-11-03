## Train Your Vim
`bash vim/scripts/pt-vim-b.sh`

## Train Your Vim at Finer Granularity
`bash vim/scripts/ft-vim-b.sh`

## Evaluate Your Vim

To evaluate `Vim-Ti` on ImageNet-1K, run:
```bash
python main.py --eval --resume /path/to/ckpt --model vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2 --data-path /path/to/imagenet
```

## Transfer Entropy Calculation Example

This example shows how to compute **Transfer Entropy (TE)** during model evaluation using the `evaluate` function.

```python
# Evaluate without skipping any layers
test_stats, init_entropy = evaluate(
    data_loader_val,
    model_ema.ema,
    device,
    amp_autocast,
    skip_blocks=[]
)

# Example: Skip block 5 and block 10, then observe entropy changes
test_stats_pruned, entropy_pruned = evaluate(
    data_loader_val,
    model_ema.ema,
    device,
    amp_autocast,
    skip_blocks=[5, 10]
)

# Compute Transfer Entropy difference
transfer_entropy = torch.abs(init_entropy - entropy_pruned)
print(f"ΔTransfer Entropy after pruning: {transfer_entropy:.4f}")
```


