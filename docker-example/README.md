# This is a small example on docker used for simulating the docker distribution

This setup trains two “split” CNNs in parallel (each on a different horizontal half of the image), then merges their weights into a full model, fine-tunes it, and compares against a baseline trained from scratch. Everything runs in containers and shares artifacts via a mounted folder.

### What you get
- Parallel training of split models (one per image half)
- Merging into a full model and fine-tuning
- Baseline training from scratch for comparison
- Saved weights, histories, metrics, and plots

## Notes and gotchas

- This is not “true distributed training”:
  - It simulates the idea by training separate models on slices and merging weights. There’s no cross‑device orchestration here, just containerization.
- Performance expectations:
  - This is a small CNN on CIFAR; don’t expect SOTA results.
  - The point is to illustrate the merging strategy and observe whether the merged+fine‑tuned model can reach or surpass a baseline trained from scratch.