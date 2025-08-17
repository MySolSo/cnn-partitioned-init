# Accelerated Neural Network Initialization via Training of Symmetrically Partitioned Models — Sample Code
This repository contains a small, test‑oriented implementation of the idea described in “Accelerated Neural Network Initialization via Training of Symmetrically Partitioned Models.”

In short: we split each image horizontally into two tiles, train two smaller CNNs (one per tile), then merge their weights into a single full‑image CNN and fine‑tune it. Finally, we compare this merged model against a reference model trained end‑to‑end in the normal way.

This code is meant for experimentation and learning, not production. It uses simple weight concatenation/averaging rules and a fixed network layout to keep the concept easy to follow.

## sample.ipynb
- Dataset splitting: takes CIFAR images and slices them horizontally into two equal parts.
- Two “split” CNNs: same architecture, reduced channels/units to match the split.
- A “merged” CNN: a full‑image model that receives weights from the split models.
- Weight merge logic:
  - Conv kernels are concatenated across channels to stitch the feature extractors together.
  - Dense layer weights are averaged.
- Training and comparison:
  - Train split models on split data, merge, on full images.
  - Train a reference model from scratch on full images and compare.

## docker-example
- using the same approach as sample.ipynb
- containerized the creation of the models

## Notes and gotchas

- This is not “true distributed training”:
  - It simulates the idea by training separate models on slices and merging weights. There’s no cross‑device orchestration here.
- Performance expectations:
  - This is a small CNN on CIFAR; don’t expect SOTA results.
  - The point is to illustrate the merging strategy and observe whether the merged+fine‑tuned model can reach or surpass a baseline trained from scratch.