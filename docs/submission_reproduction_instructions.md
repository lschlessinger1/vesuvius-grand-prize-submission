# Reproduction instructions

## Overview

Here, you'll find instructions on how to prepare the environment, train, and run inference.

## Getting started

### Prerequisites

#### System requirements 
Training:
- Disk space: ~1Tb
- RAM: 200Gb
- GPUs: 1 node, 8x H100 80Gb HBM3

Inference:
- Disk space: ~50Gb
- RAM: 80Gb
- GPUs: 1 node, 1 V100 (16Gb VRAM)

You can probably train using a single GPU such as an A100 SXM4 40 GB, but the training time will be significantly 
longer and parameters such as batch size, learning rate, and devices may have to be adjusted.

You may need a bit more RAM for inference depending on segment size.

### Installation
#### Docker build
- Image size: 13.4 Gb
- Docker version `24.0.7`

You can build the image by running:

```bash
docker build -t scroll-ink-det-gpu -f docker/scroll-ink-detection-gpu/Dockerfile .
```

#### Docker run

You can run the following to create a new container with an interactive shell:
```bash
docker run -it --rm --gpus all -v /$(pwd)/data:/workspace/data scroll-ink-det-gpu
```

Feel free to change `$(pwd)` to wherever you want to the data downloaded on the host.

> Note: Depending on your system, you may need to increase the shared memory size for this container for 
training. 

We had to use `--shm-size=10g`, making the full docker run command: 

```bash
docker run -it --rm --gpus all -v /$(pwd)/data:/workspace/data --shm-size=10g scroll-ink-det-gpu
```

#### Download and prepare the data

1. Preparing the labels

Unzip the `labels.zip` file and place them in `data/labels` (the mounted volume).

2. Downloading the surface volumes

You can download the data after you have registered for the 
[Vesuvius Challenge](https://scrollprize.org/) (see [data agreement here](https://docs.google.com/forms/d/e/1FAIpQLSf2lCOCwnO1xo0bc1QdlL0a034Uoe7zyjYBY2k33ZHslHE38Q/viewform)).
This will give you the username and password that you'll need for the next step. You must set the environment variables
`USER` and `PASS`. You can create an `.env` file (see `.env.example` for an example; run `cp .env.example .env` to 
create this file) and fill it out. You'd then have to set the environment variables, which can be done as such in a Linux shell:

```bash
export $(grep -v '^#' .env | tr -d '\r' | xargs)
```

From within the container's new shell session, you should download the surface volumes associated with the labels:
```bash
export LABEL_DIR=data/labels
./scripts/download-scroll-surface-vols-by-segment.sh 1 $(python3 tools/print_segment_ids_from_label_dir.py $LABEL_DIR)
```

3. Generate sub-segments.
Now we're ready to create sub-segments (cropped versions of the original segments). You can create them by running:
```bash
python3 vesuvius_challenge_rnd/scroll_ink_detection/evaluation/tools/create_all_subsegments.py --load_in_memory
```
You should now be able to find these sub-segments in the `data/scrolls` directory with `_C<i>` postfixes, where `i`
indicates the `i`th column.

## Usage

### Training

To train a model, you can execute the following:
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection fit --config path/to/config.yaml
```

You'll have to pass a `config.yaml` file for each run.

#### Fast development run (optional)

To check that everything is working, you can run the following fast development run config:
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection fit --config vesuvius_challenge_rnd/scroll_ink_detection/experiment_runner/configs/unet3d_segformer/fast_dev_run_docker.yaml
```

#### Full training
To train on the full dataset, you can train the following models (one run per validation segment).
It will prompt you to sign in to [Weights & Biases](https://wandb.ai/). Feel free to skip this. You can find all the submission 
run configs  under `vesuvius_challenge_rnd/scroll_ink_detection/experiment_runner/configs/unet3d_segformer/submission`.

Run 1 (*3336_C3 as validation):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection fit --config vesuvius_challenge_rnd/scroll_ink_detection/experiment_runner/configs/unet3d_segformer/submission/val_3336_C3.yaml
```

Run 2 (*5753 as validation):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection fit --config vesuvius_challenge_rnd/scroll_ink_detection/experiment_runner/configs/unet3d_segformer/submission/val_5753.yaml
```

Run 3 (*4422_C2 as validation):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection fit --config vesuvius_challenge_rnd/scroll_ink_detection/experiment_runner/configs/unet3d_segformer/submission/val_4422_C2.yaml
```

Run 4 (*4422_C3 as validation):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection fit --config vesuvius_challenge_rnd/scroll_ink_detection/experiment_runner/configs/unet3d_segformer/submission/val_4422_C3.yaml
```

Run 5 (*1321 as validation):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection fit --config vesuvius_challenge_rnd/scroll_ink_detection/experiment_runner/configs/unet3d_segformer/submission/val_1321_no0901c3_no5351.yaml
```

You can find the saved model checkpoints under `/workspace/lightning_logs/version_<v_num>/checkpoints/`. There should only be 
one checkpoint per training run. Note that `v_num` is an automatically generated experiment version ID and can be seen 
in the progress bar. For each run, note the associated `v_num` so that it can be used later on for inference.

To reproduce the run using *3336_C3 with a smaller dataset (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection fit --config vesuvius_challenge_rnd/scroll_ink_detection/experiment_runner/configs/unet3d_segformer/submission/val_3336_C3_reduced.yaml
```

### Inference

With the model checkpoints, you can generate predictions by running:
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference path/to/checkpoint.ckpt <segment id> --infer_z_reversal
```

By default, it will write the output images to `/workspace/outputs`, but you can change that using `-o path/to/new/output/dir`.

Assuming you ran the experiments in the order above and have the checkpoint paths you can run the following to reproduce the predictions:

_20231005123336_C3_ predictions (run 1):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 1 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231005123336_C3 --infer_z_reversal -o outputs/20231005123336_C3
```

_20230519215753_ predictions (run 2):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 2 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20230519215753 --infer_z_reversal -o outputs/20230519215753
```

_20231012184422_superseded_C2_ predictions (run 3):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 3 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231012184422_superseded_C2 --infer_z_reversal -o outputs/20231012184422_superseded_C2
```

_20231012184422_superseded_C3_ predictions (run 4):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 4 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231012184422_superseded_C3 --infer_z_reversal -o outputs/20231012184422_superseded_C3
```

_20231210121321_ predictions (run 5):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 5 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231210121321 --infer_z_reversal -o outputs/20231210121321
```

_20231022170901_C3_ predictions (run 5):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 5 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231022170901_C3 --infer_z_reversal -o outputs/20231022170901_C3
```

_20231106155351_ predictions (run 5):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 5 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231106155351 --infer_z_reversal -o outputs/20231106155351
```

These will all be saved in `outputs/<segment name>`.

#### Run 6 predictions (optional)
To run predictions from run 6 (*3336 as validation with reduced dataset size), you'll need to first ensure any segment that was not included in training or validation 
is downloaded as such:
```bash
./scripts/download-scroll-surface-vols-by-segment.sh 1 20231221180251 20231007101619
```

You might need to generate sub-segments for *1619:

```bash
python3 vesuvius_challenge_rnd/scroll_ink_detection/evaluation/tools/create_all_subsegments.py --load_in_memory --skip_if_exists
```

Now, you are ready to run inference on the additional segments (saved to the same output directory).

_20231221180251_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231221180251 --infer_z_reversal -o outputs/20231221180251
```

_20231031143852_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231031143852 --infer_z_reversal -o outputs/20231031143852
```

_20231005123336_C2_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231005123336_C2 --infer_z_reversal -o outputs/20231005123336_C2
```

_20231022170901_C2_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231022170901_C2 --infer_z_reversal -o outputs/20231022170901_C2
```

_20231022170901_C3_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231022170901_C3 --infer_z_reversal -o outputs/20231022170901_C3
```

_20231022170901_C4_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231022170901_C4 --infer_z_reversal -o outputs/20231022170901_C4
```

_20230929220926_C2_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20230929220926_C2 --infer_z_reversal -o outputs/20230929220926_C2
```

_20230929220926_C3_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20230929220926_C3 --infer_z_reversal -o outputs/20230929220926_C3
```

_20231007101619_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231007101619 --infer_z_reversal -o outputs/20231007101619
```

You can also run *1619 predictions as sub-segments if needed:
_20231007101619_C1_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231007101619_C1 --infer_z_reversal -o outputs/20231007101619_C1
```
_20231007101619_C2_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231007101619_C2 --infer_z_reversal -o outputs/20231007101619_C2
```
_20231007101619_C3_ predictions (run 6):
```bash
python3 -m vesuvius_challenge_rnd.scroll_ink_detection.evaluation.inference /workspace/lightning_logs/<Add v_num from run 6 here>/checkpoints/<ADD CHECKPOINT NAME HERE>.ckpt 20231007101619_C3 --infer_z_reversal -o outputs/20231007101619_C3
```
