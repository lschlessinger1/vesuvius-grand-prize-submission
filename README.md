# vesuvius-grand-prize-submission
Vesuvius challenge grand prize submission

## About

We approached the ink detection task as a 3D-to-2D binary semantic segmentation problem using surface volumes from 
scroll 1 (PHerc Paris 3). We followed a human-assisted pseudo-label-based self-training approach using the crackle signal as a surrogate
to the ink signal.

For a summary of the methods used, please see [docs/methods.md](docs/methods.md).

## Getting started

For instructions on how to train and run inference, please see [docs/submission_reproduction_instructions.md](docs/submission_reproduction_instructions.md).

A pretrained checkpoint is available [here](https://drive.google.com/file/d/1bY14CjSfY8VbqlKmjv1MW-bzhScLZOoV/view?usp=sharing) 
(associated with [val_3336_C3.yaml](vesuvius_challenge_rnd/scroll_ink_detection/experiment_runner/configs/unet3d_segformer/submission/val_3336_C3.yaml)).

## Authors
Louis Schlessinger, Arefeh Sherafati

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Credits
- [EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT](https://arxiv.org/abs/2304.02084)
- [Introducing Hann windows for reducing edge-effects in patch-based image segmentation](https://arxiv.org/abs/1910.07831)
- [1st place Kaggle Vesuvius Challenge - Ink Detection](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417496)
- [4th place Kaggle Vesuvius Challenge - Ink Detection](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417779)
- [First Ink Vesuvius Challenge](https://caseyhandmer.wordpress.com/2023/08/05/reading-ancient-scrolls/)
- [2nd place Vesuvius Challenge First Letters](https://github.com/younader/Vesuvius-First-Letters)