# Methods Summary

## Preprocessing

1. load image stack as raw uint16
2. rescale and convert to uint8
3. (optional) flip the depth dimension based on the segment
4. (optional) clip intensities

## Labels

We used two types of masks: ink labels and non-ink labels. The former corresponds to papyrus with ink on the 
*target sheet* (the sheet centered around slice 32) and the latter to papyrus without ink on the target sheet.

## Dataset generation

### Patch sampling

We generate patches (of a fixed size) using a sliding window approach, discarding any patch having a label containing 
less than some pre-specified fraction of unlabeled pixels. In practice this was set to near 100%, thus requiring almost 
every patch to be labeled. In our case, the labeled area is the union of the ink labels and non-ink labels. The 
unlabeled area is its complement. We also discard any patch that overlaps with the non-papyrus region of the segment mask.

### Dataset split

We use leave-k-segments-out cross-validation. While we didn't find any signs of significant overfitting, we tried to minimize training on segments having overlap with the 
validation segment where obvious.

### Sub-segments

We sometimes split the multi-column segments into sub-segments containing a single column of text. These sub-segments 
are created by slicing along a particular axis and saving the output sub-segment mask and TIFFs.

## Model

We use a 3D-to-2D encoder-decoder model based on a 3D U-net encoder and a 2D Segformer decoder, similar to that used by 
the winning team of the Vesuvius fragment ink detection Kaggle competition.

- Encoder: Residual 3D U-Net with the addition of concurrent spatial and channel squeeze & excitation blocks
- Decoder: Segformer-b1

The PyTorch Lightning model we used is called `UNet3dSegformerPLModel` and the underlying PyTorch model is called 
`UNet3DSegformer`. We used a feature size of 16 for the 3D U-Net, a depth of 5, and 32 output channels for 
this model. These 3D features are max-pooled along the depth dimension. The Segformer is pretrained from the 
*nvidia/mit-b1* checkpoint. Finally, the outputs are upscaled to the input resolution.

## Training

Here’s an overview of our training setup:

- Optimizer: AdamW
- Loss function: dice + softBCE
- Learning rate scheduler: gradual warmup
- Performance measure: We select the model checkpoint that maximizes the mean of the binary average precision and the $F_{0.5}$ score in the validation set.
- XY patch size: 64 x 64
- Z slices: 15 - 47
- DDP training with 1 node and 8 devices
- Batch size: 32
- FP16 mixed precision training

## Data augmentation

- horizontal and vertical flips, random brightness contrast, shift-scale-rotate, gaussian noise, gaussian blur, motion blur, coarse dropout, and cut-and-paste slices

## Inference

We used sliding window inference with the following settings:

- Stride: 8
- XY window size: 64 x 64
- Z-min: 15
- Z-max: 47

We also reverse the depth dimension for some segments.

To reconstruct the segment prediction from the overlapping patch predictions, we weight the patches using a Hann window.
