# Technical reading of the submitted thesis

## Research question

Can the core spatial and temporal ideas behind MetNet-2 be reduced to a compact
model that learns meaningful short-term precipitation dynamics from RADOLAN RW
radar data on standard academic hardware?

## Method

The study builds a lazy-loading RADOLAN data pipeline, sorts radar composites in
time, converts them to normalized rainfall tensors, and constructs sliding
sequences. Six consecutive five-minute frames form the input and eight
subsequent frames form the target.

The submitted MiniMetNet2 architecture uses convolutional feature extraction,
dilated residual blocks, and a direct multi-output head. It omits the full
attention and multi-resolution machinery of MetNet-2 to reduce compute and
memory requirements. Training uses mean squared error on 1,000 sequences; a
separate 1,000-sequence subset supports evaluation.

## Main findings

- Broad precipitation structures and dominant motion are retained most
  consistently in stratiform and moderate-rain cases.
- Small convective cells, sharp boundaries, and peak intensities are smoothed,
  particularly at longer lead times.
- MSE and MAE increase monotonically over the eight forecast steps.
- Probability of detection and critical success index decrease with lead time;
  false alarm ratio remains comparatively low.
- Dry and low-rain scenes do not show widespread artificial rain generation.

## Interpretation

The work is best read as an accessible, resource-constrained study rather than
a full reproduction of operational MetNet-2. Its contribution is the complete
experimental chain—radar decoding, sequence construction, model design,
training, visual evaluation, and metric-based interpretation—under limited
hardware and data.

## Limitations and future work

The thesis identifies limited dataset diversity, CPU-only training, a compact
receptive field, the absence of attention, and MSE-driven smoothing as the main
constraints. Proposed extensions include more diverse convective samples,
attention mechanisms, satellite or other meteorological inputs, structure-aware
losses, and GPU-based training.
