# ResNet Baseline Design

## Goal

Add a classical convolutional baseline for the current metasurface-inspired ISRJ parameter-estimation task, using the same IQ input and the same evaluation protocol as the main Transformer-based model.

## Chosen Design

The baseline will use a `1D ResNet + 3D regression head` architecture.

- Input stays unchanged: normalized IQ with shape `2 x N`
- Backbone is a 1D residual CNN
- Output is a 3-dimensional regression vector corresponding to `T_l`, `T_s`, and `x`
- Training uses parameter regression loss only
- Gate reconstruction is not included in this baseline

## Why This Design

This design keeps the comparison fair.

- It uses the same raw IQ input as the current main model
- It avoids introducing a second change in data representation, such as spectrogram conversion
- It keeps the baseline structurally simple and easy to explain in the paper
- It isolates the effect of the Transformer-based backbone and the gate-reconstruction constraint

## Rejected Alternatives

### 1. 2D ResNet on time-frequency images

Rejected because it changes both the backbone and the input representation, which weakens the fairness of the comparison.

### 2. ResNet with gate reconstruction

Rejected because it is no longer a clean baseline. If the baseline also uses gate reconstruction, the comparison between the proposed method and the baseline becomes less interpretable.

### 3. ResNet with a regression head plus auxiliary gate head

Rejected because it increases implementation and analysis complexity without being necessary for the first baseline version.

## Model Behavior

The baseline should behave as a direct regressor.

- It receives IQ input
- It extracts local and medium-range temporal features with residual 1D convolutions
- It aggregates features globally
- It predicts the three physical parameters directly

The baseline should not reuse the current gate decoder, gate bins, or waveform reconstruction path.

## Integration Plan

The implementation should follow existing repository structure and training flow.

- Add a new model class in `src/model.py`
- Extend model construction so the training script can select the new architecture by config
- Reuse the current dataset, scaler, decoding, evaluation, and reporting pipeline where possible
- Keep training loss limited to parameter regression for this architecture

## Training and Evaluation

The baseline should use the same dataset split and the same reporting protocol as the main model.

- Same train / validation / test data
- Same metrics: MAE and measurement accuracy
- Same JNR-based analysis
- Same tolerance thresholds

This allows direct insertion of the ResNet baseline into the paper as a classical CNN comparison.

## Risks

### 1. Loss path coupling

The current training code includes gate-reconstruction-related logic. The ResNet baseline must bypass those branches cleanly.

### 2. Decode interface consistency

The baseline output must match the existing downstream decoding and evaluation pipeline closely enough to avoid branching everywhere else in the codebase.

### 3. Comparison wording in the paper

The paper should describe this model as a `ResNet-based regression baseline`, not as a variant of the proposed method.

## Expected Outcome

After implementation, the repository should support training and evaluating a ResNet baseline with the same data and metric pipeline as the current Transformer-based model, making it suitable for both quantitative comparison and paper reporting.
