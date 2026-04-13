# ResNet Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `1D ResNet + 3D regression head` baseline that uses the same IQ input and evaluation pipeline as the current Transformer-based model.

**Architecture:** The baseline keeps the existing dataset and metric flow, replaces the Transformer backbone with a compact 1D ResNet encoder, and predicts `T_l`, `T_s`, and `x` directly through a regression head. Gate reconstruction is excluded so the baseline stays a clean convolutional comparison.

**Tech Stack:** Python, PyTorch, existing training/evaluation pipeline in `src/train.py`, unit tests with `unittest`

---

### Task 1: Add failing tests for ResNet model construction

**Files:**
- Modify: `/Users/savewind/Documents/gfkd/Metamaterials/tests/test_tokenizer_variants.py`
- Modify: `/Users/savewind/Documents/gfkd/Metamaterials/src/model.py`
- Modify: `/Users/savewind/Documents/gfkd/Metamaterials/src/config.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_model_supports_resnet_regression_architecture(self) -> None:
    config = ModelConfig(
        architecture="resnet_regression",
        input_channels=2,
        hidden_channels=128,
    )

    model = build_model(config)

    self.assertIsNotNone(model)
    dummy = torch.randn(2, 2, 4000)
    predictions = model(dummy)
    self.assertEqual(tuple(predictions.shape), (2, 3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_tokenizer_variants -v`

Expected: FAIL because `build_model()` does not yet support `resnet_regression`

- [ ] **Step 3: Write minimal implementation**

Add:
- `resnet_regression` architecture support to model construction
- a simple 1D ResNet regression model that returns `[B, 3]`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_tokenizer_variants -v`

Expected: PASS


### Task 2: Add failing tests for checkpoint training compatibility

**Files:**
- Create: `/Users/savewind/Documents/gfkd/Metamaterials/tests/test_resnet_regression_baseline.py`
- Modify: `/Users/savewind/Documents/gfkd/Metamaterials/src/train.py`
- Modify: `/Users/savewind/Documents/gfkd/Metamaterials/src/losses.py`

- [ ] **Step 1: Write the failing test**

```python
def test_resnet_regression_training_uses_parameter_only_loss(self) -> None:
    training = TrainingConfig()
    training.model.architecture = "resnet_regression"
    stage = _build_training_stages(training)[0]["name"]
    self.assertEqual(stage, "joint")
```

And:

```python
def test_resnet_regression_predictions_decode_to_three_parameters(self) -> None:
    raw = torch.rand(4, 3)
    scaler = ParameterScaler()
    decoded = scaler.denormalize_tensor(raw)
    self.assertEqual(tuple(decoded.shape), (4, 3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_resnet_regression_baseline -v`

Expected: FAIL because the baseline-specific path is not yet wired

- [ ] **Step 3: Write minimal implementation**

Update training/loss handling so:
- `resnet_regression` can train through the existing loop
- the model output is treated as direct normalized parameter predictions
- gate-reconstruction-specific branches are bypassed cleanly

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_resnet_regression_baseline -v`

Expected: PASS


### Task 3: Add config and evaluation support

**Files:**
- Modify: `/Users/savewind/Documents/gfkd/Metamaterials/src/config.py`
- Modify: `/Users/savewind/Documents/gfkd/Metamaterials/src/train.py`
- Modify: `/Users/savewind/Documents/gfkd/Metamaterials/src/infer.py`

- [ ] **Step 1: Add the new config path**

Ensure model selection supports:

```python
ModelConfig(architecture="resnet_regression")
```

- [ ] **Step 2: Wire evaluation through existing reporting**

Ensure the baseline reaches the same downstream metrics:
- MAE
- per-parameter measurement accuracy
- total measurement accuracy

- [ ] **Step 3: Run focused regression tests**

Run: `python3 -m unittest tests.test_tokenizer_variants tests.test_resnet_regression_baseline -v`

Expected: PASS


### Task 4: Verify end-to-end training entry point

**Files:**
- Modify: `/Users/savewind/Documents/gfkd/Metamaterials/src/train.py`
- Modify: `/Users/savewind/Documents/gfkd/Metamaterials/src/model.py`

- [ ] **Step 1: Run a smoke build of the Python modules**

Run: `python3 -m py_compile src/config.py src/model.py src/losses.py src/train.py src/infer.py tests/test_tokenizer_variants.py tests/test_resnet_regression_baseline.py`

Expected: PASS with no syntax errors

- [ ] **Step 2: Run the relevant unit tests**

Run: `python3 -m unittest tests.test_tokenizer_variants tests.test_resnet_regression_baseline tests.test_input_scale_toggle -v`

Expected: PASS

- [ ] **Step 3: Do a minimal training smoke test if the dataset exists on server**

Run:

```bash
python3 - <<'PY'
from config import TrainingConfig
from train import train_model

cfg = TrainingConfig()
cfg.model.architecture = "resnet_regression"
cfg.end_to_end_epochs = 1
cfg.batch_size = 4
train_model(cfg)
print("smoke-ok")
PY
```

Expected: either `smoke-ok` if the dataset is present, or a clear missing-dataset error that confirms the code path reaches training setup successfully
