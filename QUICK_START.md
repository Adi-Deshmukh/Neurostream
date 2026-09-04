# Quick Reference: Running the Improved NeuroStream Training

## Setup (First Time Only)

```bash
cd /Users/yashas/Documents/Neurostream
source FYP/bin/activate
pip install -e . -q
```

---

## 🚀 Quick Start Commands

### Test the Improvements (5 minutes)
```bash
python scripts/test_tier1_improvements.py
```

### Train Subject 1 - Quick Test (30 min)
```bash
python scripts/run_phase_a_improved.py \
    --subject 1 \
    --epochs 10 \
    --n-timesteps 100 \
    --batch-size 32
```

### Train Subject 1 - Full Improved (3 hours on CPU)
```bash
python scripts/run_phase_a_improved.py \
    --subject 1 \
    --epochs 100 \
    --n-timesteps 150 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --temperature 0.1 \
    --warmup-epochs 5 \
    --label-smoothing 0.1 \
    --weight-decay 1e-4 \
    --checkpoint-dir results/checkpoints
```

### Compare with Baseline
```bash
# Original (25 timesteps)
python scripts/run_phase_a.py --subject 1 --epochs 50

# Improved (150 timesteps)  
python scripts/run_phase_a_improved.py --subject 1 --epochs 100
```

### Train All Subjects (9 subjects, sequential)
```bash
for subject in {1..9}; do
    echo "Training subject $subject..."
    python scripts/run_phase_a_improved.py \
        --subject $subject \
        --epochs 100 \
        --n-timesteps 150 \
        --checkpoint-dir results/checkpoints
done
```

---

## 📊 Expected Output

After training, you should see:
```
INFO:neurostream.training.phase_a_train:Phase A subject 1: train_accuracy=62.3% test_accuracy=48.2% validation_accuracy=51.6%;
Subject 1: train=62.3%, val=51.6%, test=48.2%
Checkpoint: results/checkpoints/phase_a_subject_1_improved.pt
Prototypes: results/checkpoints/prototypes_subject_1_improved.pt
```

**Expected improvements** (rough estimates):
- Baseline: ~37%
- After Tier 1: ~48-52%
- After Tier 2: ~55-65%

---

## 🔧 Customization

### Use Smaller Dataset for Testing
```bash
python scripts/run_phase_a_improved.py \
    --subject 1 \
    --epochs 5 \
    --n-timesteps 50 \
    --batch-size 16
```

### Use Deeper Architecture  
```bash
# DeepSNNFeatureExtractor (3 layers) is default in improved training
# To use original 2-layer SNN, modify run_phase_a_improved.py line ~115
```

### Adjust Augmentation Strength
In `run_phase_a_improved.py`, modify in `train_phase_a_improved()`:
```python
# More aggressive augmentation
time_shift_prob=0.5,           # was 0.3
max_shift=20,                  # was 10
channel_dropout_prob=0.25,     # was 0.15
gaussian_noise_sigma=0.1,      # was 0.05
spike_dropout_prob=0.2,        # was 0.15
mixup_alpha=0.3,               # was 0.2
```

### Adjust Learning Rate Schedule
```bash
python scripts/run_phase_a_improved.py \
    --subject 1 \
    --learning-rate 5e-4 \        # Lower LR
    --warmup-epochs 10 \          # Longer warmup
    --epochs 150                  # More epochs
```

---

## 📈 Monitoring Training

### Watch Training Progress
```bash
# In separate terminal, monitor checkpoint files
watch -n 5 'ls -lh results/checkpoints/ | tail -5'
```

### Check Logged Metrics
```bash
# Training logs are printed to stdout
# Look for:
# - "Phase A epoch X/Y loss=... val_acc=..."
# - Early stopping messages
# - Final accuracy metrics
```

---

## 🧪 Phase B Adaptation

### Test Improved Phase B
```python
from neurostream.training.phase_b_adapt_improved import adapt_target_session_improved
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
import torch

# Load trained model and prototypes
model = SNNFeatureExtractor(in_features=66)
state = torch.load('results/checkpoints/phase_a_subject_1_improved.pt')
model.load_state_dict(state['model_state_dict'])

prototypes = torch.load('results/checkpoints/prototypes_subject_1_improved.pt')

# Adapt on target session
result, adjustments = adapt_target_session_improved(
    model, prototypes, target_spikes,
    adaptive_momentum=True,
    adaptive_threshold=True,
)

print(f"Before: {result.before_accuracy:.1%} → After: {result.after_accuracy:.1%}")
print(f"Parameters: momentum={adjustments.momentum:.3f}, "
      f"threshold={adjustments.confidence_threshold:.2f}")
```

---

## 🐛 Troubleshooting

### Issue: MOABB Dataset Download Hangs
```bash
# Solution: Pre-download dataset
python -c "from moabb.datasets import BNCI2014001; BNCI2014001().get_data([1])"
```

### Issue: Memory Error
```bash
# Solution: Reduce batch size and n_timesteps
python scripts/run_phase_a_improved.py \
    --subject 1 \
    --epochs 50 \
    --n-timesteps 100 \
    --batch-size 16
```

### Issue: Very Slow Training (CPU-bound)
```bash
# Check if GPU available
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# If GPU available but not used, set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
python scripts/run_phase_a_improved.py --subject 1 --epochs 100
```

### Issue: Training Diverges (NaN loss)
```bash
# Reduce learning rate
python scripts/run_phase_a_improved.py \
    --subject 1 \
    --learning-rate 5e-4 \
    --epochs 100
```

---

## 📊 Results Comparison

### Baseline (Original)
```
script: run_phase_a.py
params: subject=1, epochs=50, n_timesteps=25
expected accuracy: ~37%
time: ~30 min on CPU
```

### Tier 1 (Improved)
```
script: run_phase_a_improved.py
params: subject=1, epochs=100, n_timesteps=150
expected accuracy: ~48-52%
time: ~2-3 hours on CPU
improvement: +15% absolute accuracy
```

### Tier 1+2 (Deep SNN)
```
script: run_phase_a_improved.py
params: subject=1, epochs=100, n_timesteps=150, DeepSNNFeatureExtractor
expected accuracy: ~55-65%
time: ~3-4 hours on CPU
improvement: +25% absolute accuracy
```

---

## 💾 File Locations

### Checkpoints
- Original: `results/checkpoints/phase_a_subject_1.pt`
- Improved: `results/checkpoints/phase_a_subject_1_improved.pt`

### Prototypes
- Original: `results/checkpoints/prototypes_subject_1.pt`
- Improved: `results/checkpoints/prototypes_subject_1_improved.pt`

### Logs
```bash
# Training logs printed to stdout
# To save to file:
python scripts/run_phase_a_improved.py --subject 1 | tee training_logs.txt
```

---

## 🎯 Next Steps

1. **Verify setup**:
   ```bash
   python scripts/test_tier1_improvements.py
   ```

2. **Train Subject 1**:
   ```bash
   python scripts/run_phase_a_improved.py --subject 1 --epochs 100
   ```

3. **Evaluate results**:
   ```bash
   python -c "import torch; 
   ckpt = torch.load('results/checkpoints/phase_a_subject_1_improved.pt')
   print(f\"Test accuracy: {ckpt['test_accuracy']:.1%}\")"
   ```

4. **Run multi-subject training** (if satisfied with improvements):
   ```bash
   for s in {1..9}; do
       python scripts/run_phase_a_improved.py --subject $s --epochs 100
   done
   ```

---

## 📚 Documentation Files

- `TIER1_IMPROVEMENTS.md` - Detailed technical guide
- `IMPLEMENTATION_SUMMARY.md` - Complete implementation overview
- `README.md` - Original project README
- This file - Quick reference commands

