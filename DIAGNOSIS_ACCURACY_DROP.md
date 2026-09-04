# 🔍 Diagnosis: Accuracy Decrease in Improved Training

## Problem
- **Baseline test accuracy (50 epochs):** 98.61%
- **Improved test accuracy (50 epochs):** 35.11%
- **Difference:** -63.5% ❌

Training gets stuck around loss=1.28-1.35 and can't improve past 35% test accuracy.

## Root Cause Analysis

### Key Difference 1: Prototype Handling
**Baseline (Working):**
- Prototypes are LEARNED as nn.Parameter
- Trained jointly with the network
- Prototypes are optimized via gradient descent

**Improved (Broken):**
- Prototypes are RECALCULATED each epoch from full training set
- No gradient updates to prototypes
- Static prototypes limit expressiveness

### Key Difference 2: Loss Function
**Baseline:**
- `nn.CrossEntropyLoss()` - No label smoothing
- Simple, stable training

**Improved:**
- `nn.CrossEntropyLoss(label_smoothing=0.1)` - Adds regularization
- Makes model less confident, can hurt optimization

### Key Difference 3: Learning Rate Schedule
**Baseline:**
- `CosineAnnealingLR(T_max=epochs)` - Simple cosine decay from start
- Direct, stable scheduling

**Improved:**
- Custom `LambdaLR` with warmup + cosine
- More complex, harder to tune
- Warmup might start with wrong learning rate

### Key Difference 4: Augmentation
**Baseline:**
- Simple random dropout augmentation (15% of values set to 0)

**Improved:**
- 4-way augmentation: time shift, channel dropout, Gaussian noise, spike dropout
- More aggressive, can hurt training stability

### Key Difference 5: Prototype Recalculation
**Baseline:**
- Prototypes trained as parameters (updated via backprop)
- Prototypes adapt to network evolution

**Improved:**
- Prototypes recalculated each epoch from ALL training data
- Prototypes don't adapt to network, stay constant-like
- Creates moving target problem for the network

## Solution: Use Learned Prototypes Like Baseline

The improved script should use the same prototype learning strategy as baseline, but with:
1. 150 timesteps (vs 25)
2. 3-layer DeepSNN (vs 2-layer baseline)
3. Optional: Keep advanced augmentations (but reduced intensity)

### Changes Needed:
1. **Use nn.Parameter for prototypes** - Let them be learned
2. **Remove custom warmup** - Use simple CosineAnnealingLR
3. **Remove label smoothing** - Use plain CrossEntropyLoss
4. **Reduce augmentation** - Use gentler augmentations
5. **Let network and prototypes co-adapt** - Like baseline does

