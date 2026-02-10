# Research Plan: RL-Enhanced Object Removal via Preference Optimization

## Overview

This research builds upon RORem (Robust Object Remover) to explore reinforcement learning and preference optimization techniques for improving diffusion-based object removal quality. We focus on two main directions with a third as future work.

---

## Research Directions

| Direction | Method | Priority | Status |
|-----------|--------|----------|--------|
| **Direction 2** | Direct Preference Optimization (DPO) | High | Planned |
| **Direction 3** | Reward-Weighted Distillation | High | Planned |
| **Direction 4** | Online RL Fine-tuning | Future | Deferred |

---

## Direction 2: Direct Preference Optimization (DPO) for Object Removal

### Motivation

DPO eliminates the need for explicit reward model training by directly optimizing the policy using preference data. This is particularly appealing because:
- RORem already has human-annotated binary preference data (score: 0/1)
- Avoids reward hacking issues common in RLHF
- More stable training than policy gradient methods for diffusion models
- Recent success with Diffusion-DPO shows viability for image generation

### Technical Approach

**Standard DPO Loss:**
```
L_DPO = -E[log σ(β(log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))]
```

Where:
- `y_w`: preferred (winning) sample (score=1)
- `y_l`: non-preferred (losing) sample (score=0)
- `π_θ`: current policy (fine-tuned model)
- `π_ref`: reference policy (frozen pretrained model)
- `β`: temperature parameter controlling deviation from reference

**Adaptation for Diffusion Models (Diffusion-DPO):**

For diffusion models, we need to handle the sequential denoising process. Following Wallace et al. (2023), we use:

```
L_DPO-Diffusion = -E_t[log σ(β(
    ||ε - ε_θ(x_t^w, t)||² - ||ε - ε_θ(x_t^l, t)||²
    - ||ε - ε_ref(x_t^w, t)||² + ||ε - ε_ref(x_t^l, t)||²
))]
```

### Data Requirements

**Current RORem Data Format:**
```json
{"source": "path/to/source.png", "mask": "path/to/mask.png", "GT": "path/to/output.png", "score": 1}
```

**Required Preference Pair Format:**
```json
{
  "source": "path/to/source.png",
  "mask": "path/to/mask.png",
  "preferred": "path/to/good_output.png",
  "rejected": "path/to/bad_output.png"
}
```

**Data Construction Strategy:**
1. For each source-mask pair with multiple outputs at different quality levels
2. Pair score=1 samples with score=0 samples from same/similar inputs
3. If not available: generate multiple samples from current model, use discriminator to rank

### Implementation Plan

#### Phase 2.1: Data Preparation
- [ ] Analyze existing dataset structure for preference pairs
- [ ] Create `meta_to_preference_dataset_format()` function in `myutils/img_util.py`
- [ ] Build preference pair dataset from existing annotations
- [ ] If insufficient pairs: generate synthetic pairs using model + discriminator

#### Phase 2.2: DPO Training Script
- [ ] Create `train_RORem_dpo.py` based on `train_RORem.py`
- [ ] Implement DPO loss function for diffusion models
- [ ] Add reference model (frozen copy of pretrained model)
- [ ] Implement proper logging for preference optimization metrics

#### Phase 2.3: Training Infrastructure
- [ ] Modify data loader to handle preference pairs
- [ ] Implement β (beta) annealing schedule
- [ ] Add validation metrics: win rate, preference accuracy

#### Phase 2.4: Experiments
- [ ] Baseline: RORem without DPO
- [ ] DPO with different β values (0.1, 0.5, 1.0, 2.0)
- [ ] DPO with LoRA vs full fine-tuning
- [ ] Compare with discriminator-filtered training

### Key Files to Create/Modify

```
RORem/
├── train_RORem_dpo.py              # NEW: DPO training script
├── myutils/
│   ├── img_util.py                 # MODIFY: add preference data loader
│   └── dpo_utils.py                # NEW: DPO loss and utilities
├── configs/
│   └── dpo_config.yaml             # NEW: DPO hyperparameters
└── scripts/
    └── run_train_dpo.sh            # NEW: training launcher
```

### Expected Challenges

1. **Preference Pair Construction**: May need to generate synthetic pairs if not enough natural pairs exist
2. **Computational Cost**: Need to maintain reference model in memory
3. **β Tuning**: Too low = no learning; too high = mode collapse
4. **Distribution Shift**: DPO assumes on-policy samples; may need iterative retraining

---

## Direction 3: Reward-Weighted Distillation

### Motivation

Combine the speed benefits of LCM distillation (4 steps instead of 50) with quality improvements from reward signals. The key insight: during distillation, weight the loss by discriminator scores to emphasize learning from high-quality teacher outputs.

### Technical Approach

**Standard LCM Distillation Loss:**
```
L_LCM = E_t[||f_θ(z_t, t) - f_teacher(z_t', t')||²]
```

**Reward-Weighted Distillation Loss:**
```
L_RWD = E_t[w(x) * ||f_θ(z_t, t) - f_teacher(z_t', t')||²]

where w(x) = softmax(D_φ(x) / τ)  # temperature-scaled discriminator score
```

**Alternative: Reward-Filtered Distillation**
```
Only use samples where D_φ(x) > threshold for distillation
```

### Implementation Plan

#### Phase 3.1: Discriminator Integration
- [ ] Load pretrained RORem-Discriminator
- [ ] Create reward scoring function that takes generated samples
- [ ] Implement sample filtering/weighting logic

#### Phase 3.2: Modified LCM Training
- [ ] Create `train_RORem_lcm_rewarded.py` based on `train_RORem_lcm.py`
- [ ] Add discriminator scoring during training
- [ ] Implement weighted loss computation
- [ ] Add reward statistics logging

#### Phase 3.3: Weighting Strategies
- [ ] **Hard filtering**: Only use samples above threshold
- [ ] **Soft weighting**: Weight all samples by discriminator score
- [ ] **Temperature scaling**: Control sharpness of weighting
- [ ] **Curriculum**: Start with all samples, gradually increase filtering

#### Phase 3.4: Experiments
- [ ] Baseline: Standard LCM distillation (existing)
- [ ] Hard filtering with thresholds (0.3, 0.5, 0.7)
- [ ] Soft weighting with temperatures (0.1, 0.5, 1.0)
- [ ] Compare 4-step quality vs original 50-step

### Key Modifications to `train_RORem_lcm.py`

```python
# Current code (line 1302-1307):
noise_pred = unet(
    concatenated_noisy_latents,
    start_timesteps,
    encoder_hidden_states=encoder_hidden_states,
    added_cond_kwargs=added_cond_kwargs,
).sample

# Add reward weighting after generating predictions:
with torch.no_grad():
    # Decode latents to images for discriminator
    decoded_images = vae.decode(model_pred / vae.config.scaling_factor).sample
    # Get reward scores from discriminator
    reward_scores = discriminator(decoded_images, mask, source)
    # Compute weights
    weights = F.softmax(reward_scores / temperature, dim=0)

# Weighted loss (line 1424):
if args.loss_type == "l2":
    loss = (weights * F.mse_loss(model_pred.float(), target.float(), reduction="none")).mean()
```

### Key Files to Create/Modify

```
RORem/
├── train_RORem_lcm_rewarded.py     # NEW: Reward-weighted LCM training
├── myutils/
│   └── reward_utils.py             # NEW: Discriminator reward utilities
├── configs/
│   └── lcm_rewarded_config.yaml    # NEW: Hyperparameters
└── scripts/
    └── run_train_lcm_rewarded.sh   # NEW: training launcher
```

### Expected Challenges

1. **Computational Overhead**: Need to decode latents and run discriminator during training
2. **Discriminator Calibration**: Scores may not be well-calibrated for weighting
3. **Sample Efficiency**: Hard filtering reduces effective dataset size
4. **Mode Collapse**: Heavy filtering might reduce diversity

### Potential Solutions

1. **Efficient Scoring**: Score only every N steps, or use cached scores
2. **Score Normalization**: Use percentile-based weighting instead of raw scores
3. **Diversity Regularization**: Add entropy bonus or diversity loss
4. **Progressive Filtering**: Start permissive, increase threshold over training

---

## Direction 4: Online RL Fine-tuning (Future Work)

### Overview

Full online RL where the model generates samples, receives rewards from discriminator, and updates via policy gradient methods (REINFORCE, PPO, etc.).

### Why Defer?

1. **Higher Complexity**: Requires careful handling of exploration, credit assignment
2. **Computational Cost**: Online generation during training is expensive
3. **Stability**: Policy gradient methods can be unstable for diffusion models
4. **Build on Earlier Work**: DPO and reward-weighted distillation provide foundations

### Future Implementation Outline

```python
# Simplified online RL loop
for iteration in range(num_iterations):
    # 1. Generate samples with current model
    with torch.no_grad():
        samples = model.generate(source, mask, num_samples=batch_size)

    # 2. Score with discriminator
    rewards = discriminator(samples, mask, source)

    # 3. Compute policy gradient
    log_probs = model.log_prob(samples, source, mask)
    loss = -torch.mean(rewards * log_probs)  # REINFORCE

    # 4. Update model
    loss.backward()
    optimizer.step()
```

### Key Research Questions for Future

1. How to handle the sequential nature of diffusion for credit assignment?
2. What exploration strategies work for inpainting tasks?
3. Can we use the discriminator directly, or need value function?
4. How to prevent reward hacking?

---

## Experimental Setup

### Evaluation Metrics

| Metric | Description | Tool |
|--------|-------------|------|
| **Discriminator Score** | Learned quality metric | RORem-Discriminator |
| **Human Evaluation** | Success rate from annotators | Manual |
| **FID** | Distribution quality | pytorch-fid |
| **LPIPS** | Perceptual similarity | lpips library |
| **CLIP Score** | Semantic consistency | CLIP |
| **Inference Speed** | Steps and wall-clock time | Custom |

### Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| RORem Training | 200K+ triplets | Main training |
| RORem Test | 811MB | Evaluation |
| Mulan | 3.26GB | Additional training |
| Final HR | 7.9GB | High-resolution training |

### Compute Requirements

| Experiment | GPUs | Time (est.) |
|------------|------|-------------|
| DPO Training | 8x A100 | 24-48 hours |
| Reward-Weighted LCM | 8x A100 | 12-24 hours |
| Full Online RL | 8x A100 | 48-72 hours |

---

## Timeline and Milestones

### Phase 1: Foundation (Current)
- [x] Understand RORem codebase
- [x] Identify research directions
- [ ] Set up development environment
- [ ] Verify baseline training works

### Phase 2: Direction 2 - DPO
- [ ] Data preparation for preference pairs
- [ ] Implement DPO training script
- [ ] Run initial experiments
- [ ] Tune hyperparameters
- [ ] Document results

### Phase 3: Direction 3 - Reward-Weighted Distillation
- [ ] Integrate discriminator into LCM training
- [ ] Implement weighting strategies
- [ ] Run distillation experiments
- [ ] Compare with standard LCM
- [ ] Document results

### Phase 4: Analysis and Writing
- [ ] Ablation studies
- [ ] Comparison across methods
- [ ] Prepare visualizations
- [ ] Write paper/report

### Phase 5: Future Work (Direction 4)
- [ ] Design online RL framework
- [ ] Implement and experiment
- [ ] Extend to other applications

---

## References

### Key Papers to Read

1. **DPO**: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
2. **Diffusion-DPO**: Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization" (2023)
3. **LCM**: Luo et al., "Latent Consistency Models" (2023)
4. **DDPO**: Black et al., "Training Diffusion Models with Reinforcement Learning" (2023)
5. **DRaFT**: Clark et al., "Directly Fine-Tuning Diffusion Models on Differentiable Rewards" (2023)
6. **RORem**: Li et al., "Training a Robust Object Remover with Human-in-the-Loop" (2025)

### Relevant Codebases

- [diffusers](https://github.com/huggingface/diffusers) - Base diffusion library
- [trl](https://github.com/huggingface/trl) - Transformer RL, includes DPO
- [diffusion-dpo](https://github.com/huggingface/diffusion-dpo) - Reference implementation

---

## Notes

- Start with Direction 3 (reward-weighted distillation) as it requires fewer modifications
- Direction 2 (DPO) requires preference pair construction which may need additional data work
- Consider combining both: DPO for base model, then reward-weighted distillation for speed
- Keep detailed experiment logs in wandb for reproducibility
