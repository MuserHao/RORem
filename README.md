# RL-Enhanced Object Removal

Research project exploring reinforcement learning and preference optimization techniques for diffusion-based object removal, building upon [RORem](https://arxiv.org/abs/2501.00740).

## Research Focus

| Direction | Method | Description |
|-----------|--------|-------------|
| **DPO** | Direct Preference Optimization | Optimize policy directly from preference pairs without explicit reward model |
| **RWD** | Reward-Weighted Distillation | Weight LCM distillation loss by discriminator quality scores |
| **Online RL** | Policy Gradient Methods | Generate samples, score with discriminator, update via REINFORCE/PPO (future) |

See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for detailed implementation plans.

---

## Overview Framework

![pipeline](figures/pipeline.png)

The base RORem training pipeline:
- **Stage 1**: Train initial removal model on 60K triplets from open-source datasets
- **Stage 2**: Apply model to test set, human annotators select high-quality samples
- **Stage 3**: Train discriminator on human feedback, auto-annotate new samples
- **Iterate**: Repeat stages 2-3 to obtain 200K+ training triplets

Our research extends this with RL-based optimization in stages 2-3.

---

## Setup

```bash
git clone <repo-url>
cd RORem
conda env create -f environment.yaml
conda activate RORem
```

Install xformers (version must match torch):
```bash
pip install xformers==0.0.28.post3
```

Experiment tracking with wandb:
```bash
pip install wandb
wandb login
```

---

## Datasets

Available at [HuggingFace: LetsThink/RORem_dataset](https://huggingface.co/datasets/LetsThink/RORem_dataset)

| Dataset | Size | Download |
|---------|------|----------|
| RORem+RORD | 73.15GB | [Google Drive](https://drive.google.com/file/d/1sE6IOhHNCKiwFLW4a2ZWcwU4_bhvGcSA/view?usp=sharing) |
| Mulan | 3.26GB | [Google Drive](https://drive.google.com/file/d/1-dX5GfxyGEGBSfFeBgl5vMH9ODdCpbuq/view?usp=sharing) |
| Final HR | 7.9GB | [Google Drive](https://drive.google.com/file/d/1S3p_yLjPuhZbh7S15actNaAOEPvUlW5C/view?usp=sharing) |
| Test Set | 811MB | [HuggingFace](https://huggingface.co/datasets/LetsThink/RORem_dataset/resolve/main/testset_RORem.tar.gz) |

### Data Format

```
dataset/
├── source/          # Input images
├── mask/            # Object masks
├── GT/              # Ground truth (object removed)
└── meta.json        # Metadata
```

**Standard triplets:**
```json
{"source": "source/xxx.png", "mask": "mask/xxx.png", "GT": "GT/xxx.png"}
```

**With scores (for discriminator/RL):**
```json
{"source": "source/xxx.png", "mask": "mask/xxx.png", "GT": "GT/xxx.png", "score": 1}
```

**Preference pairs (for DPO):**
```json
{"source": "source/xxx.png", "mask": "mask/xxx.png", "preferred": "good.png", "rejected": "bad.png"}
```

---

## Pretrained Models

| Model | Description | Download |
|-------|-------------|----------|
| RORem | Base model (512x512) | [Google Drive](https://drive.google.com/drive/folders/1-ZOLMkifypR2SW0n4pOw6_0iIuHu2Ovy?usp=drive_link) |
| RORem-mixed | Mixed resolution (512-1024) | [Google Drive](https://drive.google.com/drive/folders/1G46Rs0-fZvoJ55OLQrC35dbRohFM917z?usp=drive_link) |
| RORem-LCM | 4-step LoRA | [Google Drive](https://drive.google.com/drive/folders/1QK8qcqT7SKRzD2AyGtgfwWwlQrUesAc1?usp=drive_link) |
| RORem-Discriminator | Reward model | [Google Drive](https://drive.google.com/drive/folders/1ka3tN_hEeP1QR2CU81Uf9QM1JBHdDvc2?usp=drive_link) |

**HuggingFace (recommended):**
```python
from diffusers import AutoPipelineForInpainting
import torch

pipe = AutoPipelineForInpainting.from_pretrained(
    "LetsThink/RORem",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
```

---

## Inference

### Standard Inference (50 steps)

```python
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from myutils.img_util import dilate_mask
import torch

pipe = AutoPipelineForInpainting.from_pretrained(
    "LetsThink/RORem",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

input_image = load_image("input.png").resize((512, 512))
mask_image = load_image("mask.png").resize((512, 512))

# Optional: dilate mask for better boundary handling
# mask_image = dilate_mask(mask_image, dilate_factor=20)

# Without CFG
result = pipe(
    prompt="",
    image=input_image,
    mask_image=mask_image,
    height=512, width=512,
    guidance_scale=1.0,
    num_inference_steps=50,
    strength=0.99
).images[0]

# With CFG (better quality)
result = pipe(
    prompt="4K, high quality, masterpiece, Highly detailed, Sharp focus, Professional, photorealistic, realistic",
    negative_prompt="low quality, worst, bad proportions, blurry, extra finger, Deformed, disfigured, unclear background",
    image=input_image,
    mask_image=mask_image,
    height=512, width=512,
    guidance_scale=1.0,
    num_inference_steps=50,
    strength=0.99
).images[0]
```

### Fast Inference (4 steps with LCM)

```bash
python inference_RORem_4S.py \
    --pretrained_model diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --RORem_unet path/to/RORem_unet \
    --RORem_LoRA path/to/RORem_LCM_lora \
    --image_path input.png \
    --mask_path mask.png \
    --inference_steps 4 \
    --save_path result.png
```

### Quality Scoring with Discriminator

```bash
python inference_RORem_discriminator.py \
    --pretrained_model diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --RORem_discriminator path/to/discriminator \
    --image_path source.png \
    --mask_path mask.png \
    --edited_path result.png
```

---

## Training

### Baseline: RORem (Supervised)

```bash
accelerate launch --multi_gpu --num_processes 8 train_RORem.py \
    --train_batch_size 16 \
    --output_dir experiment/RORem_baseline \
    --meta_path path/to/meta.json \
    --max_train_steps 50000 \
    --random_flip \
    --resolution 512 \
    --pretrained_model_name_or_path diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --mixed_precision fp16 \
    --checkpoints_total_limit 5 \
    --checkpointing_steps 5000 \
    --learning_rate 5e-5 \
    --validation_steps 2000 \
    --report_to wandb
```

**With DeepSpeed (lower memory):**
```bash
accelerate launch --config_file config/deepspeed_config.yaml \
    --multi_gpu --num_processes 8 train_RORem.py \
    [same arguments as above]
```

### LCM Distillation (4-step)

```bash
accelerate launch --multi_gpu --num_processes 8 train_RORem_lcm.py \
    --pretrained_teacher_unet path/to/RORem_unet \
    --output_dir experiment/RORem_LCM \
    --max_train_steps 20000
```

### Discriminator Training

Requires scored data:
```json
[
  {"source": "source/xxx.png", "mask": "mask/xxx.png", "GT": "GT/xxx.png", "score": 1},
  {"source": "source/yyy.png", "mask": "mask/yyy.png", "GT": "GT/yyy.png", "score": 0}
]
```

```bash
bash run_train_RORem_discriminator.sh
```

### Direction 2: DPO Training (Proposed)

```bash
accelerate launch --multi_gpu --num_processes 8 train_RORem_dpo.py \
    --pretrained_unet path/to/RORem_unet \
    --preference_data path/to/preferences.json \
    --beta 0.5 \
    --output_dir experiment/RORem_DPO \
    --max_train_steps 10000 \
    --report_to wandb
```

### Direction 3: Reward-Weighted Distillation (Proposed)

```bash
accelerate launch --multi_gpu --num_processes 8 train_RORem_lcm_rewarded.py \
    --pretrained_teacher_unet path/to/RORem_unet \
    --discriminator_path path/to/discriminator \
    --reward_temperature 0.5 \
    --reward_threshold 0.5 \
    --output_dir experiment/RORem_LCM_rewarded \
    --max_train_steps 20000 \
    --report_to wandb
```

---

## Project Structure

```
RORem/
├── train_RORem.py                  # Base model training (supervised)
├── train_RORem_lcm.py              # LCM distillation (4-step inference)
├── train_RORem_discriminator.py    # Discriminator/reward model training
├── train_RORem_dpo.py              # [PLANNED] DPO training
├── train_RORem_lcm_rewarded.py     # [PLANNED] Reward-weighted distillation
│
├── inference_RORem.py              # Standard inference (50 steps)
├── inference_RORem_4S.py           # Fast inference (4 steps)
├── inference_RORem_discriminator.py # Quality scoring
│
├── model/
│   └── unet_sdxl_discriminator.py  # UNet with classification head
├── pipelines/
│   ├── RORem_inpaint_pipeline.py   # Modified SDXL inpainting pipeline
│   └── RORem_discriminator_pipeline.py
├── myutils/
│   └── img_util.py                 # Data loading utilities
├── config/
│   └── deepspeed_config.yaml       # DeepSpeed Zero-2 config
├── validation_data/                # Sample validation images
├── figures/                        # Documentation figures
│
├── RESEARCH_PLAN.md                # Detailed implementation plans
└── README.md                       # This file
```

---

## Key Technical Details

### Discriminator Architecture

Extends UNet with classification head on mid-block features:

```python
# model/unet_sdxl_discriminator.py:393-405
self.cls_pred_branch = nn.Sequential(
    nn.Conv2d(1280, 1280, 4, 2, 1),  # 16x16 -> 8x8
    nn.GroupNorm(32, 1280), nn.SiLU(),
    nn.Conv2d(1280, 1280, 4, 2, 1),  # 8x8 -> 4x4
    nn.GroupNorm(32, 1280), nn.SiLU(),
    nn.Conv2d(1280, 1280, 4, 4, 0),  # 4x4 -> 1x1
    nn.GroupNorm(32, 1280), nn.SiLU(),
    nn.Conv2d(1280, 1, 1, 1, 0),     # scalar output
)
# Output: F.softplus(logits) -> continuous quality score
```

### DPO for Diffusion (Proposed)

```python
def dpo_diffusion_loss(model, ref_model, preferred, rejected, beta=0.5):
    log_p_w = -mse_loss(model(preferred), noise)
    log_p_l = -mse_loss(model(rejected), noise)
    log_ref_w = -mse_loss(ref_model(preferred), noise)
    log_ref_l = -mse_loss(ref_model(rejected), noise)

    logits = beta * (log_p_w - log_p_l - log_ref_w + log_ref_l)
    return -F.logsigmoid(logits).mean()
```

---

## Results

### Quantitative Comparison

![result](figures/quantative_result.png)

Discriminator scores align with human annotations (<3% deviation).

### Qualitative Results

![result](figures/result.png)

---

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

```bibtex
@article{li2024RORem,
  title={RORem: Training a Robust Object Remover with Human-in-the-Loop},
  author={Ruibin Li and Tao, Yang and Song, Guo and Lei, Zhang},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Acknowledgements

Built on [diffusers](https://github.com/huggingface/diffusers), [LCM](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sdxl_wds.py), and [SDXL-inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1).
