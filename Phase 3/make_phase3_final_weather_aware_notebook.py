import copy
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BASE = REPO_ROOT / "phase3-daformer with results.ipynb"
OUT = REPO_ROOT / "phase3-final-weather-aware.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text if isinstance(text, list) else text.splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text if isinstance(text, list) else text.splitlines(keepends=True),
    }


nb = json.loads(BASE.read_text())

CUTOFF_MARKDOWN = "## Cell 13: Training Loop"
cutoff_idx = next(
    i for i, c in enumerate(nb["cells"])
    if c["cell_type"] == "markdown" and "".join(c["source"]).strip().startswith(CUTOFF_MARKDOWN)
)
nb["cells"] = nb["cells"][:cutoff_idx]

if nb["cells"] and nb["cells"][0]["cell_type"] == "markdown":
    src = "".join(nb["cells"][0]["source"])
    src = src.replace("Phase 3", "Phase 3 -- Final Config + Weather-Aware MIC")
    src = src.replace(
        "self-training domain adaptation",
        "the best-known final configuration, with weather-aware MIC swapped in for generic MIC",
    )
    nb["cells"][0]["source"] = src.splitlines(keepends=True)


cells = []

cells.append(md(
"""## Final Config + Weather-Aware MIC

The weather-aware-vs-generic masking ablation (5 epochs each, controlled comparison) found that
weather-aware MIC beats generic MIC by +0.38 avg mIoU, concentrated almost entirely in night
(+2.18 points: 34.06% vs 31.88%), with fog/rain/snow roughly unchanged. That was tested in
isolation, on top of just BN freeze + feature-distance + MIC.

This notebook asks the follow-up question: does that same benefit hold up when weather-aware MIC
is folded into the actual best-known final configuration -- the one that already has rare-class
sampling, night-tuned augmentation, a target-consistency loss, and a tuned pseudo-label threshold?
That combined config (with *generic* MIC) reached **54.19% avg mIoU, 34.9% night** in the prior
run -- still marginally below night's 35.35% zero-shot baseline. This is the first test of whether
weather-aware masking can close that specific gap.

Only the weather-aware variant is run here (not a duplicate generic re-run) to fit inside the
remaining weekly GPU quota -- its result is compared against the already-known 54.19%/34.9%
generic-MIC number from the prior run rather than an identical-code re-run. Checkpoint-resume is
included since a single 20-epoch run takes ~5 hours, close to what's left of the weekly quota."""
))

cells.append(code(
"""del student, teacher
torch.cuda.empty_cache()
print('Freed Phase 3 leftover student/teacher; GPU memory reclaimed.')
"""))

cells.append(code(
"""from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = '/kaggle/working'
FINAL_DIR = Path(OUTPUT_DIR) / 'final_weather_aware_artifacts'
FINAL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    name: str
    epochs: int = 20
    lr: float = 1.5e-5
    ema_alpha: float = 0.999
    pseudo_conf_thresh: float = 0.975
    lambda_target: float = 1.0
    lambda_fd: float = 0.005
    lambda_consistency: float = 0.10
    use_bn_freeze: bool = True
    use_feature_distance: bool = True
    use_mic: bool = True
    mic_mode: str = 'weather_aware'    # 'generic' or 'weather_aware'
    mic_patch_size: int = 32
    mic_mask_ratio: float = 0.5
    use_target_consistency: bool = True
    use_rare_class_sampling: bool = True
    target_aug_mode: str = 'night_tuned'
    note: str = ''


RARE_TRAIN_IDS = [11, 12, 13, 15, 16, 17, 18]


def condition_from_path(path):
    for cond in CONDITIONS:
        if f'{os.sep}{cond}{os.sep}' in path:
            return cond
    return 'unknown'


def _block_mask_single(img, patch_size, mask_ratio, fill):
    C, H, W = img.shape
    masked = img.clone()
    n_h, n_w = max(1, H // patch_size), max(1, W // patch_size)
    n_patches = n_h * n_w
    n_masked = int(n_patches * mask_ratio)
    if n_masked > 0:
        idx = np.random.choice(n_patches, n_masked, replace=False)
        for i in idx:
            r, c = divmod(i, n_w)
            masked[:, r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size] = fill
    return masked


def weather_aware_mask_single(img, condition, patch_size=32, mask_ratio=0.5):
    C, H, W = img.shape
    masked = img.clone()
    if condition == 'fog':
        area_h = max(1, int(H * mask_ratio ** 0.5))
        area_w = max(1, int(W * mask_ratio ** 0.5))
        top = np.random.randint(0, max(1, H - area_h + 1))
        left = np.random.randint(0, max(1, W - area_w + 1))
        masked[:, top:top + area_h, left:left + area_w] = 0.0
    elif condition == 'rain':
        n_streaks = max(1, int((H * W * mask_ratio) / (H * 3)))
        for _ in range(n_streaks):
            col = np.random.randint(0, W)
            thickness = np.random.randint(2, 4)
            drift = np.random.choice([-1, 0, 1])
            c = col
            for row in range(H):
                c0 = max(0, min(W - thickness, c))
                masked[:, row, c0:c0 + thickness] = 0.0
                if row % 3 == 0:
                    c += drift
    elif condition == 'night':
        masked = _block_mask_single(masked, patch_size, mask_ratio, fill=-1.8)
    elif condition == 'snow':
        masked = _block_mask_single(masked, patch_size, mask_ratio, fill=1.8)
    else:
        masked = _block_mask_single(masked, patch_size, mask_ratio, fill=0.0)
    return masked


def apply_mic_mask(img, condition, exp_cfg):
    if exp_cfg.mic_mode == 'weather_aware':
        return weather_aware_mask_single(img, condition, exp_cfg.mic_patch_size, exp_cfg.mic_mask_ratio)
    return _block_mask_single(img, exp_cfg.mic_patch_size, exp_cfg.mic_mask_ratio, fill=0.0)


def target_strong_aug_factory(mode: str):
    if mode == 'night_tuned':
        return A.Compose([
            A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.06, p=0.9),
            A.GaussianBlur(blur_limit=(3, 7), p=0.20),
            A.MotionBlur(blur_limit=7, p=0.15),
            A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(16, 64),
                            hole_width_range=(16, 64), fill=0, p=0.20),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.30, hue=0.05, p=0.80),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def source_mask_contains_rare(mask_path):
    mask = remap_mask(cv2.imread(mask_path, 0))
    uniq = set(np.unique(mask).tolist())
    return any(cls in uniq for cls in RARE_TRAIN_IDS)


def build_source_weights(pairs, boost=3.0):
    weights = []
    for _, mask_path in pairs:
        weights.append(boost if source_mask_contains_rare(mask_path) else 1.0)
    return torch.DoubleTensor(weights)


def make_source_loader(exp_cfg):
    dataset = CityscapesSourceDataset(source_train_pairs)
    if exp_cfg.use_rare_class_sampling:
        weights = build_source_weights(source_train_pairs)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=CFG['source_batch'], sampler=sampler,
                          num_workers=2, pin_memory=True, drop_last=True)
    return DataLoader(dataset, batch_size=CFG['source_batch'],
                      shuffle=True, num_workers=2, pin_memory=True, drop_last=True)


def make_target_loader(img_paths, exp_cfg):
    # Two independent, condition-aware mechanisms: (1) night-tuned strong augmentation only for
    # night images (fixed after it was found to degrade fog/rain/snow when applied everywhere),
    # and (2) MIC masking, generic or weather-aware depending on exp_cfg.mic_mode.
    standard_aug = target_strong_aug_factory('standard')
    night_aug = target_strong_aug_factory('night_tuned') if exp_cfg.target_aug_mode == 'night_tuned' else standard_aug

    class _TargetDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            path = self.paths[idx]
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            geo = target_geo_only(image=image)['image']
            weak = target_weak_aug(image=geo)['image']
            cond = condition_from_path(path)
            aug = night_aug if cond == 'night' else standard_aug
            strong = aug(image=geo)['image']
            target_input = apply_mic_mask(strong, cond, exp_cfg) if exp_cfg.use_mic else strong
            return weak, target_input
    return DataLoader(_TargetDataset(img_paths), batch_size=CFG['target_batch'],
                      shuffle=True, num_workers=2, pin_memory=True, drop_last=True)


def make_model_bundle(exp_cfg):
    student = build_model()
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    if exp_cfg.use_bn_freeze:
        freeze_bn(student)

    captured = {}

    def _capture_bottleneck(module, inputs, output):
        captured['student_feat'] = output[-1]

    student.encoder.register_forward_hook(_capture_bottleneck)

    def feature_distance_loss(source_imgs):
        with torch.no_grad():
            frozen_feat = frozen_imagenet_encoder(source_imgs)[-1]
        return F.mse_loss(captured['student_feat'], frozen_feat.detach())

    return student, teacher, feature_distance_loss


@torch.no_grad()
def generate_target_logits(teacher, weak_images, conf_thresh):
    logits = teacher(weak_images)
    probs = F.softmax(logits, dim=1)
    conf, pseudo_labels = probs.max(dim=1)
    pixel_weight = (conf >= conf_thresh).float().mean()
    return logits, pseudo_labels, pixel_weight, probs


def target_consistency_loss(student_logits, teacher_probs):
    # Sum over the class dim first, then average over batch+spatial -- 'batchmean' alone
    # inflates this ~1000x on a (B, C, H, W) segmentation map (see the earlier tuning run).
    kl = F.kl_div(
        F.log_softmax(student_logits, dim=1), teacher_probs.detach(), reduction='none',
    ).sum(dim=1)
    return kl.mean()


def evaluate_condition_pack(model, loader_map):
    model.eval()
    condition_rows = {}
    with torch.no_grad():
        for cond, loader in loader_map.items():
            all_preds, all_lbls = [], []
            for imgs, lbls in loader:
                preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
                all_preds.append(preds)
                all_lbls.append(lbls.numpy())
            preds_np = np.concatenate(all_preds)
            lbls_np = np.concatenate(all_lbls)
            miou, per_class = compute_miou(preds_np, lbls_np)
            valid = lbls_np != 255
            pixel_acc = float((preds_np[valid] == lbls_np[valid]).mean()) if valid.any() else 0.0
            condition_rows[cond] = {'miou': miou * 100, 'pixel_acc': pixel_acc * 100}
    avg_miou = float(np.mean([v['miou'] for v in condition_rows.values()]))
    avg_pixel_acc = float(np.mean([v['pixel_acc'] for v in condition_rows.values()]))
    return avg_miou, avg_pixel_acc, condition_rows


def train_experiment(exp_cfg: ExperimentConfig):
    source_loader_local = make_source_loader(exp_cfg)
    target_loader_local = make_target_loader(target_train_images, exp_cfg)
    student, teacher, feature_distance_loss = make_model_bundle(exp_cfg)

    optimizer = torch.optim.AdamW(student.parameters(), lr=exp_cfg.lr, weight_decay=1e-4)
    steps_per_epoch = len(source_loader_local)
    total_iters = exp_cfg.epochs * steps_per_epoch
    warmup_iters = min(500, steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_iters:
            return step / max(1, warmup_iters)
        progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        return (1 - progress) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler()
    exp_dir = FINAL_DIR / exp_cfg.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    best_path = exp_dir / 'best_teacher.pth'
    ckpt_path = exp_dir / 'checkpoint.pth'

    history_rows = []
    best_miou = 0.0
    start_epoch = 0

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        student.load_state_dict(ckpt['student'])
        teacher.load_state_dict(ckpt['teacher'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch']
        best_miou = ckpt['best_miou']
        history_rows = ckpt['history_rows']
        print(f'{exp_cfg.name}: resuming from epoch {start_epoch}, best mIoU so far: {best_miou:.2f}%')

    cond_val_loaders_local = {
        cond: DataLoader(ACDCEvalDataset(build_acdc_labeled([cond], 'val')),
                         batch_size=4, shuffle=False, num_workers=2)
        for cond in CONDITIONS
    }

    for epoch in range(start_epoch, exp_cfg.epochs):
        student.train()
        if exp_cfg.use_bn_freeze:
            freeze_bn(student)
        running_loss = 0.0
        running_conf = 0.0
        target_iter = iter(target_loader_local)

        for source_imgs, source_masks in tqdm(source_loader_local, desc=f'{exp_cfg.name} | epoch {epoch+1}/{exp_cfg.epochs}'):
            try:
                weak_imgs, target_input = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader_local)
                weak_imgs, target_input = next(target_iter)

            source_imgs = source_imgs.to(DEVICE)
            source_masks = source_masks.to(DEVICE)
            weak_imgs = weak_imgs.to(DEVICE)
            target_input = target_input.to(DEVICE)

            teacher_logits, pseudo_labels, pixel_weight, teacher_probs = generate_target_logits(
                teacher, weak_imgs, exp_cfg.pseudo_conf_thresh)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                source_logits = student(source_imgs)
                loss_source = source_ce_loss(source_logits, source_masks) + source_dice_loss(source_logits, source_masks)

                if exp_cfg.use_feature_distance:
                    loss_source = loss_source + exp_cfg.lambda_fd * feature_distance_loss(source_imgs)

                target_logits = student(target_input)
                loss_target = target_ce_loss(target_logits, pseudo_labels) * pixel_weight

                if exp_cfg.use_target_consistency:
                    loss_target = loss_target + exp_cfg.lambda_consistency * target_consistency_loss(
                        target_logits, teacher_probs)

                loss = loss_source + exp_cfg.lambda_target * loss_target

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema_update(teacher, student, exp_cfg.ema_alpha)

            running_loss += loss.item()
            running_conf += pixel_weight.item()

        epoch_loss = running_loss / steps_per_epoch
        epoch_conf = running_conf / steps_per_epoch

        cond_miou_avg, cond_pixel_acc_avg, cond_pack = evaluate_condition_pack(teacher, cond_val_loaders_local)
        row = {'epoch': epoch + 1, 'loss': epoch_loss, 'conf_ratio': epoch_conf * 100,
               'avg_miou': cond_miou_avg, 'avg_pixel_acc': cond_pixel_acc_avg}
        row.update({f'{cond}_miou': pack['miou'] for cond, pack in cond_pack.items()})
        history_rows.append(row)

        print(
            f"{exp_cfg.name} | Epoch {epoch+1:02d} | "
            f"Loss: {epoch_loss:.3f} | Conf: {epoch_conf*100:.2f}% | "
            f"ACDC mIoU: {cond_miou_avg:.2f}% | Pixel acc: {cond_pixel_acc_avg:.2f}% | "
            + ", ".join(f"{cond}: {pack['miou']:.1f}%" for cond, pack in cond_pack.items())
        )

        if cond_miou_avg > best_miou:
            best_miou = cond_miou_avg
            torch.save(teacher.state_dict(), best_path)

        torch.save({
            'student': student.state_dict(), 'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'best_miou': best_miou,
            'history_rows': history_rows, 'exp_cfg': asdict(exp_cfg),
        }, ckpt_path)

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(exp_dir / 'history.csv', index=False)
    return {'name': exp_cfg.name, 'config': asdict(exp_cfg), 'best_path': str(best_path),
            'history': history_df, 'best_miou': best_miou}
"""))

cells.append(md(
"""## Run

Same hyperparameters as the previously-best `final_night_tuned` config (BN freeze, feature-distance,
rare-class sampling, night-tuned augmentation, target consistency, threshold 0.975, 20 epochs) --
only `mic_mode` changes, from generic to weather-aware."""
))

cells.append(code(
"""FINAL_EXPERIMENT = ExperimentConfig(
    name='final_weather_aware_tuned',
    mic_mode='weather_aware',
    note='Same as the best-known final_night_tuned config, with weather-aware MIC replacing '
         'generic MIC to test whether the night gain found in the masking ablation holds up '
         'combined with rare-class sampling, night-tuned aug, and the consistency loss.',
)

final_run = train_experiment(FINAL_EXPERIMENT)
"""))

cells.append(md(
"""## Compare Against the Known Generic-MIC Result

`GENERIC_MIC_REFERENCE` is the already-measured result from the prior `final_night_tuned` run
(generic MIC, otherwise identical config) -- not re-run here to fit inside the remaining weekly
GPU quota, so this is a same-config-different-code-path comparison rather than a literal
same-session rerun."""
))

cells.append(code(
"""GENERIC_MIC_REFERENCE = {
    'experiment': 'final_night_tuned (generic MIC, prior run)',
    'avg_miou': 54.19, 'fog_miou': 67.4, 'rain_miou': 56.6, 'snow_miou': 57.8, 'night_miou': 34.9,
}

best_row = final_run['history'].sort_values('avg_miou').iloc[-1]
weather_aware_result = {
    'experiment': 'final_weather_aware_tuned (this run)',
    'avg_miou': best_row['avg_miou'],
    'fog_miou': best_row['fog_miou'], 'rain_miou': best_row['rain_miou'],
    'snow_miou': best_row['snow_miou'], 'night_miou': best_row['night_miou'],
}

comparison_df = pd.DataFrame([GENERIC_MIC_REFERENCE, weather_aware_result])
print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:0.2f}'))

night_delta = weather_aware_result['night_miou'] - GENERIC_MIC_REFERENCE['night_miou']
avg_delta = weather_aware_result['avg_miou'] - GENERIC_MIC_REFERENCE['avg_miou']
zero_shot_night = 35.35
print()
print(f\"Night: {weather_aware_result['night_miou']:.2f}% vs generic MIC's {GENERIC_MIC_REFERENCE['night_miou']:.2f}% ({night_delta:+.2f} pts)\")
print(f\"Avg:   {weather_aware_result['avg_miou']:.2f}% vs generic MIC's {GENERIC_MIC_REFERENCE['avg_miou']:.2f}% ({avg_delta:+.2f} pts)\")
if weather_aware_result['night_miou'] > zero_shot_night:
    print(f\"Night now EXCEEDS its {zero_shot_night}% zero-shot baseline for the first time across all runs.\")
else:
    print(f\"Night still below its {zero_shot_night}% zero-shot baseline \"
          f\"(short by {zero_shot_night - weather_aware_result['night_miou']:.2f} pts).\")

plt.figure(figsize=(8, 5))
x = np.arange(4)
width = 0.35
plt.bar(x - width/2, [GENERIC_MIC_REFERENCE['fog_miou'], GENERIC_MIC_REFERENCE['rain_miou'],
                       GENERIC_MIC_REFERENCE['snow_miou'], GENERIC_MIC_REFERENCE['night_miou']],
        width, label='Generic MIC (prior run)', color='#4c72b0')
plt.bar(x + width/2, [weather_aware_result['fog_miou'], weather_aware_result['rain_miou'],
                       weather_aware_result['snow_miou'], weather_aware_result['night_miou']],
        width, label='Weather-aware MIC (this run)', color='#dd8452')
plt.axhline(zero_shot_night, color='gray', linestyle='--', linewidth=1, label='Night zero-shot baseline')
plt.xticks(x, ['Fog', 'Rain', 'Snow', 'Night'])
plt.ylabel('mIoU (%)')
plt.title('Final Config: Generic vs. Weather-Aware MIC')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
"""))

nb["cells"].extend(cells)

OUT.write_text(json.dumps(nb, indent=1))
print(f'Wrote {OUT}')
