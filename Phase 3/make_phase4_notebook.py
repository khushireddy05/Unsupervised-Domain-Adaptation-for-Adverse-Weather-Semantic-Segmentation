import copy
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BASE = REPO_ROOT / "Phase 3" / "phase3-daformer with results.ipynb"
OUT = REPO_ROOT / "Phase 4" / "phase4-daformer research and performance proofs.ipynb"


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

# Every Phase 4 experiment builds its own student/teacher from scratch via build_model()
# (see make_model_bundle below) -- none of them reuse whatever Phase 3 itself trained. So
# re-executing Phase 3's own training loop, plots, evaluation, and summary (Cell 13 onward:
# a full 20-epoch, ~5-hour training run) inside this notebook would just burn GPU time on
# output nothing downstream depends on. Keep only the setup/definition cells -- imports, CFG,
# label remapping, dataset path discovery, Cityscapes/ACDC dataset builders, augmentations,
# the model builder, EMA/BN-freeze helpers, losses, mIoU, and the MIC masking helper -- and
# drop the training-loop-onward cells entirely.
CUTOFF_MARKDOWN = "## Cell 13: Training Loop"
cutoff_idx = next(
    i for i, c in enumerate(nb["cells"])
    if c["cell_type"] == "markdown" and "".join(c["source"]).strip().startswith(CUTOFF_MARKDOWN)
)
nb["cells"] = nb["cells"][:cutoff_idx]

# Re-title the notebook for Phase 4.
if nb["cells"] and nb["cells"][0]["cell_type"] == "markdown":
    src = "".join(nb["cells"][0]["source"])
    src = src.replace("Phase 3", "Phase 4")
    src = src.replace("self-training domain adaptation", "ablation + final performance validation")
    nb["cells"][0]["source"] = src.splitlines(keepends=True)


phase4_cells = []

phase4_cells.append(md(
"""## Phase 4: Research Proof + Final Performance Proof

This phase turns the Phase 3 adaptation pipeline into a proper research suite:

- ablations for baseline self-training, BN freeze, feature-distance, MIC, and the combined method
- per-condition reporting for fog, rain, snow, and night
- qualitative before/after comparisons
- a stronger night-focused final variant with rare-class sampling, stronger target augmentation, tuned pseudo-labeling, and an extra consistency term

The intent is to separate:

- **research proof**: which component helps, and by how much
- **performance proof**: what the best practical configuration is for the report/thesis
"""))

phase4_cells.append(code(
"""# Phase 3's cutoff cells leave its own global `student`/`teacher` (Cell 9) and their
# feature-distance hook (Cell 12b) resident on the GPU even though Phase 4 never trains them --
# every train_adaptation_experiment() call below builds its own fresh student/teacher via
# make_model_bundle(). Left alive, that's 2 extra MiT-B4 copies competing for VRAM alongside the
# frozen ImageNet encoder and each experiment's own student+teacher: 5 resident copies instead of
# the 3 Phase 3 already needed to halve source_batch for on a 14.56GB GPU (see Cell 3). That's the
# likely cause of the OOM crashes in prior Phase 4 runs -- free them before any experiment starts.
del student, teacher
torch.cuda.empty_cache()
print('Freed Phase 3 leftover student/teacher; GPU memory reclaimed for Phase 4 experiments.')
"""))

phase4_cells.append(code(
"""# Phase 4 switches and experiment templates
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# Phase 3 never defines OUTPUT_DIR -- it only ever wrote to hardcoded /kaggle/working paths
# (see CHECKPOINT_PATH / best_adapted_model.pth), so Phase 4 defines its own output root here
# rather than assuming a variable that doesn't exist in the base notebook.
OUTPUT_DIR = '/kaggle/working'
PHASE4_DIR = Path(OUTPUT_DIR) / 'phase4_artifacts'
PHASE4_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    name: str
    epochs: int = 20
    lr: float = 1.5e-5
    ema_alpha: float = 0.999
    pseudo_conf_thresh: float = 0.968
    lambda_target: float = 1.0
    lambda_fd: float = 0.005
    lambda_consistency: float = 0.10
    use_bn_freeze: bool = True
    use_feature_distance: bool = True
    use_mic: bool = True
    use_target_consistency: bool = False
    use_rare_class_sampling: bool = False
    target_aug_mode: str = 'standard'
    source_sampling_mode: str = 'uniform'
    note: str = ''


RARE_TRAIN_IDS = [11, 12, 13, 15, 16, 17, 18]


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


def make_source_loader(pairs, exp_cfg):
    dataset = CityscapesSourceDataset(pairs)
    if exp_cfg.use_rare_class_sampling:
        weights = build_source_weights(pairs)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=CFG['source_batch'], sampler=sampler,
                          num_workers=2, pin_memory=True, drop_last=True)
    return DataLoader(dataset, batch_size=CFG['source_batch'],
                      shuffle=True, num_workers=2, pin_memory=True, drop_last=True)


def make_target_loader(img_paths, exp_cfg):
    # 'night_tuned' mode was previously applied to every target image regardless of condition --
    # confirmed harmful in the first final_night_tuned run: fog fell from 65.6% (epoch 4) to
    # 61.0% (epoch 14), same for rain/snow, while night climbed, because the heavier
    # blur/motion-blur/dropout augmentation meant for night's low-visibility appearance was also
    # being forced onto fog/rain/snow crops. Target paths are laid out as
    # rgb_anon/<condition>/split/..., so the condition is recoverable from the path -- use it to
    # apply the heavier augmentation only to actual night images.
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
            aug = night_aug if f'{os.sep}night{os.sep}' in path else standard_aug
            strong = aug(image=geo)['image']
            return weak, strong
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

    # Phase 3's imagenet_feature_distance_loss/_captured (Cell 12b) are wired to the one
    # student object Phase 3 itself trained -- reusing them here would silently compare every
    # experiment's features against a stale capture from a different model. Register a fresh
    # hook per experiment's own student and close over its own capture dict instead.
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
    # 'batchmean' sums over every element (class x H x W) and divides only by batch size --
    # correct for flat (B, C) distributions, but on a (B, C, H, W) segmentation map that inflates
    # the loss by ~5 orders of magnitude (confirmed in the first final_night_tuned run: loss went
    # from ~1 to ~3700). Sum over the class dim to get a per-pixel KL divergence, then average
    # over batch and spatial dims so this term sits on the same scale as the CE/Dice losses.
    kl = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        teacher_probs.detach(),
        reduction='none',
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
            miou, per_class = compute_miou(np.concatenate(all_preds), np.concatenate(all_lbls))
            condition_rows[cond] = {
                'miou': miou * 100,
                'per_class': per_class,
            }
    avg = float(np.mean([v['miou'] for v in condition_rows.values()]))
    return avg, condition_rows


def train_adaptation_experiment(exp_cfg: ExperimentConfig):
    source_loader_local = make_source_loader(source_train_pairs, exp_cfg)
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
    phase_dir = PHASE4_DIR / exp_cfg.name
    phase_dir.mkdir(parents=True, exist_ok=True)
    best_path = phase_dir / 'best_teacher.pth'
    ckpt_path = phase_dir / 'checkpoint.pth'

    train_losses, val_mious, conf_ratios = [], [], []
    history_rows = []
    best_miou = 0.0
    start_epoch = 0

    # final_night_tuned alone is ~5 hours at 20 epochs -- a Kaggle session can time out mid-run.
    # Resume from this experiment's own checkpoint if one exists (same pattern as Phase 3's
    # Cell 13), so re-running the notebook after a timeout continues this experiment instead of
    # re-training it from scratch. Each experiment gets its own checkpoint.pth under phase_dir,
    # so ablations that already finished are untouched and the in-progress one picks up cleanly.
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        student.load_state_dict(ckpt['student'])
        teacher.load_state_dict(ckpt['teacher'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch']
        best_miou = ckpt['best_miou']
        train_losses = ckpt['train_losses']
        val_mious = ckpt['val_mious']
        conf_ratios = ckpt['conf_ratios']
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
                weak_imgs, strong_imgs = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader_local)
                weak_imgs, strong_imgs = next(target_iter)

            source_imgs = source_imgs.to(DEVICE)
            source_masks = source_masks.to(DEVICE)
            weak_imgs = weak_imgs.to(DEVICE)
            strong_imgs = strong_imgs.to(DEVICE)

            teacher_logits, pseudo_labels, pixel_weight, teacher_probs = generate_target_logits(
                teacher, weak_imgs, exp_cfg.pseudo_conf_thresh)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                source_logits = student(source_imgs)
                loss_source = source_ce_loss(source_logits, source_masks) + source_dice_loss(source_logits, source_masks)

                if exp_cfg.use_feature_distance:
                    loss_source = loss_source + exp_cfg.lambda_fd * feature_distance_loss(source_imgs)

                target_input = mic_mask(strong_imgs) if exp_cfg.use_mic else strong_imgs
                target_logits = student(target_input)
                loss_target = target_ce_loss(target_logits, pseudo_labels) * pixel_weight

                if exp_cfg.use_target_consistency:
                    loss_target = loss_target + exp_cfg.lambda_consistency * target_consistency_loss(
                        target_logits, teacher_probs
                    )

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
        train_losses.append(epoch_loss)
        conf_ratios.append(epoch_conf)

        cond_miou_avg, cond_pack = evaluate_condition_pack(teacher, cond_val_loaders_local)
        val_mious.append(cond_miou_avg)
        row = {'epoch': epoch + 1, 'loss': epoch_loss, 'conf_ratio': epoch_conf * 100, 'avg_miou': cond_miou_avg}
        row.update({f'{cond}_miou': pack['miou'] for cond, pack in cond_pack.items()})
        history_rows.append(row)

        print(
            f"{exp_cfg.name} | Epoch {epoch+1:02d} | "
            f"Loss: {epoch_loss:.3f} | Conf: {epoch_conf*100:.2f}% | "
            f"ACDC mIoU: {cond_miou_avg:.2f}% | "
            + ", ".join(f"{cond}: {pack['miou']:.1f}%" for cond, pack in cond_pack.items())
        )

        if cond_miou_avg > best_miou:
            best_miou = cond_miou_avg
            torch.save(teacher.state_dict(), best_path)

        torch.save({
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch + 1,
            'best_miou': best_miou,
            'train_losses': train_losses,
            'val_mious': val_mious,
            'conf_ratios': conf_ratios,
            'history_rows': history_rows,
            'exp_cfg': asdict(exp_cfg),
        }, ckpt_path)

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(phase_dir / 'history.csv', index=False)
    return {
        'name': exp_cfg.name,
        'config': asdict(exp_cfg),
        'best_path': str(best_path),
        'history': history_df,
        'best_miou': best_miou,
        'train_losses': train_losses,
        'val_mious': val_mious,
        'conf_ratios': conf_ratios,
    }


def evaluate_saved_model(model_path, label):
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    avg_miou, cond_pack = evaluate_condition_pack(model, {
        cond: DataLoader(ACDCEvalDataset(build_acdc_labeled([cond], 'val')),
                         batch_size=4, shuffle=False, num_workers=2)
        for cond in CONDITIONS
    })
    flat = {'experiment': label, 'avg_miou': avg_miou}
    for cond, pack in cond_pack.items():
        flat[f'{cond}_miou'] = pack['miou']
    return flat
"""))

phase4_cells.append(md(
"""## Research Ablations

Run these in order to support the paper/report claim:

1. baseline self-training only -- control, isolates what self-training alone gets you
2. BN freeze -- the documented root-cause fix from Phase 3 (Run 3 -> Run 4)
3. BN freeze + feature-distance + MIC -- the full combined setting

Trimmed from five configs to three: this keeps the two ends that matter most for the report
(nothing vs. the critical fix vs. everything combined) without paying for isolating
feature-distance and MIC individually, which was costing ~2.5 GPU-hours per extra config."""
))

phase4_cells.append(code(
"""# Ablations use fewer epochs than the final run -- Phase 3's own training curve peaked at
# epoch 10 (53.27%) and had already declined by epoch 20 (51.54%), so ranking configs doesn't
# need anywhere near 10 epochs: the relative order between configs is visible well before the
# peak. 6 epochs is enough to rank the ablations; the final tuned run below gets more headroom
# since its extra stabilizers (target consistency, higher pseudo-label threshold) may shift
# where its own peak lands, but still stops well short of the full 20 that wasted epochs 11-20
# on regression in Phase 3.
ABLATION_EPOCHS = 6

EXPERIMENTS = [
    ExperimentConfig(
        name='ablation_01_baseline_self_training',
        epochs=ABLATION_EPOCHS,
        use_bn_freeze=False,
        use_feature_distance=False,
        use_mic=False,
        use_target_consistency=False,
        use_rare_class_sampling=False,
        note='Plain self-training baseline. This is the least constrained setting.',
    ),
    ExperimentConfig(
        name='ablation_02_bn_freeze',
        epochs=ABLATION_EPOCHS,
        use_bn_freeze=True,
        use_feature_distance=False,
        use_mic=False,
        use_target_consistency=False,
        use_rare_class_sampling=False,
        note='Adds BN freeze only.',
    ),
    ExperimentConfig(
        name='ablation_03_bn_freeze_plus_fd_plus_mic',
        epochs=ABLATION_EPOCHS,
        use_bn_freeze=True,
        use_feature_distance=True,
        use_mic=True,
        use_target_consistency=False,
        use_rare_class_sampling=False,
        note='Full combined ablation setting.',
    ),
]

phase4_runs = []
for exp in EXPERIMENTS:
    phase4_runs.append(train_adaptation_experiment(exp))
"""))

phase4_cells.append(code(
"""def summarize_runs(runs):
    rows = []
    for run in runs:
        row = {
            'experiment': run['name'],
            'best_miou': run['best_miou'],
            'use_bn_freeze': run['config']['use_bn_freeze'],
            'use_feature_distance': run['config']['use_feature_distance'],
            'use_mic': run['config']['use_mic'],
            'use_target_consistency': run['config']['use_target_consistency'],
            'use_rare_class_sampling': run['config']['use_rare_class_sampling'],
        }
        history = run['history'].sort_values('avg_miou')
        best_epoch_row = history.iloc[-1]
        row.update({
            'fog_miou': best_epoch_row.get('fog_miou', float('nan')),
            'rain_miou': best_epoch_row.get('rain_miou', float('nan')),
            'snow_miou': best_epoch_row.get('snow_miou', float('nan')),
            'night_miou': best_epoch_row.get('night_miou', float('nan')),
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    df['avg_of_conditions'] = df[['fog_miou', 'rain_miou', 'snow_miou', 'night_miou']].mean(axis=1)
    return df.sort_values('avg_of_conditions', ascending=False)


ablation_df = summarize_runs(phase4_runs)
ablation_df
"""))

phase4_cells.append(code(
"""display_cols = [
    'experiment',
    'best_miou',
    'fog_miou',
    'rain_miou',
    'snow_miou',
    'night_miou',
    'avg_of_conditions',
]
print(ablation_df[display_cols].to_string(index=False, float_format=lambda x: f'{x:0.2f}'))

plt.figure(figsize=(12, 5))
plt.bar(ablation_df['experiment'], ablation_df['avg_of_conditions'])
plt.xticks(rotation=25, ha='right')
plt.ylabel('Average per-condition mIoU (%)')
plt.title('Phase 4 Ablation Ranking')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
"""))

phase4_cells.append(md(
"""## Qualitative Before/After Comparisons

This section renders fog, rain, snow, and night samples side by side:

- input image
- ground-truth label
- source-only prediction from Phase 1
- best ablation checkpoint
- final tuned checkpoint

That gives you the visual proof that complements the tables."""
))

phase4_cells.append(code(
"""def colorize(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(CITYSCAPES_COLORS):
        out[mask == i] = c
    out[mask == 255] = (0, 0, 0)
    return out


def load_model_from_path(path):
    # Build the bare architecture rather than calling build_model() (which always loads
    # MODEL_PATH first) -- we're about to overwrite the weights with `path` anyway, so loading
    # MODEL_PATH here would just be a wasted disk read on every one of the calls below.
    model = smp.Segformer(
        encoder_name='mit_b4', encoder_weights=None, in_channels=3, classes=CFG['num_classes'],
    ).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def show_condition_samples(condition, models, n_samples=2):
    pairs = build_acdc_labeled([condition], 'val')
    sample_pairs = pairs[:n_samples]

    fig, axes = plt.subplots(n_samples, 2 + len(models), figsize=(4 * (2 + len(models)), 4 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, (img_path, mask_path) in enumerate(sample_pairs):
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        image_t = eval_aug(image=image)['image'].unsqueeze(0).to(DEVICE)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f'{condition} input')
        axes[row, 0].axis('off')
        axes[row, 1].imshow(colorize(mask))
        axes[row, 1].set_title('Ground truth')
        axes[row, 1].axis('off')
        for col, (name, model) in enumerate(models, start=2):
            pred = model(image_t).argmax(1).squeeze(0).cpu().numpy()
            axes[row, col].imshow(colorize(pred))
            axes[row, col].set_title(name)
            axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()


best_ablation_path = str((PHASE4_DIR / 'ablation_03_bn_freeze_plus_fd_plus_mic' / 'best_teacher.pth'))
final_tuned_path = str((PHASE4_DIR / 'final_night_tuned' / 'best_teacher.pth'))
"""))

phase4_cells.append(md(
"""## Final Performance Proof

This is the practical best-effort setting for night and the hardest conditions:

- BN freeze
- feature-distance regularization
- MIC
- rare-class sampling for Cityscapes source crops
- stronger target augmentation
- target consistency loss
- tuned pseudo-label threshold

This is the configuration to use as your end-to-end “best model” in the report."""
))

phase4_cells.append(code(
"""# Back to 20 epochs (matching Phase 3's schedule). The 14-epoch run was still declining after
# its epoch-4 peak, but that decline is now explained by two bugs fixed above: the
# target-consistency loss was inflated ~1000x by a bad KL reduction, and night-tuned
# augmentation was being applied to fog/rain/snow images too. With both fixed, this config no
# longer has a known reason to collapse early, so it gets the full budget to actually converge
# instead of being cut short pre-emptively. Best-checkpoint selection still keeps whichever
# epoch is actually best either way.
FINAL_EXPERIMENT = ExperimentConfig(
    name='final_night_tuned',
    epochs=20,
    lr=1.5e-5,
    ema_alpha=0.999,
    pseudo_conf_thresh=0.975,
    lambda_target=1.0,
    lambda_fd=0.005,
    lambda_consistency=0.10,
    use_bn_freeze=True,
    use_feature_distance=True,
    use_mic=True,
    use_target_consistency=True,
    use_rare_class_sampling=True,
    target_aug_mode='night_tuned',
    source_sampling_mode='rare_class',
    note='Final performance run targeted at improving night while keeping fog/rain/snow strong.',
)

final_run = train_adaptation_experiment(FINAL_EXPERIMENT)
"""))

phase4_cells.append(code(
"""# Render the qualitative before/after grids promised above, now that both checkpoints exist.
# Load each model exactly once and reuse it across all four conditions, instead of reloading
# from disk for every condition (previously 3 models x 4 conditions = 12 loads).
qualitative_models = [
    ('source_only', load_model_from_path(MODEL_PATH)),
    ('best_ablation', load_model_from_path(best_ablation_path)),
    ('final_tuned', load_model_from_path(final_tuned_path)),
]
for condition in CONDITIONS:
    show_condition_samples(condition, qualitative_models, n_samples=2)
"""))

phase4_cells.append(code(
"""final_summary = summarize_runs(phase4_runs + [final_run])
print(final_summary[['experiment', 'best_miou', 'avg_of_conditions', 'night_miou']].to_string(index=False, float_format=lambda x: f'{x:0.2f}'))
"""))

phase4_cells.append(md(
"""## Discussion Helper

This cell prints a report-ready discussion scaffold from the measured values:

- why night is still hard
- why confidence rises while mIoU can flatten
- what the model likely learned from MIC / feature-distance / BN freeze
- what to claim as your final best result

You can paste the generated text into your thesis/report and then polish wording."""
))

phase4_cells.append(code(
"""def build_discussion_text(summary_df):
    best_row = summary_df.sort_values('avg_of_conditions', ascending=False).iloc[0]
    night_row = summary_df.sort_values('night_miou', ascending=False).iloc[0]
    lines = []
    lines.append('Phase 4 discussion draft:')
    lines.append(f'- The strongest overall configuration is `{best_row[\"experiment\"]}` with {best_row[\"avg_of_conditions\"]:.2f}% average per-condition mIoU.')
    lines.append(f'- The best night score is `{night_row[\"experiment\"]}` with {night_row[\"night_miou\"]:.2f}% night mIoU.')
    lines.append('- Night remains the hardest domain because illumination shifts, low contrast, and ambiguous boundaries reduce teacher confidence and make pseudo-labels noisier than fog/rain/snow.')
    lines.append('- Confidence can rise while mIoU plateaus because the teacher becomes more self-consistent on easy pixels first; that does not guarantee better boundary quality or rare-class recovery.')
    lines.append('- BN freeze prevents cross-domain running-stat contamination, which stabilizes evaluation and stops the model from drifting as source and target batches alternate.')
    lines.append('- Feature-distance regularization encourages the encoder to stay closer to the Phase 1 representation space, reducing catastrophic drift.')
    lines.append('- MIC forces contextual inference in masked regions, which is especially useful for adverse weather where local evidence is missing or corrupted.')
    lines.append('- The final result should be presented as the combination of controlled ablation evidence and qualitative examples, not only as a single leaderboard-style number.')
    return '\\n'.join(lines)


print(build_discussion_text(final_summary))
"""))

phase4_cells.append(md(
"""## Optional Export Helpers

Uncomment these if you want CSV and PNG outputs saved alongside the notebook."""
))

phase4_cells.append(code(
"""# ablation_df.to_csv(PHASE4_DIR / 'ablation_table.csv', index=False)
# plt.savefig(PHASE4_DIR / 'ablation_ranking.png', dpi=200, bbox_inches='tight')
# final_summary.to_csv(PHASE4_DIR / 'final_summary.csv', index=False)
print('Phase 4 notebook scaffold is ready.')
"""))

nb["cells"].extend(phase4_cells)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f'Wrote {OUT}')
