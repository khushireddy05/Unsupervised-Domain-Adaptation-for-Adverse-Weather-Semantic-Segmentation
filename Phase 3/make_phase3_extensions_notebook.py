import copy
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BASE = REPO_ROOT / "phase3-daformer with results.ipynb"
OUT = REPO_ROOT / "phase3-extensions.ipynb"


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
    src = src.replace("Phase 3", "Phase 3 -- Remaining Proposal Extensions")
    src = src.replace(
        "self-training domain adaptation",
        "weather-aware masking vs. generic MIC, and a confidence-threshold sweep",
    )
    nb["cells"][0]["source"] = src.splitlines(keepends=True)


cells = []

cells.append(md(
"""## Remaining Proposal Extensions: Weather-Aware Masking + Confidence-Threshold Sweep

Two open items from the proposal, combined into one notebook since they share almost all the same
infrastructure:

1. **Weather-aware masking** -- does replacing MIC's generic random-block masking with masks
   shaped to match each condition's actual degradation (haze region for fog, streaks for rain,
   dark blocks for night, bright blocks for snow) beat generic masking?
2. **Confidence-threshold sweep** -- `pseudo_conf_thresh` has only ever been tried at two values
   by intuition (0.968, then 0.975 in the final tuned run); this sweeps it systematically.

Five experiments run in total, not six: the generic-MIC run at the default threshold (0.968) does
double duty as both the "generic" side of the masking comparison and the middle point of the
threshold sweep, saving a redundant ~75-minute run. Each is BN freeze + feature-distance + MIC,
5 epochs, on top of the Phase 1 checkpoint -- only `mic_mode` and `pseudo_conf_thresh` vary.
Pixel accuracy is reported throughout, which the proposal asks for but no prior run has logged.
"""))

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
EXT_DIR = Path(OUTPUT_DIR) / 'phase3_extensions_artifacts'
EXT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    name: str
    epochs: int = 5
    lr: float = 1.5e-5
    ema_alpha: float = 0.999
    pseudo_conf_thresh: float = 0.968
    lambda_target: float = 1.0
    lambda_fd: float = 0.005
    use_bn_freeze: bool = True
    use_feature_distance: bool = True
    use_mic: bool = True
    mic_mode: str = 'generic'          # 'generic' or 'weather_aware'
    mic_patch_size: int = 32
    mic_mask_ratio: float = 0.5
    note: str = ''


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
    \"\"\"Condition-specific mask shape and fill value: a contiguous haze-band region for fog,
    diagonal streaks for rain, dark-filled blocks for night, bright-filled blocks for snow --
    deliberately different in kind, not just fill value, from generic MIC's scattered blocks.\"\"\"
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


def make_source_loader():
    dataset = CityscapesSourceDataset(source_train_pairs)
    return DataLoader(dataset, batch_size=CFG['source_batch'],
                      shuffle=True, num_workers=2, pin_memory=True, drop_last=True)


def make_target_loader(img_paths, exp_cfg):
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
            strong = target_strong_aug(image=geo)['image']
            cond = condition_from_path(path)
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
    return logits, pseudo_labels, pixel_weight


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
            condition_rows[cond] = {
                'miou': miou * 100,
                'pixel_acc': pixel_acc * 100,
                'per_class': per_class,
            }
    avg_miou = float(np.mean([v['miou'] for v in condition_rows.values()]))
    avg_pixel_acc = float(np.mean([v['pixel_acc'] for v in condition_rows.values()]))
    return avg_miou, avg_pixel_acc, condition_rows


def train_experiment(exp_cfg: ExperimentConfig):
    source_loader_local = make_source_loader()
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
    exp_dir = EXT_DIR / exp_cfg.name
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

            teacher_logits, pseudo_labels, pixel_weight = generate_target_logits(
                teacher, weak_imgs, exp_cfg.pseudo_conf_thresh)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                source_logits = student(source_imgs)
                loss_source = source_ce_loss(source_logits, source_masks) + source_dice_loss(source_logits, source_masks)

                if exp_cfg.use_feature_distance:
                    loss_source = loss_source + exp_cfg.lambda_fd * feature_distance_loss(source_imgs)

                target_logits = student(target_input)
                loss_target = target_ce_loss(target_logits, pseudo_labels) * pixel_weight

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
"""## Run All Five Experiments

`mic_generic_thresh0968` is the shared anchor point: generic MIC at the default threshold. It
feeds both analyses below."""
))

cells.append(code(
"""EXPERIMENTS = [
    ExperimentConfig(name='mic_generic_thresh0968', mic_mode='generic', pseudo_conf_thresh=0.968,
        note='Anchor run: generic MIC at the default threshold. Doubles as the "generic" side of '
             'the masking comparison and the middle point of the threshold sweep.'),
    ExperimentConfig(name='mic_weather_aware', mic_mode='weather_aware', pseudo_conf_thresh=0.968,
        note='Condition-shaped MIC masks: haze region for fog, streaks for rain, dark blocks for '
             'night, bright blocks for snow.'),
    ExperimentConfig(name='thresh_0_90', mic_mode='generic', pseudo_conf_thresh=0.90),
    ExperimentConfig(name='thresh_0_95', mic_mode='generic', pseudo_conf_thresh=0.95),
    ExperimentConfig(name='thresh_0_985', mic_mode='generic', pseudo_conf_thresh=0.985),
]

runs = {}
for exp in EXPERIMENTS:
    runs[exp.name] = train_experiment(exp)
"""))

cells.append(md(
"""## Analysis 1: Generic MIC vs. Weather-Aware MIC

Answers the proposal's question: "Can weather-aware masking strategies further improve robustness
compared to generic masking approaches?" """
))

cells.append(code(
"""def best_epoch_row(run):
    return run['history'].sort_values('avg_miou').iloc[-1]

def summarize(names, index_label):
    rows = []
    for name in names:
        run = runs[name]
        row = best_epoch_row(run)
        rows.append({
            index_label: name,
            'best_miou': run['best_miou'],
            'fog_miou': row.get('fog_miou', float('nan')),
            'rain_miou': row.get('rain_miou', float('nan')),
            'snow_miou': row.get('snow_miou', float('nan')),
            'night_miou': row.get('night_miou', float('nan')),
            'avg_pixel_acc': row.get('avg_pixel_acc', float('nan')),
        })
    df = pd.DataFrame(rows)
    df['avg_of_conditions'] = df[['fog_miou', 'rain_miou', 'snow_miou', 'night_miou']].mean(axis=1)
    return df


masking_summary = summarize(['mic_generic_thresh0968', 'mic_weather_aware'], 'experiment')
print(masking_summary.to_string(index=False, float_format=lambda x: f'{x:0.2f}'))

plt.figure(figsize=(7, 5))
bars = plt.bar(masking_summary['experiment'], masking_summary['avg_of_conditions'],
                color=['#4c72b0', '#dd8452'])
for b, v in zip(bars, masking_summary['avg_of_conditions']):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold')
plt.ylabel('Average per-condition mIoU (%)')
plt.title('Generic MIC vs. Weather-Aware MIC')
plt.xticks(rotation=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
"""))

cells.append(md(
"""### Qualitative Comparison"""
))

cells.append(code(
"""def colorize_pred(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(CITYSCAPES_COLORS):
        out[mask == i] = c
    out[mask == 255] = (0, 0, 0)
    return out


def load_model_from_path(path):
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
        axes[row, 1].imshow(colorize_pred(mask))
        axes[row, 1].set_title('Ground truth')
        axes[row, 1].axis('off')
        for col, (name, model) in enumerate(models, start=2):
            pred = model(image_t).argmax(1).squeeze(0).cpu().numpy()
            axes[row, col].imshow(colorize_pred(pred))
            axes[row, col].set_title(name)
            axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()


qualitative_models = [
    ('source_only', load_model_from_path(MODEL_PATH)),
    ('generic_mic', load_model_from_path(runs['mic_generic_thresh0968']['best_path'])),
    ('weather_aware_mic', load_model_from_path(runs['mic_weather_aware']['best_path'])),
]
for condition in CONDITIONS:
    show_condition_samples(condition, qualitative_models, n_samples=2)
"""))

cells.append(md(
"""## Analysis 2: Confidence-Threshold Sweep

`pseudo_conf_thresh` doesn't mask individual pixels -- per Phase 3's Cell 11 design, every pixel's
pseudo-label is kept, and the confident-pixel ratio is used as a single global weight on the whole
target loss for that batch. So this sweep shows how strongly the target loss ends up weighted, and
whether that trades off between overall performance and night specifically."""
))

cells.append(code(
"""threshold_summary = summarize(
    ['thresh_0_90', 'thresh_0_95', 'mic_generic_thresh0968', 'thresh_0_985'], 'experiment')
threshold_summary['threshold'] = [0.90, 0.95, 0.968, 0.985]
threshold_summary = threshold_summary.sort_values('threshold')
threshold_summary['final_conf_ratio'] = [
    best_epoch_row(runs[n])['conf_ratio']
    for n in ['thresh_0_90', 'thresh_0_95', 'mic_generic_thresh0968', 'thresh_0_985']
]
print(threshold_summary.to_string(index=False, float_format=lambda x: f'{x:0.2f}'))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(threshold_summary['threshold'], threshold_summary['avg_of_conditions'], marker='o', label='Avg (all conditions)')
axes[0].plot(threshold_summary['threshold'], threshold_summary['night_miou'], marker='o', label='Night only')
axes[0].set_xlabel('pseudo_conf_thresh')
axes[0].set_ylabel('mIoU (%)')
axes[0].set_title('mIoU vs. Confidence Threshold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(threshold_summary['threshold'], threshold_summary['final_conf_ratio'], marker='o', color='#8e44ad')
axes[1].set_xlabel('pseudo_conf_thresh')
axes[1].set_ylabel('Confident-pixel ratio at final epoch (%)')
axes[1].set_title('How Much of the Target Loss Actually Counts')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
"""))

cells.append(md(
"""## Conclusion Helper"""
))

cells.append(code(
"""best_mask = masking_summary.sort_values('avg_of_conditions', ascending=False).iloc[0]
other_mask = masking_summary.sort_values('avg_of_conditions', ascending=False).iloc[1]
delta = best_mask['avg_of_conditions'] - other_mask['avg_of_conditions']
verdict = 'weather-aware' if best_mask['experiment'] == 'mic_weather_aware' else 'generic'
print(f\"Masking verdict: {verdict} MIC is stronger, {best_mask['avg_of_conditions']:.2f}% vs \"
      f\"{other_mask['avg_of_conditions']:.2f}% avg mIoU ({delta:+.2f} pts).\")

best_t = threshold_summary.sort_values('avg_of_conditions', ascending=False).iloc[0]
best_t_night = threshold_summary.sort_values('night_miou', ascending=False).iloc[0]
print(f\"Threshold verdict: {best_t['threshold']} is best overall ({best_t['avg_of_conditions']:.2f}% avg), \"
      f\"{best_t_night['threshold']} is best for night ({best_t_night['night_miou']:.2f}%).\")
if best_t['threshold'] == best_t_night['threshold']:
    print('Same threshold wins both -- no real trade-off in this range.')
else:
    print('Different thresholds win overall vs. night -- there is a real trade-off to choose from.')
"""))

nb["cells"].extend(cells)

OUT.write_text(json.dumps(nb, indent=1))
print(f'Wrote {OUT}')
