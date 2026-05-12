# ----- Cell 1: Imports & helper -----
import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

def parse_yolo_label(label_path):
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = map(float, parts)
                    boxes.append([int(cls), cx, cy, w, h])
    return boxes

def gather_dataset(image_dir: Path, label_dir: Path, disease_label_str):
    """Walk image_dir, match label files, assign image-level label."""
    records = []
    image_exts = ['*.png', '*.jpg', '*.jpeg']
    for ext in image_exts:
        for img_path in image_dir.glob(ext):
            lbl_path = label_dir / (img_path.stem + '.txt')
            boxes = parse_yolo_label(lbl_path)
            has_disease = len(boxes) > 0
            records.append({
                'image_id': img_path.stem,
                'image_path': str(img_path),
                'label_path': str(lbl_path),
                'disease_str': disease_label_str if has_disease else 'normal',
                'source_dataset': disease_label_str   # for stratification
            })
    return pd.DataFrame(records)

# Collect all datasets
df_scd = gather_dataset(Path('/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/scd_dataset/images'), Path('/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/scd_dataset/labels'), 'sickle')
df_malaria = gather_dataset(Path('/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/malaria_dataset/images'), Path('/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/malaria_dataset/labels'), 'malaria')
df_leuk = gather_dataset(Path('/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/leukemia_balanced_subset/images'), Path('/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/leukemia_balanced_subset/labels'), 'leukemia')

# Merge
df_all = pd.concat([df_scd, df_malaria, df_leuk], ignore_index=True)
# Numeric encoding: 0=normal, 1=sickle, 2=malaria, 3=leukemia
mapping = {'normal':0, 'sickle':1, 'malaria':2, 'leukemia':3}
df_all['disease'] = df_all['disease_str'].map(mapping)

print(df_all['disease_str'].value_counts())
df_all.head()
df_all.tail(5)
# ----- Cell 2 (corrected): Stratified split -----
from sklearn.model_selection import train_test_split

# Stratify by disease alone (0,1,2,3) – no need for source_dataset
train, temp = train_test_split(
    df_all,
    test_size=0.3,
    stratify=df_all['disease'],         # <-- changed here
    random_state=42
)
val, test = train_test_split(
    temp,
    test_size=0.5,
    stratify=temp['disease'],           # <-- and here
    random_state=42
)

print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
print("Train class distribution:\n", train['disease_str'].value_counts())

# Leak check remains the same
assert set(train['image_id']).isdisjoint(val['image_id']), "Leak train-val!"
assert set(train['image_id']).isdisjoint(test['image_id']), "Leak train-test!"
assert set(val['image_id']).isdisjoint(test['image_id']), "Leak val-test!"
# ----- Cell 4: Class distribution plot -----
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(12,4))
train['disease_str'].value_counts().plot(kind='bar', ax=axes[0], title='Training Set')
test['disease_str'].value_counts().plot(kind='bar', ax=axes[1], title='Test Set', color = 'orange')
val['disease_str'].value_counts().plot(kind='bar', ax=axes[2], title='Val Set', color = 'green')
plt.tight_layout()
plt.show()
# ----- Cell 5: Bounding box EDA -----
import warnings
warnings.filterwarnings("ignore")
def bbox_statistics(label_dir, disease_name):
    stats = []
    for lbl_file in Path(label_dir).glob('*.txt'):
        boxes = parse_yolo_label(lbl_file)
        for box in boxes:
            _, cx, cy, w, h = box
            stats.append({
                'disease': disease_name,
                'width': w, 'height': h,
                'area': w * h,
                'aspect_ratio': w / h if h > 0 else 0,
                'cx': cx, 'cy': cy
            })
    return pd.DataFrame(stats)

bbox_scd = bbox_statistics('/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/scd_dataset/labels', 'Sickle')
bbox_mal = bbox_statistics('/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/malaria_dataset/labels', 'Malaria')
bbox_leuk = bbox_statistics('/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/leukemia_balanced_subset/labels', 'Leukemia')
bbox_all = pd.concat([bbox_scd, bbox_mal, bbox_leuk], ignore_index=True)

# Box counts per image (positive images only)
plt.figure(figsize=(6,4))
box_counts = bbox_all.groupby('disease').size().reset_index(name='total_boxes')
sns.barplot(data=box_counts, x='disease', y='total_boxes',palette = 'Set2')
plt.title('Total Bounding Boxes per Disease')
plt.show()

# Object area distribution
fig, axes = plt.subplots(1,3, figsize=(12,4))
for ax, (name, grp) in zip(axes, bbox_all.groupby('disease')):
    ax.hist(grp['area'], bins=30, alpha=0.7)
    ax.set_title(f'{name} Object Area (normalised)')
plt.tight_layout()
plt.show()

# Aspect ratio boxplot
plt.figure(figsize=(6,4))
sns.boxplot(data=bbox_all, x='disease', y='aspect_ratio',palette = 'Set2')
plt.ylim(0, 5)
plt.title('Aspect Ratio by Disease')
plt.show()
# ----- Cell 6: Multi‑sample visualization (Sickle Cell) -----
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random

# ---------- CONFIGURE THESE PATHS ----------
SCD_IMG_DIR = '/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/scd_dataset/images'
SCD_LBL_DIR = '/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/scd_dataset/labels'
# --------------------------------------------

def find_all_images(image_dir):
    """Return sorted list of all image files (png, jpg, jpeg)."""
    image_dir = Path(image_dir)
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(sorted(image_dir.glob(ext)))
    return image_files

def is_positive(img_path, label_dir):
    """Check if the image has at least one sickle cell (class 0)."""
    lbl_path = Path(label_dir) / (img_path.stem + '.txt')
    if not lbl_path.exists():
        return False
    boxes = parse_yolo_label(lbl_path)
    return len(boxes) > 0

def draw_boxes_on_axis(ax, img_path, label_dir):
    """Read image, draw boxes, set title."""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    lbl_path = Path(label_dir) / (img_path.stem + '.txt')
    boxes = parse_yolo_label(lbl_path)
    
    ax.imshow(img)
    for box in boxes:
        _, cx, cy, bw, bh = box
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(img_path.stem[:20], fontsize=8)  # show first 20 chars of filename
    ax.axis('off')

# Get all sickle-positive images
all_imgs = find_all_images(SCD_IMG_DIR)
positive_imgs = [p for p in all_imgs if is_positive(p, SCD_LBL_DIR)]

# Take first 8 (or random 8)
num_to_show = min(8, len(positive_imgs))
samples = random.sample(positive_imgs, num_to_show)  # or just positive_imgs[:num_to_show] for deterministic

# Plot grid
cols = 4
rows = (num_to_show + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = axes.flatten()

for idx, img_path in enumerate(samples):
    draw_boxes_on_axis(axes[idx], img_path, SCD_LBL_DIR)

# Hide unused axes
for idx in range(num_to_show, len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Sickle Cell Disease – Positive Samples', fontsize=14)
plt.tight_layout()
plt.show()
# ---------- CONFIGURE THESE PATHS ----------
IMG_DIR = '/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/malaria_dataset/images'
LBL_DIR = '/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/malaria_dataset/labels'
# --------------------------------------------


# Get all malaria-positive images
all_imgs = find_all_images(IMG_DIR)
positive_imgs = [p for p in all_imgs if is_positive(p, LBL_DIR)]

# Take first 8 (or random 8)
num_to_show = min(8, len(positive_imgs))
samples = random.sample(positive_imgs, num_to_show)  # or just positive_imgs[:num_to_show] for deterministic

# Plot grid
cols = 4
rows = (num_to_show + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = axes.flatten()

for idx, img_path in enumerate(samples):
    draw_boxes_on_axis(axes[idx], img_path, LBL_DIR)

# Hide unused axes
for idx in range(num_to_show, len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Malaria Disease – Positive Samples', fontsize=14)
plt.tight_layout()
plt.show()
# ---------- CONFIGURE THESE PATHS ----------
IMG_DIR = '/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/leukemia_balanced_subset/images'
LBL_DIR = '/kaggle/input/datasets/sometimessubodh/multi-disease-dataset/dataset/leukemia_balanced_subset/labels'
# --------------------------------------------


# Get all malaria-positive images
all_imgs = find_all_images(IMG_DIR)
positive_imgs = [p for p in all_imgs if is_positive(p, LBL_DIR)]

# Take first 8 (or random 8)
num_to_show = min(8, len(positive_imgs))
samples = random.sample(positive_imgs, num_to_show)  # or just positive_imgs[:num_to_show] for deterministic

# Plot grid
cols = 4
rows = (num_to_show + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = axes.flatten()

for idx, img_path in enumerate(samples):
    draw_boxes_on_axis(axes[idx], img_path, LBL_DIR)

# Hide unused axes
for idx in range(num_to_show, len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Lukemia – Positive Samples', fontsize=14)
plt.tight_layout()
plt.show()
# ----- Cell 8: Classification dataset -----
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ClsDataset(Dataset):
    def __init__(self, df, transform=None, normalizer=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        #self.normalizer = normalizer      # <-- new

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        img_np = np.array(img)

        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented['image']
        else:
            img_tensor = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0

        label = row['disease']
        return img_tensor, label

# Transforms
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=25, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
    A.ISONoise(p=0.2),
    A.ToGray(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=25, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
    A.ISONoise(p=0.2),
    A.ToGray(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Datasets
train_ds = ClsDataset(train, transform=train_transform)
val_ds = ClsDataset(val, transform=val_transform)

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    return imgs, labels

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)
# Helper for TTA that returns probabilities
@torch.no_grad()
def evaluate_with_tta_probs(model, loader, device, n_views=4):
    model.eval()
    all_probs = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        acc_probs = torch.zeros(imgs.size(0), 4).to(device)
        acc_probs += torch.softmax(model(imgs), dim=1)
        acc_probs += torch.softmax(model(TF.hflip(imgs)), dim=1)
        acc_probs += torch.softmax(model(TF.vflip(imgs)), dim=1)
        acc_probs += torch.softmax(model(TF.rotate(imgs, 90)), dim=1)
        acc_probs /= 4
        all_probs.append(acc_probs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)
# ----- Cell 9: Model & training -----
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.utils.class_weight import compute_class_weight

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=4, backbone='efficientnet_b0', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DiseaseClassifier().to(device)

# Class weights (handling imbalance)
y_train = train['disease'].values
cls_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cls_weights = torch.tensor(cls_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=cls_weights)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
# # ----- Cell 10: Training loop -----
# from sklearn.metrics import accuracy_score
# import os
# import matplotlib.pyplot as plt
from timm.data import Mixup
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

# os.makedirs('models', exist_ok=True)

# Decide whether to use MixUp
use_mixup = True          # toggle this
mixup_fn = Mixup(
    mixup_alpha=0.2,
    cutmix_alpha=1.0,
    label_smoothing=0.1,
    num_classes=4
) if use_mixup else None

# Loss: if mixup, use soft target loss; else standard CE
if use_mixup:
    criterion = SoftTargetCrossEntropy()
else:
    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=np.unique(y_train), y=y_train),
        dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

# # def train_one_epoch(loader):
# #     model.train()
# #     total_loss = 0.0
# #     for imgs, labels in loader:
# #         imgs = imgs.to(device)
# #         labels = labels.to(device)
# #         if mixup_fn is not None:
# #             imgs, labels = mixup_fn(imgs, labels)
# #         optimizer.zero_grad()
# #         outputs = model(imgs)
# #         loss = criterion(outputs, labels)
# #         loss.backward()
# #         optimizer.step()
# #         total_loss += loss.item() * imgs.size(0)
# #     return total_loss / len(loader.dataset)

# # @torch.no_grad()
# # def evaluate(loader):
# #     model.eval()
# #     total_loss = 0.0
# #     all_preds, all_labels = [], []
# #     for imgs, labels in loader:
# #         imgs = imgs.to(device)
# #         labels = labels.to(device)
# #         outputs = model(imgs)
# #         # Use standard CE loss for evaluation (we don't mix validation)
# #         loss = torch.nn.functional.cross_entropy(outputs, labels)
# #         total_loss += loss.item() * imgs.size(0)
# #         pred_class = outputs.argmax(1)
# #         all_preds.extend(pred_class.cpu().numpy())
# #         all_labels.extend(labels.cpu().numpy())
# #     return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds), all_preds, all_labels

# # # Store metrics
# # history = {
# #     'train_loss': [], 'train_acc': [],
# #     'val_loss': [], 'val_acc': []
# # }

# # best_acc = 0.0
# # for epoch in range(1, 11):
# #     train_loss = train_one_epoch(train_loader)
# #     val_loss, val_acc, _, _ = evaluate(val_loader)
# #     scheduler.step(val_acc)
    
# #     # Save to history
# #     history['train_loss'].append(train_loss)
# #     #history['train_acc'].append(train_acc)
# #     history['val_loss'].append(val_loss)
# #     history['val_acc'].append(val_acc)
    
# #     #print(f"Epoch {epoch:02d}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
# #     print(f"Epoch {epoch:02d}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
    
# #     if val_acc > best_acc:
# #         best_acc = val_acc
# #         torch.save(model.state_dict(), 'models/classifier_best.pth')

# # print(f"Best Val Acc: {best_acc:.4f}")
# # ----- Cell 11: Test evaluation with TTA -----
# import torch
# import torchvision.transforms.functional as TF
# from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # ----- TTA evaluation function -----
# @torch.no_grad()
# def evaluate_with_tta_probs(model, loader, device, n_views=4):
#     model.eval()
#     all_probs = []
#     all_labels = []
#     for imgs, labels in loader:
#         imgs = imgs.to(device)
#         acc_probs = torch.zeros(imgs.size(0), 4).to(device)
#         acc_probs += torch.softmax(model(imgs), dim=1)
#         acc_probs += torch.softmax(model(TF.hflip(imgs)), dim=1)
#         acc_probs += torch.softmax(model(TF.vflip(imgs)), dim=1)
#         acc_probs += torch.softmax(model(TF.rotate(imgs, 90)), dim=1)
#         acc_probs /= 4
#         all_probs.append(acc_probs.cpu().numpy())
#         all_labels.append(labels.numpy())
#     return np.concatenate(all_probs), np.concatenate(all_labels)

# # ----- Load test dataset -----
# test_ds = ClsDataset(test, transform=val_transform)
# test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

# # Load best model
# model.load_state_dict(torch.load('models/classifier_best.pth', map_location=device))
# model.eval()

# # ---- ✨ Use TTA here ✨ ----
# #test_acc, test_preds, test_labels = evaluate_with_tta(model, test_loader, device
# test_probs, test_labels = evaluate_with_tta_probs(model, test_loader, device)

# # ---- Metrics ----
# labels_names = ['Normal', 'Sickle', 'Malaria', 'Leukemia']
# print(f"Test Accuracy (TTA): {test_acc:.4f}")
# print(classification_report(test_labels, test_preds, target_names=labels_names))

# # Confusion matrix
# cm = confusion_matrix(test_labels, test_preds)
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels_names, yticklabels=labels_names, cmap='viridis')
# plt.title('Confusion Matrix (with TTA)')
# plt.show()

# # Per-class sensitivity/specificity
# mcm = multilabel_confusion_matrix(test_labels, test_preds)
# for i, matrix in enumerate(mcm):
#     tn, fp, fn, tp = matrix.ravel()
#     sens = tp / (tp + fn) if (tp + fn) > 0 else 0
#     spec = tn / (tn + fp) if (tn + fp) > 0 else 0
#     print(f"{labels_names[i]}:   Sensitivity={sens:.3f}, Specificity={spec:.3f}")
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import label_binarize

# # Binarize labels for multi-class ROC
# test_labels_bin = label_binarize(test_labels, classes=[0,1,2,3])
# n_classes = 4

# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_probs[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# plt.figure(figsize=(8,6))
# colors = ['blue', 'red', 'green', 'orange']
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label=f'{labels_names[i]} (AUC = {roc_auc[i]:.3f})')
# plt.plot([0,1], [0,1], 'k--', lw=1)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Multi-class ROC (One-vs-Rest)')
# plt.legend(loc="lower right")
# plt.show()
# ----- Cell 10: K-Fold Cross-Validation -----
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.metrics import multilabel_confusion_matrix   

# ----- Configuration -----
k_folds = 4
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model_path = 'models/classifier_best_cv.pth'   # final model after CV

# Combine train and val for cross-validation
cv_df = pd.concat([train, val], ignore_index=True)
cv_labels = cv_df['disease'].values
cv_dataset = ClsDataset(cv_df, transform=train_transform)  # we'll create train/val subsets per fold

# KFold
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Storage for per-fold metrics
fold_train_losses = []
fold_val_losses = []
fold_val_accs = []
fold_val_aucs = []
fold_sensitivities = []
fold_specificities = []
all_val_labels = []
all_val_preds = []
all_val_probs = []

# For ROC plotting (across folds)
labels_names = ['Normal', 'Sickle', 'Malaria', 'Leukemia']
n_classes = 4

for fold, (train_idx, val_idx) in enumerate(skf.split(cv_df, cv_labels)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold+1}/{k_folds}")
    print(f"{'='*50}")
    
    # Split data
    train_fold_df = cv_df.iloc[train_idx].reset_index(drop=True)
    val_fold_df = cv_df.iloc[val_idx].reset_index(drop=True)
    
    train_fold_ds = ClsDataset(train_fold_df, transform=train_transform)
    val_fold_ds = ClsDataset(val_fold_df, transform=val_transform)
    
    train_loader = DataLoader(train_fold_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4,drop_last=True)
    val_loader = DataLoader(val_fold_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Fresh model
    model = DiseaseClassifier(num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Class weights for this fold
    y_fold = train_fold_df['disease'].values
    cls_weights = compute_class_weight('balanced', classes=np.unique(y_fold), y=y_fold)
    cls_weights = torch.tensor(cls_weights, dtype=torch.float).to(device)
    
    # MixUp and criterion
    use_mixup = True
    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=4)
    train_criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss(weight=cls_weights)
    
    best_val_acc = 0.0
    best_model_state = None
    fold_train_loss = []
    fold_val_loss = []
    
    for epoch in range(1, epochs+1):
        # Training
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, labels = mixup_fn(imgs, labels)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = train_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)
        fold_train_loss.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        all_preds_list = []
        all_labels_list = []
        all_probs_list = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = val_criterion(outputs, labels)
                total_val_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds_list.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())
                all_probs_list.append(probs.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels_list, all_preds_list)
        fold_val_loss.append(avg_val_loss)
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_preds = all_preds_list.copy()
            best_val_labels = all_labels_list.copy()
            best_val_probs = np.concatenate(all_probs_list)
    
    # Load best model and evaluate final fold metrics
    model.load_state_dict(best_model_state)
    fold_val_accs.append(best_val_acc)
    fold_train_losses.append(fold_train_loss)
    fold_val_losses.append(fold_val_loss)
    
    # Compute AUC for this fold
    val_labels_bin = label_binarize(best_val_labels, classes=[0,1,2,3])
    try:
        auc_score = roc_auc_score(val_labels_bin, best_val_probs, average='macro', multi_class='ovr')
    except:
        auc_score = 0.0
    fold_val_aucs.append(auc_score)
    
    # Compute sensitivity/specificity per class
    sens = []
    spec = []
    mcm = multilabel_confusion_matrix(best_val_labels, best_val_preds)
    for i in range(n_classes):
        tn, fp, fn, tp = mcm[i].ravel()
        sens.append(tp/(tp+fn) if (tp+fn)>0 else 0)
        spec.append(tn/(tn+fp) if (tn+fp)>0 else 0)
    fold_sensitivities.append(sens)
    fold_specificities.append(spec)
    
    # Append for aggregated confusion matrix
    all_val_labels.extend(best_val_labels)
    all_val_preds.extend(best_val_preds)
    all_val_probs.append(best_val_probs)
    
    print(f"Fold {fold+1} – Best Val Acc: {best_val_acc:.4f} (epoch {best_epoch}), AUC: {auc_score:.4f}")
# ----- Cell 10.5: CV Summary -----
# Convert per-fold sensitivities/specificities to arrays
fold_sens = np.array(fold_sensitivities)   # shape (k_folds, n_classes)
fold_spec = np.array(fold_specificities)

print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS (MEAN ± STD)")
print("="*60)
print(f"Validation Accuracy : {np.mean(fold_val_accs):.4f} ± {np.std(fold_val_accs):.4f}")
print(f"Macro AUC           : {np.mean(fold_val_aucs):.4f} ± {np.std(fold_val_aucs):.4f}")

for i, name in enumerate(labels_names):
    print(f"{name:10s} – Sensitivity: {np.mean(fold_sens[:, i]):.3f} ± {np.std(fold_sens[:, i]):.3f}  "
          f"Specificity: {np.mean(fold_spec[:, i]):.3f} ± {np.std(fold_spec[:, i]):.3f}")

# Aggregated confusion matrix (all validation samples from all folds)
cm_agg = confusion_matrix(all_val_labels, all_val_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm_agg, annot=True, fmt='d', xticklabels=labels_names, yticklabels=labels_names, cmap='viridis')
plt.title('Aggregated Confusion Matrix (Cross-Validation)')
plt.show()
import os

os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

# ----- Cell 11: Final Model (train on all train+val) and Test Evaluation -----
import torchvision.transforms.functional as TF
print("\nTraining final model on combined train+val set...")

# Combine train+val again (already have cv_df)
full_train_ds = ClsDataset(cv_df, transform=train_transform)
full_train_loader = DataLoader(full_train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4,drop_last=True)

# One more fresh model
final_model = DiseaseClassifier(num_classes=4).to(device)
optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Use all data class weights
y_all = cv_df['disease'].values
cls_weights = compute_class_weight('balanced', classes=np.unique(y_all), y=y_all)
cls_weights = torch.tensor(cls_weights, dtype=torch.float).to(device)
criterion = SoftTargetCrossEntropy()
mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=4)

best_loss = float('inf')
best_state = None
for epoch in range(1, epochs+1):
    final_model.train()
    total_loss = 0.0
    for imgs, labels in full_train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, labels = mixup_fn(imgs, labels)
        optimizer.zero_grad()
        outputs = final_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    avg_loss = total_loss / len(full_train_loader.dataset)
    scheduler.step(avg_loss)  # using loss for scheduling (no val loss available)
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_state = deepcopy(final_model.state_dict())
    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} – Train Loss: {avg_loss:.4f}")

final_model.load_state_dict(best_state)
torch.save(final_model.state_dict(), best_model_path)
print(f"Final model saved to {best_model_path}")

# ----- Test Evaluation with TTA -----
test_ds = ClsDataset(test, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

@torch.no_grad()
def evaluate_with_tta_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        acc_probs = torch.zeros(imgs.size(0), 4).to(device)
        acc_probs += torch.softmax(model(imgs), dim=1)
        acc_probs += torch.softmax(model(TF.hflip(imgs)), dim=1)
        acc_probs += torch.softmax(model(TF.vflip(imgs)), dim=1)
        acc_probs += torch.softmax(model(TF.rotate(imgs, 90)), dim=1)
        acc_probs /= 4
        all_probs.append(acc_probs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)

test_probs, test_labels = evaluate_with_tta_probs(final_model, test_loader, device)
test_preds = np.argmax(test_probs, axis=1)
test_acc = accuracy_score(test_labels, test_preds)

print(f"\nTest Accuracy (TTA): {test_acc:.4f}")
print(classification_report(test_labels, test_preds, target_names=labels_names))

# Confusion matrix
cm_test = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm_test, annot=True, fmt='d', xticklabels=labels_names, yticklabels=labels_names, cmap='viridis')
plt.title('Test Confusion Matrix (Final Model)')
plt.show()

# Per-class sensitivity/specificity on test
mcm_test = multilabel_confusion_matrix(test_labels, test_preds)
for i, matrix in enumerate(mcm_test):
    tn, fp, fn, tp = matrix.ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    print(f"{labels_names[i]}: Sensitivity={sens:.3f}, Specificity={spec:.3f}")

# Test ROC curves
test_labels_bin = label_binarize(test_labels, classes=[0,1,2,3])
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
colors = ['blue', 'red', 'green', 'orange']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{labels_names[i]} (AUC = {roc_auc[i]:.3f})')
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC Curves (Final Model)')
plt.legend(loc="lower right")
plt.show()
if 'final_model' in globals():
    torch.save(final_model.state_dict(), best_model_path)
else:
    print("final_model not found — run training cell first.")
pip install grad-cam
# ----- Cell 11.5: Grad‑CAM Grid for Multiple Classes -----
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Set device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
final_model.eval()

# Choose the last convolutional layer for EfficientNet‑B0
target_layer = final_model.backbone.conv_head
cam = GradCAM(model=final_model, target_layers=[target_layer])

# Transform for single image (same as val_transform, but we need a function)
def prepare_image(img_path, size=(224,224)):
    """Load and preprocess an image for the classifier."""
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size)
    img_norm = val_transform(image=img_resized)['image'].unsqueeze(0).to(device)
    return img_resized, img_norm   # original (for overlay), tensor

# Select up to 2 examples per disease from the test set (or any)
samples_to_show = []
disease_names = ['normal', 'sickle', 'malaria', 'leukemia']
for disease in disease_names:
    subset = test[test['disease_str'] == disease]
    if len(subset) >= 2:
        chosen = subset.sample(2, random_state=42)  # use fixed seed for reproducibility
    else:
        chosen = subset  # take all if less than 2
    for _, row in chosen.iterrows():
        samples_to_show.append((row['image_path'], disease))

# Create figures: 2 rows x 4 columns = 8 images
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (img_path, true_label) in enumerate(samples_to_show):
    original_img, input_tensor = prepare_image(img_path)
    
    # Generate Grad‑CAM heatmap for the predicted class (or you can enforce the true class)
    # To see what the model thinks is important for its prediction:
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    
    # Overlay heatmap on original (resized) image
    visualization = show_cam_on_image(original_img.astype(np.float32)/255.0,
                                      grayscale_cam,
                                      use_rgb=True)
    
    axes[idx].imshow(visualization)
    axes[idx].set_title(f'True: {true_label}', fontsize=10)
    axes[idx].axis('off')

# Hide unused subplots
for idx in range(len(samples_to_show), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Grad‑CAM Visualizations Across All Classes', fontsize=14, weight='bold')
plt.tight_layout()
plt.show()