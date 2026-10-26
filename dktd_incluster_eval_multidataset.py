"""Standalone DKTD original-cluster evaluation on three image datasets.

Train one seed:
    python dktd_incluster_eval_multidataset.py --dataset mnist --seed 4
Train and summarize ten seeds:
    python dktd_incluster_eval_multidataset.py --dataset fashion_mnist --seeds 0-9
Summarize existing completed runs without training:
    python dktd_incluster_eval_multidataset.py --aggregate --result-root results/dktd_incluster_multidataset

Only DKTD is implemented. Before/after describe the same distillation update,
not two independent no-KD/KD training configurations. No holdout score affects
training, teacher selection, rollback, or checkpoint choice.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

PROTOCOL_ID = "dktd_original_cluster_multidataset_v1"
DATASET_KEY = "cifar10"
DATASET_SPECS = {
    "cifar10": {
        "name": "CIFAR-10", "torchvision_class": "CIFAR10",
        "model_class": "ResNet18Classifier", "input_shape": [3, 32, 32],
        "feature_dim": 512, "train_samples": 50000,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "feature_hist_range": [-3.0, 3.0],
        "model": "torchvision ResNet-18 backbone plus new ten-class linear classifier",
        "model_configuration_source": "existing CIFAR-10 DKTD evaluation protocol",
        "feature_extractor": "initial shared ResNet-18 backbone, evaluation mode, 512-dimensional output",
        "train_transform": "RandomCrop(32,padding=4), RandomHorizontalFlip, ToTensor, ImageNet normalization",
        "eval_transform": "ToTensor, ImageNet normalization; no random augmentation",
    },
    "mnist": {
        "name": "MNIST", "torchvision_class": "MNIST",
        "model_class": "CNNMNIST", "input_shape": [1, 28, 28],
        "feature_dim": 50, "train_samples": 60000,
        "mean": [0.1307], "std": [0.3081], "feature_hist_range": [0.0, 10.0],
        "model": "CNNMNIST: Conv1-10-20(k5), pool2 twice, Dropout2d, FC320-50-10, training dropout",
        "model_configuration_source": "author-confirmed existing CNNMNIST architecture; output adapted to logits",
        "feature_extractor": "initial shared CNNMNIST encode: fc1 ReLU, evaluation mode, 50-dimensional output",
        "train_transform": "ToTensor, Normalize(mean=0.1307,std=0.3081); single channel 28x28; no random augmentation",
        "eval_transform": "ToTensor, Normalize(mean=0.1307,std=0.3081); single channel 28x28; no random augmentation",
    },
    "fashion_mnist": {
        "name": "Fashion-MNIST", "torchvision_class": "FashionMNIST",
        "model_class": "SimpleCNNClassifier", "input_shape": [1, 28, 28],
        "feature_dim": 128, "train_samples": 60000,
        "mean": [0.2860], "std": [0.3530], "feature_hist_range": [0.0, 10.0],
        "model": "SimpleCNNClassifier: Conv1-32-64(k5,p2), pool2 twice, FC3136-128-10",
        "model_configuration_source": "existing Fashion-MNIST DKTD CNN and preprocessing",
        "feature_extractor": "initial shared SimpleCNNClassifier.encode, evaluation mode, 128-dimensional output",
        "train_transform": "ToTensor, Normalize(mean=0.2860,std=0.3530); single channel 28x28; no random augmentation",
        "eval_transform": "ToTensor, Normalize(mean=0.2860,std=0.3530); single channel 28x28; no random augmentation",
    },
}
NUM_CLIENTS, NUM_CLASSES, NUM_CLUSTERS = 50, 10, 5
LOCAL_EPOCHS, BATCH_SIZE, LR = 1, 32, 0.01
TEACHER_M, DISTILL_INTERVAL, DISTILL_EPOCHS = 3, 5, 1
DISTILL_TAU, KD_LAMBDA, DISTILL_LR = 4.0, 0.7, 0.01
LAMBDA_LABEL, LAMBDA_FEAT, PROTO_HIST_ALPHA = 0.5, 0.5, 0.5
FEATURE_HIST_BINS, FEATURE_HIST_RANGE, ENCODER_BATCH_SIZE = 16, (-3.0, 3.0), 256
DKTD_RHO, DKTD_ALPHA, DKTD_BETA_D, DKTD_GAMMA = 5.0, 1.0, 0.5, 0.8
VAL_FRACTION, CLUSTER_HOLDOUT_FRACTION = 0.10, 0.10
SEED, NUM_WORKERS = 0, 2
PRETRAINED_LOAD_STATUS = "not_requested"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train, transform_test = None, None

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 为了速度，这里不强制 deterministic。多 seed 正式实验时可根据需要改为 True/False。
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def normalize_dataset_batch(batch):
    """Deterministic raw-image preprocessing matching the selected eval transform.

    CIFAR-10 stores numpy NHWC bytes. Both grayscale datasets store torch NHW
    bytes. Prototype extraction does not apply random training augmentation.
    """
    spec = DATASET_SPECS[DATASET_KEY]
    if isinstance(batch, np.ndarray):
        batch = torch.from_numpy(batch)
    if not torch.is_tensor(batch):
        raise TypeError("raw dataset images must be a numpy array or torch tensor")
    batch = batch.float() / 255.0
    if DATASET_KEY == "cifar10":
        if batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError("CIFAR-10 raw images must have shape [N,32,32,3]")
        batch = batch.permute(0, 3, 1, 2)
    elif batch.ndim == 3:
        batch = batch.unsqueeze(1)
    elif batch.ndim == 4 and batch.shape[-1] == 1:
        batch = batch.permute(0, 3, 1, 2)
    if list(batch.shape[1:]) != spec["input_shape"]:
        raise ValueError("raw input shape does not match the selected dataset/model")
    mean = batch.new_tensor(spec["mean"]).view(1, -1, 1, 1)
    std = batch.new_tensor(spec["std"]).view(1, -1, 1, 1)
    return (batch - mean) / std


def canonical_dataset_key(value):
    token = str(value).strip().lower().replace("-", "").replace("_", "")
    aliases = {"cifar10": "cifar10", "mnist": "mnist", "fashionmnist": "fashion_mnist"}
    if token not in aliases:
        raise argparse.ArgumentTypeError("dataset must be cifar10, mnist, or fashion_mnist")
    return aliases[token]


def build_dataset_transforms(dataset_key):
    spec = DATASET_SPECS[dataset_key]
    eval_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(tuple(spec["mean"]), tuple(spec["std"])),
    ])
    if dataset_key == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(tuple(spec["mean"]), tuple(spec["std"])),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(tuple(spec["mean"]), tuple(spec["std"])),
        ])
    return train_transform, eval_transform

def dirichlet_split_noniid(labels, num_clients, num_classes, beta, seed=42):
    """
    使用 Dirichlet 分布将数据划分为 Non-IID。
    同一 seed 下，所有 strategy 使用完全相同的数据划分。
    """
    rng = np.random.default_rng(seed)

    class_indices = [np.where(labels == cls)[0] for cls in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        idx_cls = class_indices[cls].copy()
        rng.shuffle(idx_cls)

        proportions = rng.dirichlet(np.repeat(beta, num_clients))
        split_points = (np.cumsum(proportions) * len(idx_cls)).astype(int)[:-1]
        split_indices = np.split(idx_cls, split_points)

        for client_id, split in enumerate(split_indices):
            client_indices[client_id].extend(split.tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices

def split_client_indices_for_validation_and_holdout(
    client_indices,
    labels,
    seed,
    validation_fraction=VAL_FRACTION,
    holdout_fraction=CLUSTER_HOLDOUT_FRACTION,
):
    """按客户端、按类别拆分索引；验证集和簇内留出集不参与训练。"""
    if validation_fraction < 0 or holdout_fraction < 0:
        raise ValueError("validation/holdout fraction must be non-negative")
    if validation_fraction + holdout_fraction >= 0.5:
        raise ValueError("validation and holdout fractions must sum to less than 0.5")

    train_indices, validation_indices, holdout_indices = [], [], []
    for client_id, indices in enumerate(client_indices):
        client_train, client_val, client_holdout = [], [], []
        rng = np.random.default_rng(seed + 7919 * (client_id + 1))
        indices = np.asarray(indices, dtype=np.int64)

        for class_id in range(NUM_CLASSES):
            class_indices = indices[labels[indices] == class_id].copy()
            rng.shuffle(class_indices)
            n = len(class_indices)
            n_val = int(round(n * validation_fraction))
            n_holdout = int(round(n * holdout_fraction))

            # Small client/class cells still reserve held-out examples where
            # possible, while always retaining at least one training example.
            if n >= 3 and validation_fraction > 0:
                n_val = max(1, n_val)
            if n >= 3 and holdout_fraction > 0:
                n_holdout = max(1, n_holdout)
            while n_val + n_holdout >= n and (n_val > 0 or n_holdout > 0):
                if n_val >= n_holdout and n_val > 0:
                    n_val -= 1
                elif n_holdout > 0:
                    n_holdout -= 1

            client_val.extend(class_indices[:n_val].tolist())
            client_holdout.extend(
                class_indices[n_val:n_val + n_holdout].tolist()
            )
            client_train.extend(class_indices[n_val + n_holdout:].tolist())

        rng.shuffle(client_train)
        rng.shuffle(client_val)
        rng.shuffle(client_holdout)
        train_indices.append(client_train)
        validation_indices.append(client_val)
        holdout_indices.append(client_holdout)

    return train_indices, validation_indices, holdout_indices

def build_resnet18_backbone(use_pretrained=True):
    global PRETRAINED_LOAD_STATUS
    weights = None

    if use_pretrained:
        try:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        except AttributeError:
            weights = None

    try:
        backbone = torchvision.models.resnet18(weights=weights)
        if use_pretrained and weights is not None:
            PRETRAINED_LOAD_STATUS = "loaded"
            print("使用预训练 ResNet18 作为共享 encoder")
        elif use_pretrained:
            PRETRAINED_LOAD_STATUS = "not_available_in_torchvision"
        return backbone
    except Exception as exc:
        print(f"加载预训练 ResNet18 失败，回退到随机初始化: {exc}")
        if use_pretrained:
            PRETRAINED_LOAD_STATUS = f"fallback_random: {type(exc).__name__}: {exc}"
        return torchvision.models.resnet18(weights=None)

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10, use_pretrained=True):
        super().__init__()

        backbone = build_resnet18_backbone(use_pretrained=use_pretrained)
        in_features = backbone.fc.in_features
        self.feature_dim = in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def encode(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.encode(x)
        return self.classifier(features)


class CNNMNIST(nn.Module):
    """Author-confirmed MNIST network; expose fc1 features and raw logits.

    Convolutions, pooling and both training dropout stages match the existing
    network. Its final log_softmax is omitted because CE/KD consume logits.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_dim = 50
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, self.feature_dim)
        self.fc2 = nn.Linear(self.feature_dim, num_classes)

    def encode(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        x = self.encode(x)
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


class SimpleCNNClassifier(nn.Module):
    """Fashion-MNIST classifier and 128-dimensional encoder from its DKTD runner."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_dim = 128
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(64 * 7 * 7, self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def encode(self, x):
        return F.relu(self.fc1(torch.flatten(self.conv(x), 1)))

    def forward(self, x):
        return self.classifier(self.encode(x))


def build_dataset_model(use_pretrained=False):
    if DATASET_KEY == "cifar10":
        return ResNet18Classifier(NUM_CLASSES, use_pretrained=use_pretrained)
    if use_pretrained:
        raise ValueError("ImageNet pretrained weights apply only to the CIFAR-10 ResNet-18")
    if DATASET_KEY == "mnist":
        return CNNMNIST(NUM_CLASSES)
    return SimpleCNNClassifier(NUM_CLASSES)

def compute_prototype_histogram(prototype, num_bins, value_range):
    hist, _ = np.histogram(prototype, bins=num_bins, range=value_range)
    hist = hist.astype(np.float32)

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist

def extract_client_statistics(
    client_indices,
    dataset_data,
    labels,
    num_classes,
    encoder,
    batch_size,
):
    """
    每个客户端本地提取：
    1. 标签分布；
    2. 类别原型；
    3. 原型直方图。
    """
    encoder.eval()
    encoder_device = next(encoder.parameters()).device
    feature_dim = int(encoder.feature_dim)

    client_label_histograms = []
    client_prototypes = []
    client_feature_histograms = []

    with torch.no_grad():
        for indices in client_indices:
            label_hist = np.zeros(num_classes, dtype=np.float32)
            prototype_sums = None
            prototype_counts = np.zeros(num_classes, dtype=np.int64)

            if len(indices) == 0:
                client_label_histograms.append(label_hist)
                client_prototypes.append(np.zeros((num_classes, feature_dim), dtype=np.float32))
                client_feature_histograms.append(
                    np.zeros((num_classes, FEATURE_HIST_BINS), dtype=np.float32)
                )
                continue

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_images = dataset_data[batch_indices]
                batch_labels = labels[batch_indices]

                batch_images = normalize_dataset_batch(batch_images).to(encoder_device)
                features = encoder.encode(batch_images).cpu().numpy()
                if features.ndim != 2 or features.shape[1] != feature_dim:
                    raise RuntimeError("encoder output does not match its declared feature dimension")

                if prototype_sums is None:
                    prototype_sums = np.zeros(
                        (num_classes, features.shape[1]), dtype=np.float32
                    )

                for feat, label in zip(features, batch_labels):
                    label_hist[label] += 1
                    prototype_sums[label] += feat
                    prototype_counts[label] += 1

            if label_hist.sum() > 0:
                label_hist /= label_hist.sum()

            prototypes = np.zeros((num_classes, feature_dim), dtype=np.float32)
            feature_hist = np.zeros((num_classes, FEATURE_HIST_BINS), dtype=np.float32)

            if prototype_sums is not None:
                for cls in range(num_classes):
                    if prototype_counts[cls] == 0:
                        continue

                    prototype = prototype_sums[cls] / prototype_counts[cls]
                    prototypes[cls] = prototype

                    feature_hist[cls] = compute_prototype_histogram(
                        prototype, FEATURE_HIST_BINS, FEATURE_HIST_RANGE
                    )

            client_label_histograms.append(label_hist)
            client_prototypes.append(prototypes)
            client_feature_histograms.append(feature_hist)

    return (
        np.array(client_label_histograms),
        np.array(client_prototypes),
        np.array(client_feature_histograms),
    )

def js_distance(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if p.sum() <= eps and q.sum() <= eps:
        return 0.0

    p = p + eps
    q = q + eps

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))

def compute_prototype_distance(prototypes_i, prototypes_j, label_hist_i, label_hist_j):
    class_weights = 0.5 * (label_hist_i + label_hist_j)

    if class_weights.sum() == 0:
        return 0.0

    distance = 0.0

    for cls in range(len(class_weights)):
        if class_weights[cls] == 0:
            continue

        proto_i = prototypes_i[cls]
        proto_j = prototypes_j[cls]

        norm_i = np.linalg.norm(proto_i)
        norm_j = np.linalg.norm(proto_j)

        if norm_i == 0.0 or norm_j == 0.0:
            continue

        cosine_sim = float(np.dot(proto_i, proto_j) / (norm_i * norm_j))
        distance += class_weights[cls] * (1.0 - cosine_sim)

    return float(distance / (class_weights.sum() + 1e-12))

def compute_histogram_distance(feature_hist_i, feature_hist_j, label_hist_i, label_hist_j):
    class_weights = 0.5 * (label_hist_i + label_hist_j)

    if class_weights.sum() == 0:
        return 0.0

    distance = 0.0

    for cls in range(len(class_weights)):
        if class_weights[cls] == 0:
            continue

        distance += class_weights[cls] * js_distance(
            feature_hist_i[cls], feature_hist_j[cls]
        )

    return float(distance / (class_weights.sum() + 1e-12))

def compute_feature_distance(
    prototypes_i,
    prototypes_j,
    feature_hist_i,
    feature_hist_j,
    label_hist_i,
    label_hist_j,
    alpha,
):
    d_proto = compute_prototype_distance(
        prototypes_i, prototypes_j, label_hist_i, label_hist_j
    )

    d_hist = compute_histogram_distance(
        feature_hist_i, feature_hist_j, label_hist_i, label_hist_j
    )

    return alpha * d_proto + (1.0 - alpha) * d_hist

def build_combined_distance_matrix(
    label_histograms,
    client_prototypes,
    feature_histograms,
    lambda_label,
    lambda_feat,
    alpha,
):
    num_clients = label_histograms.shape[0]
    distance_matrix = np.zeros((num_clients, num_clients), dtype=np.float32)

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            d_label = js_distance(label_histograms[i], label_histograms[j])

            d_feat = compute_feature_distance(
                client_prototypes[i],
                client_prototypes[j],
                feature_histograms[i],
                feature_histograms[j],
                label_histograms[i],
                label_histograms[j],
                alpha,
            )

            distance = lambda_label * d_label + lambda_feat * d_feat

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

def kmedoids_precomputed(distance_matrix, num_clusters, random_state=42, max_iter=100):
    num_samples = distance_matrix.shape[0]

    if num_clusters > num_samples:
        raise ValueError("num_clusters 不能大于样本数量")

    rng = np.random.default_rng(random_state)
    medoid_indices = rng.choice(num_samples, size=num_clusters, replace=False)

    cluster_labels = None

    for _ in range(max_iter):
        medoid_distances = distance_matrix[:, medoid_indices]
        new_cluster_labels = np.argmin(medoid_distances, axis=1)

        new_medoid_indices = medoid_indices.copy()

        for cluster_id in range(num_clusters):
            cluster_members = np.where(new_cluster_labels == cluster_id)[0]

            if len(cluster_members) == 0:
                candidate_pool = np.setdiff1d(
                    np.arange(num_samples), new_medoid_indices, assume_unique=False
                )

                if len(candidate_pool) > 0:
                    new_medoid_indices[cluster_id] = rng.choice(candidate_pool)

                continue

            intra_distances = distance_matrix[np.ix_(cluster_members, cluster_members)]
            total_distances = intra_distances.sum(axis=1)
            best_member = cluster_members[np.argmin(total_distances)]
            new_medoid_indices[cluster_id] = best_member

        if cluster_labels is not None and np.array_equal(new_medoid_indices, medoid_indices):
            cluster_labels = new_cluster_labels
            break

        medoid_indices = new_medoid_indices
        cluster_labels = new_cluster_labels

    else:
        medoid_distances = distance_matrix[:, medoid_indices]
        cluster_labels = np.argmin(medoid_distances, axis=1)

    return cluster_labels, medoid_indices

def fedavg_weighted_state(base_state, client_states, client_weights):
    """
    加权 FedAvg。
    BatchNorm 的 num_batches_tracked 不参与平均，直接复用权重最大客户端的值。
    """
    new_state = deepcopy(base_state)

    if len(client_states) == 0:
        return new_state

    reference_idx = int(np.argmax(client_weights)) if len(client_weights) > 0 else 0

    for key in new_state.keys():
        if "num_batches_tracked" in key:
            new_state[key] = client_states[reference_idx][key].clone()
            continue

        dtype = new_state[key].dtype

        if dtype not in [torch.float16, torch.float32, torch.float64]:
            weighted_sum = torch.zeros_like(new_state[key], dtype=torch.float32)
            total_weight = 0.0

            for idx, state in enumerate(client_states):
                weighted_sum += client_weights[idx] * state[key].float()
                total_weight += client_weights[idx]

            if total_weight > 0:
                new_state[key] = (weighted_sum / total_weight).to(dtype)

        else:
            weighted_sum = torch.zeros_like(new_state[key])
            total_weight = 0.0

            for idx, state in enumerate(client_states):
                weighted_sum += client_weights[idx] * state[key]
                total_weight += client_weights[idx]

            if total_weight > 0:
                new_state[key] = weighted_sum / total_weight

    return new_state

def evaluate_metrics(model, dataloader, num_classes=10):
    if dataloader is None or len(dataloader) == 0:
        return {
            "acc": float("nan"),
            "loss": float("nan"),
            "macro_f1": float("nan"),
            "per_class_recall": np.full(num_classes, np.nan, dtype=np.float64),
        }

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()

            y_np = y.cpu().numpy()
            p_np = preds.cpu().numpy()

            for true_label, pred_label in zip(y_np, p_np):
                confusion[true_label, pred_label] += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    tp = np.diag(confusion)
    pred_sum = confusion.sum(axis=0)
    true_sum = confusion.sum(axis=1)

    precision = tp / (pred_sum + 1e-12)
    recall = tp / (true_sum + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    macro_f1 = 100.0 * np.mean(f1)
    per_class_recall = 100.0 * recall

    return {
        "acc": float(acc),
        "loss": float(avg_loss),
        "macro_f1": float(macro_f1),
        "per_class_recall": per_class_recall,
    }

def softmax_np(scores, temperature=1.0):
    scores = np.asarray(scores, dtype=np.float64) / temperature
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / (np.sum(exp_scores) + 1e-12)

def minmax_normalize_values(values):
    values = np.asarray(values, dtype=np.float64)
    v_min = np.min(values)
    v_max = np.max(values)
    return (values - v_min) / (v_max - v_min + 1e-12)

def compute_complementarity(target_q, teacher_q, rho=DKTD_RHO):
    """
    Comp(t,r)=sum_c u_{t,c} q_{r,c}
    u_{t,c}=exp(-rho*q_{t,c})
    """
    demand = np.exp(-rho * target_q)
    return float(np.sum(demand * teacher_q))

def compute_risk(target_q, teacher_q, target_proto, teacher_proto, rho=DKTD_RHO):
    """
    Risk(t,r): 共享类别上的原型距离。
    对目标簇弱势类别赋予更高权重。
    """
    shared_classes = [
        cls for cls in range(len(target_q))
        if target_q[cls] > 0 and teacher_q[cls] > 0
    ]

    if len(shared_classes) == 0:
        return 0.0

    demand = np.exp(-rho * target_q)
    shared_demand = np.array([demand[cls] for cls in shared_classes], dtype=np.float64)
    shared_demand = shared_demand / (shared_demand.sum() + 1e-12)

    risk = 0.0

    for weight, cls in zip(shared_demand, shared_classes):
        dist = np.linalg.norm(target_proto[cls] - teacher_proto[cls])
        risk += weight * dist

    return float(risk)

def compute_confidence(model, dataloader):
    """
    Rel(r): 教师可靠性。
    这里用教师模型在其簇独立验证集上的平均最大预测置信度近似；
    调用方必须传入无随机增强、无训练样本的 validation loader。
    """
    if dataloader is None or len(dataloader) == 0:
        return 0.0

    model.eval()
    total_conf = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)

            total_conf += max_probs.sum().item()
            total_samples += x.size(0)

    return total_conf / total_samples if total_samples > 0 else 0.0

def compute_candidate_components(
    target_id,
    teacher_id,
    cluster_label_centers,
    cluster_prototypes,
    reliability_cache,
):
    target_q = cluster_label_centers[target_id]
    teacher_q = cluster_label_centers[teacher_id]

    target_proto = cluster_prototypes[target_id]
    teacher_proto = cluster_prototypes[teacher_id]

    comp = compute_complementarity(target_q, teacher_q, rho=DKTD_RHO)
    risk = compute_risk(target_q, teacher_q, target_proto, teacher_proto, rho=DKTD_RHO)
    rel = reliability_cache.get(teacher_id, 0.0)

    return {
        "target_cluster": target_id,
        "candidate_teacher": teacher_id,
        "Comp": comp,
        "Rel": rel,
        "Risk": risk,
    }

def kd_loss_with_soft_targets(student_logits, teacher_soft, tau=4.0):
    log_p_s = F.log_softmax(student_logits / tau, dim=1)
    return F.kl_div(log_p_s, teacher_soft, reduction="batchmean") * (tau * tau)

def distill_cluster_multiteacher(
    student_model,
    teacher_models,
    teacher_weights,
    dataloader,
    epochs=DISTILL_EPOCHS,
    tau=DISTILL_TAU,
    lambda_kd=KD_LAMBDA,
    lr=DISTILL_LR,
    return_stats=False,
):
    """
    为一个目标簇客户端执行本地多教师蒸馏。
    调用方为每个客户端单独调用此函数，并在服务器端按本地样本数聚合学生模型。
    """
    distilled_model = deepcopy(student_model).to(DEVICE)
    distilled_model.train()

    for teacher_model in teacher_models:
        teacher_model.eval()

    optimizer = torch.optim.SGD(
        distilled_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    ce_loss = nn.CrossEntropyLoss()

    teacher_weights_tensor = torch.tensor(teacher_weights, dtype=torch.float32, device=DEVICE)
    update_steps = 0
    examples_processed = 0
    teacher_forward_batches = 0

    for _ in range(epochs):
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            student_logits = distilled_model(x)

            with torch.no_grad():
                teacher_soft = None

                for idx, teacher_model in enumerate(teacher_models):
                    teacher_logits = teacher_model(x)
                    teacher_forward_batches += 1
                    soft = F.softmax(teacher_logits / tau, dim=1)

                    if teacher_soft is None:
                        teacher_soft = teacher_weights_tensor[idx] * soft
                    else:
                        teacher_soft += teacher_weights_tensor[idx] * soft

            loss_ce = ce_loss(student_logits, y)
            loss_kd = kd_loss_with_soft_targets(student_logits, teacher_soft, tau)
            loss = loss_ce + lambda_kd * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_steps += 1
            examples_processed += int(y.size(0))

    stats = {
        "student_update_steps": update_steps,
        "examples_processed": examples_processed,
        "teacher_forward_batches": teacher_forward_batches,
    }
    return (distilled_model, stats) if return_stats else distilled_model

def make_safe_train_loader(subset, batch_size, shuffle=True):
    """
    避免 ResNet BatchNorm 在训练模式下遇到 batch size = 1。
    如果样本数不足 2，直接返回 None。
    如果最后一个 batch 会只剩 1 个样本，则 drop_last=True。
    """
    n = len(subset)

    if n < 2:
        return None

    effective_batch_size = min(batch_size, n)
    drop_last = (n % effective_batch_size == 1)

    return DataLoader(
        subset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=drop_last,
    )

def make_client_loaders(trainset, client_data_indices):
    client_loaders = []
    client_weights = []

    for cid in range(NUM_CLIENTS):
        indices = client_data_indices[cid]
        client_weights.append(len(indices))

        if len(indices) < 2:
            client_loaders.append(None)
            continue

        subset = Subset(trainset, indices)
        loader = make_safe_train_loader(subset=subset, batch_size=BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)

    return client_loaders, client_weights

def make_eval_loader(dataset, indices, batch_size=128):
    if indices is None or len(indices) == 0:
        return None
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        generator=torch.Generator().manual_seed(SEED + 104729),
    )

def make_cluster_eval_loaders(dataset, client_indices, cluster_client_ids):
    loaders = []
    for cluster_id in range(NUM_CLUSTERS):
        indices = []
        for client_id in cluster_client_ids[cluster_id]:
            indices.extend(client_indices[client_id])
        loaders.append(make_eval_loader(dataset, indices))
    return loaders

def reset_training_loader_rng(loaders, seed_offset=0):
    """各方法开始时重置 shuffle/worker 随机流，避免方法间数据顺序串扰。"""
    for index, loader in enumerate(loaders):
        if loader is None:
            continue
        generator = torch.Generator()
        generator.manual_seed(SEED + seed_offset + index)
        loader.generator = generator
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "generator"):
            sampler.generator = generator

def verify_client_partitions(original, train, validation, holdout, dataset_size):
    assigned = np.concatenate([np.asarray(x, dtype=np.int64) for x in original])
    if len(assigned) != dataset_size or len(np.unique(assigned)) != dataset_size:
        raise RuntimeError("Dirichlet partition must assign every training sample exactly once")
    for client_id in range(len(original)):
        sets = [set(x) for x in (train[client_id], validation[client_id], holdout[client_id])]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise RuntimeError(f"sample leakage between client splits for client {client_id}")
        if set.union(*sets) != set(original[client_id]):
            raise RuntimeError(f"client split does not cover original samples for client {client_id}")

def local_train(model, dataloader, local_epochs, learning_rate):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=5e-4)
    steps = examples = 0
    for _ in range(local_epochs):
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            steps += 1
            examples += int(y.size(0))
    return model.state_dict(), steps, examples


def compute_dktd_cluster_centers(histograms, prototypes, assignments):
    centers = np.zeros((NUM_CLUSTERS, NUM_CLASSES), dtype=np.float32)
    cluster_prototypes = np.zeros((NUM_CLUSTERS, NUM_CLASSES, prototypes.shape[-1]),
                                  dtype=np.float32)
    for cluster_id in range(NUM_CLUSTERS):
        mask = assignments == cluster_id
        if np.any(mask):
            centers[cluster_id] = histograms[mask].mean(axis=0)
            cluster_prototypes[cluster_id] = prototypes[mask].mean(axis=0)
    return centers, cluster_prototypes


def select_dktd_teachers(target_id, groups, client_loaders, client_weights,
                         label_centers, prototypes, reliability):
    candidates = [cid for cid in range(NUM_CLUSTERS) if cid != target_id
                  and any(client_loaders[i] is not None and client_weights[i] > 0
                          and len(client_loaders[i]) > 0 for i in groups[cid])]
    rows = [compute_candidate_components(target_id, cid, label_centers,
                                          prototypes, reliability) for cid in candidates]
    if not rows:
        return [], []
    comp = minmax_normalize_values([row["Comp"] for row in rows])
    rel = minmax_normalize_values([row["Rel"] for row in rows])
    risk = minmax_normalize_values([row["Risk"] for row in rows])
    for i, row in enumerate(rows):
        row["Score"] = float(DKTD_ALPHA * comp[i] + DKTD_BETA_D * rel[i] - DKTD_GAMMA * risk[i])
    selected = sorted(rows, key=lambda row: row["Score"], reverse=True)[:TEACHER_M]
    return [row["candidate_teacher"] for row in selected], softmax_np([row["Score"] for row in selected]).tolist()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")
    os.replace(str(temporary), str(path))


def append_csv(path, fields, rows):
    path = Path(path)
    first_write = not path.exists()
    with path.open("a", encoding="utf-8-sig" if first_write else "utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if first_write:
            writer.writeheader()
        for row in rows:
            values = {key: json_safe(row.get(key)) for key in fields}
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value
                             for key, value in values.items()})


def finite_mean(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


IDENTITY_FIELDS = ["dataset", "K", "M", "beta", "seed", "strategy"]
EVENT_FIELDS = IDENTITY_FIELDS + [
    "event_id", "round", "original_cluster_id", "original_client_ids", "holdout_samples",
    "holdout_indices_sha256", "evaluation_status", "is_final_round_pair",
    "selected_teachers", "teacher_weights", "own_accuracy_before", "own_accuracy_after",
    "own_accuracy_delta_pp", "own_macro_f1_before", "own_macro_f1_after", "own_macro_f1_delta_pp",
    "own_macro_recall_before", "own_macro_recall_after", "own_macro_recall_delta_pp",
    "own_per_class_recall_before", "own_per_class_recall_after", "declined_strictly",
    "declined_beyond_threshold", "decline_threshold_pp", "student_update_steps", "examples_processed",
]
CLUSTER_FIELDS = IDENTITY_FIELDS + [
    "round", "original_cluster_id", "original_client_ids", "holdout_samples", "evaluation_status",
    "own_accuracy", "own_macro_f1", "own_macro_recall", "own_per_class_recall",
]
ROUND_FIELDS = IDENTITY_FIELDS + [
    "round", "active_original_clusters", "evaluable_original_clusters", "mean_cluster_holdout_acc",
    "worst_cluster_holdout_acc", "mean_cluster_holdout_macro_f1", "round_time_sec",
]


def prepare_original_clusters(args, run_dir, config):
    spec = DATASET_SPECS[DATASET_KEY]
    dataset_class = getattr(torchvision.datasets, spec["torchvision_class"])
    trainset = dataset_class(root=args.data_root, train=True,
                             download=args.download, transform=transform_train)
    if len(trainset) != spec["train_samples"]:
        raise ValueError("dataset size differs from the official training split")
    eval_dataset = copy(trainset)
    eval_dataset.transform = transform_test
    labels = np.asarray(trainset.targets, dtype=np.int64)
    original = dirichlet_split_noniid(labels, NUM_CLIENTS, NUM_CLASSES, args.beta, seed=SEED)
    train, validation, holdout = split_client_indices_for_validation_and_holdout(original, labels, SEED)
    verify_client_partitions(original, train, validation, holdout, len(trainset))
    partitions = {"client_original_indices": original, "client_train_indices": train,
                  "client_validation_indices": validation, "client_holdout_indices": holdout}
    config["client_data_partition_sha256"] = hashlib.sha256(
        json.dumps(partitions, separators=(",", ":")).encode("utf-8")).hexdigest()
    write_json(run_dir / "client_splits.json", partitions)
    base = build_dataset_model(use_pretrained=args.pretrained).to(DEVICE)
    histograms, prototypes, feature_histograms = extract_client_statistics(
        train, trainset.data, labels, NUM_CLASSES, base, ENCODER_BATCH_SIZE)
    distances = build_combined_distance_matrix(histograms, prototypes, feature_histograms,
                                               LAMBDA_LABEL, LAMBDA_FEAT, PROTO_HIST_ALPHA)
    assignments, medoids = kmedoids_precomputed(distances, NUM_CLUSTERS, random_state=SEED)
    centers, cluster_prototypes = compute_dktd_cluster_centers(histograms, prototypes, assignments)
    groups = [[int(i) for i in np.flatnonzero(assignments == cid)] for cid in range(NUM_CLUSTERS)]
    client_loaders, client_weights = make_client_loaders(trainset, train)
    validation_loaders = make_cluster_eval_loaders(eval_dataset, validation, groups)
    holdout_loaders = make_cluster_eval_loaders(eval_dataset, holdout, groups)
    contexts = []
    for cid, clients in enumerate(groups):
        indices = [int(index) for client_id in clients for index in holdout[client_id]]
        loader = holdout_loaders[cid]
        actual_indices = list(loader.dataset.indices) if loader is not None else []
        if indices != actual_indices:
            raise RuntimeError("Original-cluster holdout indices changed for C{}".format(cid))
        contexts.append({"original_cluster_id": cid, "original_client_ids": clients,
                         "holdout_samples": len(indices), "holdout_indices": indices,
                         "holdout_indices_sha256": hashlib.sha256(
                             json.dumps(indices, separators=(",", ":")).encode("utf-8")).hexdigest(),
                         "class_support": np.bincount(labels[indices], minlength=NUM_CLASSES).tolist(),
                         "medoid_client_id": int(medoids[cid])})
    write_json(run_dir / "original_clusters.json", contexts)
    write_json(run_dir / "client_cluster_assignment.json", assignments.tolist())
    initial_state = deepcopy(base.state_dict())
    config["pretrained_load_status"] = PRETRAINED_LOAD_STATUS
    config["training_protocol"]["pretrained_load_status"] = PRETRAINED_LOAD_STATUS
    config["split_samples"] = {"train": sum(map(len, train)), "validation": sum(map(len, validation)),
                               "holdout": sum(map(len, holdout))}
    del base
    return initial_state, groups, contexts, client_loaders, client_weights, validation_loaders, holdout_loaders, centers, cluster_prototypes


def train_dktd(args, run_dir, config, prepared):
    initial_state, groups, contexts, client_loaders, client_weights, validation_loaders, holdout_loaders, centers, prototypes = prepared
    models = []
    for _ in range(NUM_CLUSTERS):
        model = build_dataset_model(use_pretrained=False).to(DEVICE)
        model.load_state_dict(deepcopy(initial_state))
        models.append(model)
    set_seed(SEED)
    reset_training_loader_rng(client_loaders)
    identity = {key: config[key] for key in ("dataset", "K", "M", "beta", "seed")}
    identity["strategy"] = "full_dktd"
    events, final_rows = [], []
    append_csv(run_dir / "dktd_original_cluster_events.csv", EVENT_FIELDS, [])
    append_csv(run_dir / "cluster_metrics.csv", CLUSTER_FIELDS, [])
    append_csv(run_dir / "round_metrics.csv", ROUND_FIELDS, [])
    for round_idx in range(1, args.rounds + 1):
        started = time.perf_counter()
        valid_clients = [[i for i in clients if client_loaders[i] is not None
                          and client_weights[i] > 0 and len(client_loaders[i]) > 0] for clients in groups]
        for cid, clients in enumerate(valid_clients):
            if not clients:
                continue
            states, weights = [], []
            for client_id in clients:
                local = deepcopy(models[cid]).to(DEVICE)
                state, _, _ = local_train(local, client_loaders[client_id], args.local_epochs, args.lr)
                states.append(deepcopy(state))
                weights.append(client_weights[client_id])
                del local
            normalized = [weight / sum(weights) for weight in weights]
            models[cid].load_state_dict(fedavg_weighted_state(models[cid].state_dict(), states, normalized))
        after_by_cluster = {}
        if round_idx % args.distill_interval == 0:
            # Match the original implementation: refresh confidence once per KD
            # cycle, then update targets in ascending order using current models.
            reliability = {cid: compute_confidence(model, validation_loaders[cid])
                           for cid, model in enumerate(models)}
            for cid, clients in enumerate(valid_clients):
                if not clients:
                    continue
                teacher_ids, teacher_weights = select_dktd_teachers(
                    cid, groups, client_loaders, client_weights, centers, prototypes, reliability)
                if not teacher_ids:
                    continue
                before = evaluate_metrics(models[cid], holdout_loaders[cid], NUM_CLASSES)
                states, weights = [], []
                steps = examples = 0
                for client_id in clients:
                    student, stats = distill_cluster_multiteacher(
                        models[cid], [models[i] for i in teacher_ids], teacher_weights,
                        client_loaders[client_id], epochs=args.distill_epochs, tau=args.tau,
                        lambda_kd=args.kd_lambda, lr=args.distill_lr, return_stats=True)
                    states.append(deepcopy(student.state_dict()))
                    weights.append(client_weights[client_id])
                    steps += stats["student_update_steps"]
                    examples += stats["examples_processed"]
                    del student
                models[cid].load_state_dict(fedavg_weighted_state(models[cid].state_dict(), states, weights))
                after = evaluate_metrics(models[cid], holdout_loaders[cid], NUM_CLASSES)
                after_by_cluster[cid] = after
                delta = after["acc"] - before["acc"]
                eligible = contexts[cid]["holdout_samples"] > 0 and np.isfinite(delta)
                event = {**identity, "event_id": "full_dktd:r{}:c{}".format(round_idx, cid),
                         "round": round_idx, "original_cluster_id": cid,
                         "original_client_ids": groups[cid], "holdout_samples": contexts[cid]["holdout_samples"],
                         "holdout_indices_sha256": contexts[cid]["holdout_indices_sha256"],
                         "evaluation_status": "evaluated" if eligible else "invalid_holdout",
                         "is_final_round_pair": int(round_idx == args.rounds),
                         "selected_teachers": teacher_ids, "teacher_weights": teacher_weights,
                         "declined_strictly": int(delta < 0) if eligible else None,
                         "declined_beyond_threshold": int(delta < -args.decline_threshold_pp) if eligible else None,
                         "decline_threshold_pp": args.decline_threshold_pp,
                         "student_update_steps": steps, "examples_processed": examples}
                for stem, key in (("own_accuracy", "acc"), ("own_macro_f1", "macro_f1")):
                    event[stem + "_before"], event[stem + "_after"] = before[key], after[key]
                    event[stem + "_delta_pp"] = after[key] - before[key]
                for phase, metrics in (("before", before), ("after", after)):
                    event["own_macro_recall_" + phase] = float(np.mean(metrics["per_class_recall"]))
                    event["own_per_class_recall_" + phase] = metrics["per_class_recall"].tolist()
                event["own_macro_recall_delta_pp"] = event["own_macro_recall_after"] - event["own_macro_recall_before"]
                events.append(json_safe(event))
                append_csv(run_dir / "dktd_original_cluster_events.csv", EVENT_FIELDS, [event])
                print("Round {} C{} original holdout: {:.2f}% -> {:.2f}%; delta={:+.2f} pp".format(
                    round_idx, cid, before["acc"], after["acc"], delta))
        rows = []
        for cid, model in enumerate(models):
            metrics = after_by_cluster[cid] if cid in after_by_cluster else evaluate_metrics(model, holdout_loaders[cid], NUM_CLASSES)
            eligible = bool(groups[cid]) and contexts[cid]["holdout_samples"] > 0 and np.isfinite(metrics["acc"])
            rows.append({**identity, "round": round_idx, "original_cluster_id": cid,
                         "original_client_ids": groups[cid], "holdout_samples": contexts[cid]["holdout_samples"],
                         "evaluation_status": "evaluated" if eligible else "invalid_holdout",
                         "own_accuracy": metrics["acc"], "own_macro_f1": metrics["macro_f1"],
                         "own_macro_recall": float(np.mean(metrics["per_class_recall"])),
                         "own_per_class_recall": metrics["per_class_recall"].tolist()})
        valid = [row for row in rows if row["evaluation_status"] == "evaluated"]
        round_row = {**identity, "round": round_idx,
                     "active_original_clusters": sum(bool(clients) for clients in groups),
                     "evaluable_original_clusters": len(valid),
                     "mean_cluster_holdout_acc": finite_mean([row["own_accuracy"] for row in valid]),
                     "worst_cluster_holdout_acc": min((row["own_accuracy"] for row in valid), default=None),
                     "mean_cluster_holdout_macro_f1": finite_mean([row["own_macro_f1"] for row in valid]),
                     "round_time_sec": time.perf_counter() - started}
        append_csv(run_dir / "cluster_metrics.csv", CLUSTER_FIELDS, rows)
        append_csv(run_dir / "round_metrics.csv", ROUND_FIELDS, [round_row])
        final_rows = rows
        print("Round {}/{}: mean own accuracy={}, worst={}, time={:.1f}s".format(
            round_idx, args.rounds, round_row["mean_cluster_holdout_acc"],
            round_row["worst_cluster_holdout_acc"], round_row["round_time_sec"]))
    latest = {event["original_cluster_id"]: event for event in events}
    final_report = []
    for row in final_rows:
        pair = latest.get(row["original_cluster_id"])
        final_report.append({**row, "final_round": args.rounds,
                             "last_distillation_round": pair["round"] if pair else None,
                             "pair_is_final_round": bool(pair and pair["round"] == args.rounds),
                             "pair_evaluation_status": pair["evaluation_status"] if pair else "no_distillation_event",
                             **{key: pair.get(key) if pair else None for key in (
                                 "own_accuracy_before", "own_accuracy_after", "own_accuracy_delta_pp",
                                 "own_macro_f1_before", "own_macro_f1_after", "own_macro_f1_delta_pp")}})
    pd.DataFrame(json_safe(final_report)).to_csv(run_dir / "dktd_original_cluster_final.csv",
                                                index=False, encoding="utf-8-sig")


def configure(args, seed):
    global NUM_CLIENTS, NUM_CLUSTERS, BATCH_SIZE, NUM_WORKERS, TEACHER_M, SEED, DEVICE
    global PRETRAINED_LOAD_STATUS
    global DATASET_KEY, FEATURE_HIST_RANGE, transform_train, transform_test
    DATASET_KEY = args.dataset
    FEATURE_HIST_RANGE = tuple(DATASET_SPECS[DATASET_KEY]["feature_hist_range"])
    transform_train, transform_test = build_dataset_transforms(DATASET_KEY)
    NUM_CLIENTS, NUM_CLUSTERS, BATCH_SIZE = args.clients, args.clusters, args.batch_size
    NUM_WORKERS, TEACHER_M, SEED = args.workers, args.teachers, seed
    PRETRAINED_LOAD_STATUS = "pending" if args.pretrained else "not_requested"
    DEVICE = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)


def run_seed(args, seed):
    configure(args, seed)
    set_seed(seed)
    spec = DATASET_SPECS[DATASET_KEY]
    run_name = "{}_K{}_M{}_beta{:g}_seed{}".format(DATASET_KEY, args.clusters, args.teachers, args.beta, seed)
    run_dir = Path(args.result_root).expanduser().resolve() / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        run_dir = run_dir.with_name(run_name + "_repeat{}_{}".format(time.strftime("%Y%m%d_%H%M%S"), time.time_ns() % 1000000))
    run_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().read_bytes()
    (run_dir / "source_snapshot.py").write_bytes(source)
    config = {"protocol_id": PROTOCOL_ID, "status": "preprocessing",
              "dataset": spec["name"], "K": args.clusters, "M": args.teachers,
              "beta": args.beta, "seed": seed, "rounds": args.rounds,
              "distill_interval": args.distill_interval,
              "source_script_sha256": hashlib.sha256(source).hexdigest(),
              "training_protocol": {
                  "dataset_key": DATASET_KEY, "dataset_source": "torchvision official training split",
                  "dataset_training_samples": spec["train_samples"],
                  "clients": args.clients, "model": spec["model"],
                  "model_class": spec["model_class"], "model_configuration_source": spec["model_configuration_source"],
                  "model_output": "ten-class logits", "input_shape": spec["input_shape"],
                  "normalization_mean": spec["mean"], "normalization_std": spec["std"],
                  "feature_extractor": spec["feature_extractor"], "feature_dimension": spec["feature_dim"],
                  "prototype_inputs": "local training indices only; deterministic dataset-specific normalization",
                  "prototype_histogram_range_policy": "new nonnegative MNIST ReLU-feature configuration"
                      if DATASET_KEY == "mnist" else "same range as the corresponding DKTD source",
                  "pretrained_requested": args.pretrained, "local_epochs": args.local_epochs,
                  "batch_size": args.batch_size, "local_lr": args.lr, "momentum": 0.9, "weight_decay": 5e-4,
                  "distill_epochs": args.distill_epochs, "distill_lr": args.distill_lr,
                  "tau": args.tau, "kd_lambda": args.kd_lambda, "rho": DKTD_RHO,
                  "score_weights": [DKTD_ALPHA, DKTD_BETA_D, DKTD_GAMMA],
                  "distance_weights": [LAMBDA_LABEL, LAMBDA_FEAT, PROTO_HIST_ALPHA],
                  "feature_hist_bins": FEATURE_HIST_BINS, "feature_hist_range": FEATURE_HIST_RANGE,
                  "validation_fraction": VAL_FRACTION, "holdout_fraction": CLUSTER_HOLDOUT_FRACTION,
                  "decline_threshold_pp": args.decline_threshold_pp,
                  "teacher_reliability": "mean maximum softmax confidence on each teacher cluster validation union; refreshed once per KD cycle",
                  "target_update_order": "ascending cluster IDs, sequential updates; current cluster models are teachers",
                  "cluster_membership": "initial fixed K-Medoids assignments",
                  "eval_batch_size": 128, "eval_rng_policy": "separate seeded evaluation-loader generators",
                  "train_transform": spec["train_transform"], "eval_transform": spec["eval_transform"],
                  "workers": args.workers, "device_type": DEVICE.type,
                  "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
              }, "evaluation_use": "reporting only; never use holdout metrics for teachers, rollback, or checkpoint selection",
              "python_version": platform.python_version()}
    write_json(run_dir / "run_config.json", config)
    print("\nDKTD only | dataset={} | model={} | seed={} | K={} M={} beta={} | device={}\nOutput: {}".format(
        spec["name"], spec["model_class"], seed, args.clusters, args.teachers, args.beta, DEVICE, run_dir))
    started = time.perf_counter()
    try:
        prepared = prepare_original_clusters(args, run_dir, config)
        config["status"] = "training"
        write_json(run_dir / "run_config.json", config)
        train_dktd(args, run_dir, config, prepared)
    except BaseException as exc:
        config.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       "failure_type": type(exc).__name__, "failure_message": str(exc)})
        write_json(run_dir / "run_config.json", config)
        raise
    config.update({"status": "completed", "total_time_sec": time.perf_counter() - started})
    write_json(run_dir / "run_config.json", config)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DKTD-only fixed original-cluster before/after evaluation and seed-level paired summary")
    parser.add_argument("--dataset", type=canonical_dataset_key, choices=tuple(DATASET_SPECS), default="cifar10",
                        help="cifar10, mnist, fashion_mnist; hyphenated names are also accepted")
    parser.add_argument("--K", "--clusters", dest="clusters", type=int, default=5)
    parser.add_argument("--M", "--teachers", dest="teachers", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.2)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, default=None)
    seeds.add_argument("--seeds", help="Comma-separated nonnegative seeds, or an inclusive range such as 0-9")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--distill-interval", type=int, default=5)
    parser.add_argument("--distill-epochs", type=int, default=1)
    parser.add_argument("--distill-lr", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--kd-lambda", type=float, default=0.7)
    parser.add_argument("--decline-threshold-pp", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--result-root", default="results/dktd_incluster_multidataset")
    parser.add_argument("--download", action="store_true", help="Download the selected dataset if not already present")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    parser.add_argument("--aggregate", action="store_true", help="Summarize existing completed runs without training")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.pretrained is None:
        args.pretrained = args.dataset == "cifar10"
    if args.pretrained and args.dataset != "cifar10" and not args.aggregate:
        parser.error("--pretrained is only supported for CIFAR-10; grayscale CNNs use random initialization")
    if not 2 <= args.clusters <= args.clients or not 1 <= args.teachers < args.clusters:
        parser.error("require 2 <= K <= clients and 1 <= M < K")
    if min(args.rounds, args.local_epochs, args.batch_size, args.distill_interval, args.distill_epochs, args.bootstrap_repetitions) < 1:
        parser.error("rounds, epochs, interval, batch size and bootstrap repetitions must be positive")
    if args.batch_size < 2 or args.workers < 0:
        parser.error("batch size must be >=2 and workers >=0")
    for name in ("beta", "lr", "distill_lr", "tau"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + " must be finite and positive")
    for name in ("kd_lambda", "decline_threshold_pp"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(name + " must be finite and nonnegative")
    if args.device == "cuda" and not args.aggregate and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    try:
        if args.seeds is None:
            args.seed_list = [0 if args.seed is None else args.seed]
        elif "," not in args.seeds and "-" in args.seeds:
            bounds = [int(value) for value in args.seeds.split("-")]
            if len(bounds) != 2 or bounds[0] < 0 or bounds[1] < bounds[0]:
                raise ValueError("invalid range")
            args.seed_list = list(range(bounds[0], bounds[1] + 1))
        else:
            args.seed_list = [int(value.strip()) for value in args.seeds.split(",")]
        if not args.seed_list or min(args.seed_list) < 0 or len(set(args.seed_list)) != len(args.seed_list):
            raise ValueError("negative or duplicate seeds")
    except ValueError:
        parser.error("seeds must be unique nonnegative integers, e.g. 0,1,2 or 0-9")
    return args


# Seed-level reporting uses only completed runs of this standalone protocol.
# Pair metrics require every evaluable original cluster at the same scheduled
# KD update(s). Seed is the independent unit; clusters/rounds are subsamples.
# Final models, last-ten-round models and instantaneous KD pairs remain separate.
_OC_PROTOCOL_ID = PROTOCOL_ID


def _oc_finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _oc_integer(value, name, minimum=0):
    number = _oc_finite(value)
    if isinstance(value, bool) or number is None or not number.is_integer() or number < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(number)


def _oc_json_safe(value):
    return json_safe(value)


def _oc_write_json(path, value):
    write_json(path, value)


def _oc_read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), set(reader.fieldnames or [])


def _oc_write_csv(path, rows, required_columns=()):
    columns = list(required_columns)
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            row = _oc_json_safe(row)
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _oc_bootstrap_mean_ci(values, seed, n_boot):
    # Same percentile-bootstrap algorithm as the original experiment script.
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _oc_aggregate(values, bootstrap_seed=None, n_boot=10000):
    values = [_oc_finite(value) for value in values]
    values = [value for value in values if value is not None]
    result = {"n": len(values), "mean": float(np.mean(values)) if values else None,
              "sample_sd": float(np.std(values, ddof=1)) if len(values) >= 2 else None}
    if bootstrap_seed is not None:
        low, high = _oc_bootstrap_mean_ci(values, bootstrap_seed, n_boot)
        result.update({"bootstrap_ci95_low": low, "bootstrap_ci95_high": high})
    return result


def _oc_group_identity(config):
    dataset = config.get("dataset")
    if dataset not in {spec["name"] for spec in DATASET_SPECS.values()}:
        raise ValueError("dataset must be CIFAR-10, MNIST, or Fashion-MNIST")
    k = _oc_integer(config.get("K"), "K", 2)
    m = _oc_integer(config.get("M"), "M", 1)
    if m >= k:
        raise ValueError("M must be smaller than K")
    beta = _oc_finite(config.get("beta"))
    if beta is None or beta <= 0:
        raise ValueError("beta must be finite and positive")
    rounds = _oc_integer(config.get("rounds"), "rounds", 1)
    interval = _oc_integer(config.get("distill_interval"), "distill_interval", 1)
    source_hash = config.get("source_script_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(
            char not in "0123456789abcdef" for char in source_hash.lower()):
        raise ValueError("source_script_sha256 must be a SHA-256 hex digest")
    protocol = config.get("training_protocol")
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError("training_protocol must be a nonempty dict")
    dataset_key = protocol.get("dataset_key")
    if dataset_key not in DATASET_SPECS or DATASET_SPECS[dataset_key]["name"] != dataset:
        raise ValueError("dataset name/key mismatch in training protocol")
    spec = DATASET_SPECS[dataset_key]
    for field, expected in (
            ("model_class", spec["model_class"]), ("input_shape", spec["input_shape"]),
            ("feature_dimension", spec["feature_dim"]), ("feature_hist_range", spec["feature_hist_range"]),
            ("normalization_mean", spec["mean"]), ("normalization_std", spec["std"]),
            ("dataset_training_samples", spec["train_samples"])):
        if protocol.get(field) != expected:
            raise ValueError("dataset/model protocol mismatch for " + field)
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    protocol_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"dataset": dataset, "K": k, "M": m, "beta": beta, "rounds": rounds,
            "distill_interval": interval, "source_script_sha256": source_hash.lower(),
            "training_protocol_sha256": protocol_hash}


def _oc_group_key(identity):
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _oc_validate_original_clusters(path, k):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        original = json.load(handle)
    if not isinstance(original, list):
        raise ValueError("original_clusters.json must contain a list")
    all_ids, all_clients, evaluable = set(), set(), {}
    for cluster in original:
        cluster_id = _oc_integer(cluster.get("original_cluster_id"), "original_cluster_id")
        if cluster_id >= k or cluster_id in all_ids:
            raise ValueError("duplicate or out-of-range original cluster ID")
        all_ids.add(cluster_id)
        clients = cluster.get("original_client_ids")
        if not isinstance(clients, list):
            raise ValueError("original_client_ids must be a list")
        clients = [_oc_integer(client, "client ID") for client in clients]
        if len(set(clients)) != len(clients) or set(clients) & all_clients:
            raise ValueError("original client membership is duplicated")
        all_clients.update(clients)
        samples = _oc_integer(cluster.get("holdout_samples"), "holdout_samples")
        if not clients and samples:
            raise ValueError("empty original cluster cannot have holdout samples")
        if clients and samples > 0:
            evaluable[cluster_id] = samples
    if all_ids != set(range(k)):
        raise ValueError("original_clusters.json must record every ID from 0 to K-1, including empty clusters")
    return evaluable


def _oc_pair_metrics(events, expected_ids, planned_rounds):
    """Return one seed's round/cluster means only with complete pair coverage."""
    if not planned_rounds:
        return None, "no scheduled distillation in this interval"
    if not expected_ids:
        return None, "no nonempty original cluster has holdout data"
    expected_keys = {(round_id, cluster_id) for round_id in planned_rounds
                     for cluster_id in expected_ids}
    found = {}
    for event in events:
        if event["round"] not in planned_rounds:
            continue
        key = (event["round"], event["original_cluster_id"])
        if event["original_cluster_id"] not in expected_ids:
            continue
        if key in found:
            return None, "duplicate original-cluster distillation event"
        found[key] = event
    if set(found) != expected_keys:
        return None, "incomplete planned-round x original-cluster coverage"
    valid = list(found.values())
    for event in valid:
        if event.get("evaluation_status") != "evaluated":
            return None, "at least one original-cluster event was not evaluated"
        if event["holdout_samples"] != expected_ids[event["original_cluster_id"]]:
            return None, "event holdout sample count differs from original cluster"
        for stem in ("own_accuracy", "own_macro_f1"):
            before = _oc_finite(event.get(stem + "_before"))
            after = _oc_finite(event.get(stem + "_after"))
            delta = _oc_finite(event.get(stem + "_delta_pp"))
            if before is None or after is None or delta is None:
                return None, "nonfinite or missing paired metrics"
            if not 0 <= before <= 100 or not 0 <= after <= 100:
                return None, "paired accuracy/F1 outside [0,100]"
            if not math.isclose(after - before, delta, rel_tol=1e-7, abs_tol=1e-6):
                return None, "logged paired delta differs from after minus before"
            event[stem + "_before"] = before
            event[stem + "_after"] = after
            event[stem + "_delta_pp"] = after - before
    metrics = {"rounds": sorted(planned_rounds), "n_original_clusters": len(expected_ids),
               "n_events": len(valid)}
    for stem in ("own_accuracy", "own_macro_f1"):
        for suffix in ("before", "after", "delta_pp"):
            name = stem + "_" + suffix
            metrics[name] = float(np.mean([event[name] for event in valid]))
    metrics["worst_accuracy_before"] = float(np.mean([
        min(found[(round_id, cid)]["own_accuracy_before"] for cid in expected_ids)
        for round_id in planned_rounds]))
    metrics["worst_accuracy_after"] = float(np.mean([
        min(found[(round_id, cid)]["own_accuracy_after"] for cid in expected_ids)
        for round_id in planned_rounds]))
    metrics["minimum_cluster_accuracy_delta_pp"] = min(
        event["own_accuracy_delta_pp"] for event in valid)
    metrics["fraction_cluster_events_declining"] = float(np.mean([
        event["own_accuracy_delta_pp"] < 0 for event in valid]))
    return metrics, None


def _oc_validate_csv_identity(row, identity, seed):
    if row.get("dataset") != identity["dataset"] or row.get("strategy") != "full_dktd":
        raise ValueError("CSV dataset/strategy differs from run_config")
    for field, expected in (("K", identity["K"]), ("M", identity["M"]), ("seed", seed)):
        if _oc_integer(row.get(field), "CSV " + field) != expected:
            raise ValueError("CSV " + field + " differs from run_config")
    beta = _oc_finite(row.get("beta"))
    if beta is None or not math.isclose(beta, identity["beta"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("CSV beta differs from run_config")


def _oc_read_one_run(config_path, config, identity):
    run_dir = config_path.parent
    seed = _oc_integer(config.get("seed"), "seed")
    if hashlib.sha256((run_dir / "source_snapshot.py").read_bytes()).hexdigest() != identity["source_script_sha256"]:
        raise ValueError("source_snapshot.py does not match the configured source SHA-256")
    expected = _oc_validate_original_clusters(run_dir / "original_clusters.json", identity["K"])
    with (run_dir / "original_clusters.json").open("r", encoding="utf-8-sig") as handle:
        original_contexts = {int(row["original_cluster_id"]): row for row in json.load(handle)}
    for context in original_contexts.values():
        indices = context.get("holdout_indices")
        if not isinstance(indices, list) or len(indices) != context["holdout_samples"]:
            raise ValueError("original-cluster holdout indices/count mismatch")
        indices = [_oc_integer(index, "holdout index") for index in indices]
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate original-cluster holdout index")
        digest = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode("utf-8")).hexdigest()
        if digest != context.get("holdout_indices_sha256"):
            raise ValueError("original-cluster holdout index SHA-256 mismatch")
    events, event_columns = _oc_read_csv(run_dir / "dktd_original_cluster_events.csv")
    required_events = set(IDENTITY_FIELDS) | {"round", "original_cluster_id", "evaluation_status", "holdout_samples",
                       "original_client_ids", "holdout_indices_sha256",
                       "own_accuracy_before", "own_accuracy_after", "own_accuracy_delta_pp",
                       "own_macro_f1_before", "own_macro_f1_after", "own_macro_f1_delta_pp"}
    if not required_events <= event_columns:
        raise ValueError("missing event CSV columns: " + ",".join(sorted(required_events - event_columns)))
    rounds, interval = identity["rounds"], identity["distill_interval"]
    for event in events:
        _oc_validate_csv_identity(event, identity, seed)
        event["round"] = _oc_integer(event.get("round"), "event round", 1)
        event["original_cluster_id"] = _oc_integer(event.get("original_cluster_id"), "event cluster ID")
        event["holdout_samples"] = _oc_integer(event.get("holdout_samples"), "event holdout samples")
        if event["round"] > rounds or event["round"] % interval:
            raise ValueError("event is outside planned distillation schedule")
        if event["original_cluster_id"] >= identity["K"]:
            raise ValueError("event has out-of-range original cluster ID")
        context = original_contexts[event["original_cluster_id"]]
        if json.loads(event["original_client_ids"]) != context["original_client_ids"]:
            raise ValueError("event original-cluster client membership mismatch")
        if event["holdout_indices_sha256"] != context["holdout_indices_sha256"]:
            raise ValueError("event original-cluster holdout SHA-256 mismatch")
    last_planned = rounds // interval * interval
    tail_start = max(1, rounds - 9)
    tail_planned = list(range(((tail_start + interval - 1) // interval) * interval,
                             rounds + 1, interval))
    last_pair, last_reason = _oc_pair_metrics(events, expected, [last_planned] if last_planned else [])
    tail_pair, tail_reason = _oc_pair_metrics(events, expected, tail_planned)

    round_rows, round_columns = _oc_read_csv(run_dir / "round_metrics.csv")
    required_rounds = set(IDENTITY_FIELDS) | {"round", "mean_cluster_holdout_acc", "worst_cluster_holdout_acc",
                       "evaluable_original_clusters"}
    if not required_rounds <= round_columns:
        raise ValueError("missing round CSV columns: " + ",".join(sorted(required_rounds - round_columns)))
    round_values = {}
    for row in round_rows:
        _oc_validate_csv_identity(row, identity, seed)
        round_id = _oc_integer(row.get("round"), "round metric round", 1)
        if round_id > rounds or round_id in round_values:
            raise ValueError("duplicate or out-of-range round metrics")
        mean_acc = _oc_finite(row.get("mean_cluster_holdout_acc"))
        worst_acc = _oc_finite(row.get("worst_cluster_holdout_acc"))
        evaluated = _oc_integer(row.get("evaluable_original_clusters"), "evaluable original clusters")
        if evaluated != len(expected):
            mean_acc = worst_acc = None
        if mean_acc is not None and worst_acc is not None:
            if not 0 <= worst_acc <= mean_acc <= 100:
                raise ValueError("round mean/worst accuracy outside valid range/order")
        round_values[round_id] = {"mean_accuracy": mean_acc, "worst_accuracy": worst_acc}
    final = round_values.get(rounds)
    final_reason = None
    if not expected or not final or any(value is None for value in final.values()):
        final = None
        final_reason = "final-round original-cluster accuracy is missing/nonfinite"
    tail_model = None
    tail_model_reason = None
    tail_ids = list(range(tail_start, rounds + 1))
    if expected and all(round_id in round_values and all(
            value is not None for value in round_values[round_id].values()) for round_id in tail_ids):
        tail_model = {name: float(np.mean([round_values[round_id][name] for round_id in tail_ids]))
                      for name in ("mean_accuracy", "worst_accuracy")}
    else:
        tail_model_reason = "last-ten-round original-cluster metrics have incomplete coverage"
    # A scheduled KD update is the final model update in that communication
    # round, so its complete after metrics must equal that round's model metric.
    for round_id in sorted(set(tail_planned + ([last_planned] if last_planned else []))):
        pair, _ = _oc_pair_metrics(events, expected, [round_id])
        round_metric = round_values.get(round_id)
        if pair and round_metric:
            for model_name, pair_name in (("mean_accuracy", "own_accuracy_after"),
                                          ("worst_accuracy", "worst_accuracy_after")):
                logged = round_metric[model_name]
                if logged is not None and not math.isclose(logged, pair[pair_name], rel_tol=1e-7, abs_tol=1e-6):
                    raise ValueError("after-KD paired accuracy differs from round_metrics at round {}".format(round_id))
    return {"run_dir": str(run_dir.resolve()), "seed": seed, "identity": identity,
            "expected_original_clusters": len(expected), "last_planned_distillation_round": last_planned or None,
            "pair_is_final_round": bool(last_planned and last_planned == rounds),
            "last10_round_start": tail_start, "last10_round_end": rounds,
            "last10_distillation_rounds": tail_planned, "last_planned_pair": last_pair,
            "last10_distillation_pairs": tail_pair, "final_model": final,
            "last10_round_models": tail_model,
            "section_exclusions": {key: reason for key, reason in (
                ("last_planned_pair", last_reason), ("last10_distillation_pairs", tail_reason),
                ("final_model", final_reason), ("last10_round_models", tail_model_reason)) if reason}}


def summarize_dktd_runs(result_root, n_boot=10000):
    """Write per-seed results, independent-seed summaries, and the paired table.

    No training, re-evaluation, checkpoint selection, or imputation is performed.
    A seed with incomplete pair coverage is excluded only from that pair section;
    its valid final/last-ten-round model metrics remain available. Duplicate seeds
    within an identical experimental group are all excluded and audited.
    """
    root = Path(result_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("result_root must be an existing directory")
    n_boot = _oc_integer(n_boot, "n_boot", 1)
    candidates, audit = [], []
    for config_path in sorted(root.rglob("run_config.json")):
        try:
            with config_path.open("r", encoding="utf-8-sig") as handle:
                config = json.load(handle)
            if not isinstance(config, dict):
                raise ValueError("run_config must be an object")
            if config.get("protocol_id") != _OC_PROTOCOL_ID:
                continue
            if config.get("status") != "completed":
                audit.append({"run_dir": str(config_path.parent.resolve()), "scope": "whole_run",
                              "reason": "run status is not completed"})
                continue
            identity = _oc_group_identity(config)
            seed = _oc_integer(config.get("seed"), "seed")
            candidates.append((config_path, config, identity, seed))
        except (OSError, TypeError, ValueError, OverflowError) as error:
            audit.append({"run_dir": str(config_path.parent.resolve()), "scope": "whole_run",
                          "reason": str(error)})

    duplicate_index = defaultdict(list)
    for index, (_, _, identity, seed) in enumerate(candidates):
        duplicate_index[(_oc_group_key(identity), seed)].append(index)
    duplicates = {index for indices in duplicate_index.values() if len(indices) > 1 for index in indices}
    grouped = defaultdict(list)
    for index, (config_path, config, identity, seed) in enumerate(candidates):
        if index in duplicates:
            audit.append({"run_dir": str(config_path.parent.resolve()), "seed": seed,
                          "scope": "whole_run", "reason": "duplicate seed in the same experimental group; all duplicates excluded"})
            continue
        try:
            run = _oc_read_one_run(config_path, config, identity)
            grouped[_oc_group_key(identity)].append(run)
            for section, reason in run["section_exclusions"].items():
                audit.append({"run_dir": run["run_dir"], "seed": seed, "scope": section, "reason": reason})
        except (OSError, TypeError, ValueError, OverflowError) as error:
            audit.append({"run_dir": str(config_path.parent.resolve()), "seed": seed,
                          "scope": "whole_run", "reason": str(error)})

    per_seed, summary_rows, groups, table_rows = [], [], [], []
    for group_key in sorted(grouped):
        runs = sorted(grouped[group_key], key=lambda run: run["seed"])
        identity = runs[0]["identity"]
        group_id = hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:16]
        group_base = {"group_id": group_id, **identity}
        sections = {}
        for run in runs:
            row = {**group_base, "seed": run["seed"], "run_dir": run["run_dir"],
                   "expected_original_clusters": run["expected_original_clusters"],
                   "last_planned_distillation_round": run["last_planned_distillation_round"],
                   "pair_is_final_round": run["pair_is_final_round"],
                   "last10_round_start": run["last10_round_start"], "last10_round_end": run["last10_round_end"],
                   "last10_distillation_rounds": run["last10_distillation_rounds"]}
            for section in ("last_planned_pair", "last10_distillation_pairs", "final_model", "last10_round_models"):
                metrics = run[section]
                row[section + "_complete"] = metrics is not None
                if metrics:
                    for metric, value in metrics.items():
                        row[section + "__" + metric] = value
            per_seed.append(row)

        for section in ("last_planned_pair", "last10_distillation_pairs", "final_model", "last10_round_models"):
            eligible = [run for run in runs if run[section] is not None]
            section_summary = {"n_seeds": len(eligible), "seeds": [run["seed"] for run in eligible],
                               "ten_seed_requirement_met": len(eligible) >= 10, "metrics": {}}
            numeric_metrics = sorted({metric for run in eligible for metric, value in run[section].items()
                                      if isinstance(value, (int, float)) and not isinstance(value, bool)})
            for metric in numeric_metrics:
                is_paired_delta = metric in ("own_accuracy_delta_pp", "own_macro_f1_delta_pp")
                bootstrap_seed = int.from_bytes(hashlib.sha256(
                    (group_key + "|" + section + "|" + metric).encode("utf-8")).digest()[:4], "little")
                values = [run[section].get(metric) for run in eligible]
                result = _oc_aggregate(values, bootstrap_seed if is_paired_delta else None, n_boot)
                section_summary["metrics"][metric] = result
                summary_rows.append({**group_base, "section": section, "metric": metric,
                                     "n_group_seeds": len(runs), "eligible_seeds": section_summary["seeds"],
                                     "ten_seed_requirement_met": section_summary["ten_seed_requirement_met"],
                                     **result})
            sections[section] = section_summary

        paired = sections["last_planned_pair"]
        planned_round = runs[0]["last_planned_distillation_round"]
        for label, flag, metric in (
                ("DKTD, before the current cross-cluster distillation update", "no (current update only)", "own_accuracy_before"),
                ("DKTD, after the current cross-cluster distillation update", "yes", "own_accuracy_after"),
                ("Paired delta, after minus before", "--", "own_accuracy_delta_pp")):
            metric_result = paired["metrics"].get(metric, {"n": 0, "mean": None, "sample_sd": None})
            mean, sd = metric_result["mean"], metric_result["sample_sd"]
            value_text = "" if mean is None else ("{:.2f} +/- {:.2f}".format(mean, sd)
                                                   if sd is not None else "{:.2f}".format(mean))
            low = metric_result.get("bootstrap_ci95_low")
            high = metric_result.get("bootstrap_ci95_high")
            table_rows.append({**group_base, "method": label, "cross_cluster_distillation": flag,
                               "paired_at_round": planned_round,
                               "pair_is_final_round": runs[0]["pair_is_final_round"],
                               "n_paired_seeds": paired["n_seeds"], "paired_seeds": paired["seeds"],
                               "ten_seed_requirement_met": paired["ten_seed_requirement_met"],
                               "unit": "pp" if metric.endswith("delta_pp") else "%",
                               "value_mean_plus_sample_sd": value_text, "mean": mean, "sample_sd": sd,
                               "paired_bootstrap_ci95_low": low, "paired_bootstrap_ci95_high": high,
                               "paired_ci95_text": "[{:+.2f}, {:+.2f}]".format(low, high)
                               if low is not None and high is not None else ""})
        groups.append({**group_base, "n_valid_distinct_seeds": len(runs), "seeds": [run["seed"] for run in runs],
                       "last_planned_distillation_round": planned_round,
                       "pair_is_final_round": runs[0]["pair_is_final_round"], "sections": sections})

    output_dir = root / "original_cluster_multidataset_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    _oc_write_csv(output_dir / "retention_per_seed.csv", per_seed, ("group_id", "seed", "run_dir"))
    _oc_write_csv(output_dir / "retention_across_seed_summary.csv", summary_rows,
                  ("group_id", "section", "metric", "n", "mean", "sample_sd"))
    _oc_write_csv(output_dir / "incluster_paired_table.csv", table_rows,
                  ("group_id", "method", "paired_at_round", "n_paired_seeds", "mean", "sample_sd"))
    _oc_write_csv(output_dir / "excluded_or_invalid_runs.csv", audit, ("run_dir", "seed", "scope", "reason"))
    report = {"protocol_id": _OC_PROTOCOL_ID, "output_directory": str(output_dir),
              "statistical_unit": "one independent experimental seed",
              "cluster_weighting": "equal weight among nonempty original clusters with holdout data",
              "pairing": "same original-cluster holdout, before-current-KD after intra-cluster FedAvg versus after KD aggregation",
              "uncertainty": "sample SD across actual seeds; paired percentile-bootstrap 95% CI on per-seed mean deltas",
              "bootstrap_repetitions": n_boot, "bootstrap_seed_policy": "stable SHA-256 of experimental group, section and metric",
              "n_groups": len(groups), "groups": groups,
              "notes": ["The before-current-KD model can include earlier rounds of distillation.",
                        "The final scheduled KD pair is not the final trained model unless its round equals rounds.",
                        "Last-ten-round model means include every available scheduled round in the complete final window.",
                        "Last-ten-round KD pair means include only scheduled KD updates, with complete round x cluster coverage.",
                        "When rounds is below ten, last-ten-round means use all rounds and report their actual range.",
                        "No missing seed or metric is imputed; one seed has no sample SD or confidence interval.",
                        "Negative mean differences and confidence intervals that permit degradation are retained."]}
    _oc_write_json(output_dir / "retention_across_seed_summary.json", report)
    print("DKTD original-cluster summary: {} group(s), {} valid run(s), output {}".format(
        len(groups), len(per_seed), output_dir))
    return report


def main(argv=None):
    args = parse_args(argv)
    if not args.aggregate:
        for seed in args.seed_list:
            run_seed(args, seed)
    report = summarize_dktd_runs(args.result_root, n_boot=args.bootstrap_repetitions)
    print("Paired table: " + str(Path(report["output_directory"]) / "incluster_paired_table.csv"))
    return 0 if report["n_groups"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
