import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import random
from copy import copy, deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt


# ============================================================
# 1. 全局配置
# ============================================================

DATASET_NAME = "CIFAR-10"

NUM_CLIENTS = 50
NUM_CLASSES = 10

# 论文写作中建议称为 Dirichlet 参数 beta
DIRICHLET_BETA = 0.2

# CLI 的 --rounds 控制轮数；DEBUG_MODE 仅保留作元数据兼容字段。
DEBUG_MODE = False
NUM_ROUNDS = 5 if DEBUG_MODE else 100

LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.01

NUM_CLUSTERS = 5
LAMBDA_LABEL = 0.5
LAMBDA_FEAT = 0.5
PROTO_HIST_ALPHA = 0.5

FEATURE_HIST_BINS = 16
FEATURE_HIST_RANGE = (-3.0, 3.0)
ENCODER_BATCH_SIZE = 256
USE_PRETRAINED_ENCODER = True

SEED = 2
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1.1 实验策略配置
# ============================================================

# 调试时先只跑 full_dktd
#RUN_STRATEGIES = [
#    "full_dktd",
#]

RUN_STRATEGIES = [
     "dktd_no_cross_distill",
     "all_equal",
     "random_m",
     "similarity_topm",
     "dktd_topm_equal",
     "full_dktd",
]

# 可通过 --methods 单独选择，或传入 all 运行本文件包含的全部方法。
ALL_METHODS = [
    "dktd_no_cross_distill",
    "all_equal",
    "random_m",
    "similarity_topm",
    "dktd_topm_equal",
    "full_dktd",
    "comp_only",
    "comp_rel",
    "comp_risk",
    "dktd_step_matched",
    "ifca",
    "fedavg",
    "fedprox",
    "feddf",
]
METHOD_ALIASES = {"cfl": "dktd_no_cross_distill"}

# Figure 3 compares the original strategy curves with the actual IFCA baseline
# and the closest global-distillation baseline, FedDF.
CONVERGENCE_CURVE_METHODS = [
    "dktd_no_cross_distill",
    "similarity_topm",
    "dktd_topm_equal",
    "full_dktd",
    "ifca",
    "feddf",
]
CONVERGENCE_CURVE_LABELS = {
    "dktd_no_cross_distill": "DKTD w/o cross-cluster distillation",
    "similarity_topm": "Similarity-TopM",
    "dktd_topm_equal": "DKTD-TopM-Equal",
    "full_dktd": "DKTD",
    "ifca": "IFCA",
    "feddf": "FedDF",
}
CONVERGENCE_CURVE_METRICS = [
    "mean_cluster_global_test_acc",  # manuscript's Average Cluster Accuracy
    "mean_cluster_holdout_acc",      # cluster-specific held-out performance
]
STABILITY_WINDOW_ROUNDS = 20
STABILITY_METRIC_SOURCES = {
    "last20_sd_mean_cluster_global_test_acc": "mean_cluster_global_test_acc",
    "last20_sd_mean_cluster_holdout_acc": "mean_cluster_holdout_acc",
}

# DKTD 分量消融可改为：
# RUN_STRATEGIES = [
#     "comp_only",
#     "comp_rel",
#     "comp_risk",
#     "full_dktd",
# ]

TEACHER_M = 3
DISTILL_INTERVAL = 5
DISTILL_EPOCHS = 1
DISTILL_TAU = 4.0
KD_LAMBDA = 0.7
DISTILL_LR = 0.01
FEDPROX_MU = 0.01

# DKTD 评分参数
DKTD_RHO = 5.0
DKTD_ALPHA = 1.0       # Comp 权重
DKTD_BETA_D = 0.5      # Rel 权重，避免与 Dirichlet beta 混淆
DKTD_GAMMA = 0.8       # Risk 权重

# 下降超过 0.1 个百分点记为一次负迁移
NEG_TRANSFER_EPS_PERCENT = 0.1

# 每个目标簇中标签占比最低的后 30% 类别定义为弱势类别
WEAK_CLASS_RATIO = 0.3

# 正式修订实验与旧试跑结果隔离，避免历史文件被误纳入跨 seed 汇总。
RESULT_ROOT = "results/fedcmt_dktd/revision_experiments"
DATA_ROOT = "./data"
PRETRAINED_LOAD_STATUS = "not_requested"
VAL_FRACTION = 0.10
CLUSTER_HOLDOUT_FRACTION = 0.10
# Frozen main-experiment protocol: pool the per-client CIFAR-10 validation
# images as an unlabeled server proxy. An external proxy requires a new ID.
FEDDF_PROXY_PROTOCOL = "cifar10_client_validation_union_unlabeled_v1"
EXPERIMENT_GROUP = "main"
SENSITIVITY_AXIS = "not_applicable"
RUN_ID = "unset"


def stamp_run_metadata(frame):
    """将批次标签写入每份表格，避免只能依赖目录名辨认实验。"""
    if frame is None:
        return frame
    for column, value in reversed([
        ("experiment_group", EXPERIMENT_GROUP),
        ("sensitivity_axis", SENSITIVITY_AXIS),
        ("run_id", RUN_ID),
    ]):
        if column in frame.columns:
            frame[column] = value
        else:
            frame.insert(0, column, value)
    return frame

DISTILL_LOG_COLUMNS = [
    "dataset", "K", "M", "beta", "seed", "event_id", "round", "strategy", "score_type",
    "target_cluster", "candidate_teacher",
    "Comp", "Rel", "Risk", "Comp_norm", "Rel_norm", "Risk_norm", "Score",
    "selected", "teacher_weight",
]

DISTILL_EVENT_COLUMNS = [
    "dataset", "K", "M", "beta", "seed", "event_id", "round", "strategy",
    "event_type", "target_cluster", "selected_teachers", "selected_teacher_count",
    "global_acc_before", "global_acc_after", "global_acc_delta",
    "cluster_holdout_acc_before", "cluster_holdout_acc_after", "cluster_holdout_acc_delta",
    "negative_transfer_global", "negative_transfer_cluster_holdout",
    "weak_recall_before", "weak_recall_after", "weak_recall_delta",
    "student_update_steps", "examples_processed", "teacher_forward_batches", "event_time_sec",
]

CLUSTER_METRIC_COLUMNS = [
    "dataset", "K", "M", "beta", "seed", "round", "strategy", "cluster_id",
    "model_scope", "active", "assigned_clients", "train_samples", "holdout_samples",
    "cluster_holdout_acc", "cluster_holdout_macro_f1", "cluster_global_test_acc",
    "cluster_global_test_macro_f1",
]

ASSIGNMENT_COLUMNS = [
    "dataset", "K", "M", "beta", "seed", "round", "client_id", "cluster_id",
    "assignment_loss", "candidate_losses",
]

ROUND_LOG_COLUMNS = [
    "dataset", "K", "M", "beta", "seed", "round", "strategy", "active_clusters",
    "mean_cluster_holdout_acc", "worst_cluster_holdout_acc", "std_cluster_holdout_acc",
    "mean_cluster_global_test_acc", "mean_cluster_holdout_macro_f1",
    "global_model_acc", "weighted_cluster_ensemble_acc", "round_time_sec",
]

RESOURCE_LOG_COLUMNS = [
    "dataset", "K", "M", "beta", "seed", "round", "strategy",
    "local_update_steps", "local_examples_processed", "extra_student_update_steps",
    "extra_student_examples_processed", "distillation_student_uploads",
    "distillation_teacher_model_downloads", "assignment_forward_batches", "teacher_forward_batches",
    "server_distill_kd_loss", "estimated_uplink_bytes", "estimated_downlink_bytes", "round_time_sec",
    "communication_measurement_kind",
]


# ============================================================
# 2. 随机种子
# ============================================================

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


# ============================================================
# 3. CIFAR-10 数据加载
# ============================================================

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

IMAGENET_MEAN = torch.tensor(
    (0.485, 0.456, 0.406), dtype=torch.float32
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    (0.229, 0.224, 0.225), dtype=torch.float32
).view(1, 3, 1, 1)


def normalize_cifar_batch(batch):
    batch = batch.float() / 255.0
    batch = batch.permute(0, 3, 1, 2)
    return (batch - IMAGENET_MEAN) / IMAGENET_STD


# ============================================================
# 4. Dirichlet Non-IID 划分
# ============================================================

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


def save_client_partition_artifacts(
    run_root,
    original_indices,
    train_indices,
    validation_indices,
    holdout_indices,
    labels,
    cluster_labels,
):
    """持久化精确样本索引、客户端簇分配和按类样本计数。"""
    split_payload = {
        "client_original_indices": original_indices,
        "client_train_indices": train_indices,
        "client_validation_indices": validation_indices,
        "client_holdout_indices": holdout_indices,
    }
    split_json = json.dumps(split_payload, separators=(",", ":"))
    digest = hashlib.sha256(split_json.encode("utf-8")).hexdigest()
    saved_payload = dict(split_payload)
    saved_payload["_run_metadata"] = {
        "experiment_group": EXPERIMENT_GROUP,
        "sensitivity_axis": SENSITIVITY_AXIS,
        "run_id": RUN_ID,
        "client_data_partition_sha256": digest,
    }
    with open(os.path.join(run_root, "client_splits.json"), "w", encoding="utf-8") as f:
        json.dump(saved_payload, f, ensure_ascii=False)

    client_rows = []
    for client_id in range(len(original_indices)):
        row = {
            "experiment_group": EXPERIMENT_GROUP,
            "sensitivity_axis": SENSITIVITY_AXIS,
            "run_id": RUN_ID,
            "dataset": DATASET_NAME,
            "K": NUM_CLUSTERS,
            "M": TEACHER_M,
            "beta": DIRICHLET_BETA,
            "seed": SEED,
            "client_id": client_id,
            "initial_cluster_id": int(cluster_labels[client_id]),
            "original_samples": len(original_indices[client_id]),
            "train_samples": len(train_indices[client_id]),
            "validation_samples": len(validation_indices[client_id]),
            "holdout_samples": len(holdout_indices[client_id]),
        }
        for class_id in range(NUM_CLASSES):
            row[f"train_class_{class_id}"] = int(
                np.sum(labels[np.asarray(train_indices[client_id], dtype=np.int64)] == class_id)
            )
        client_rows.append(row)

    pd.DataFrame(client_rows).to_csv(
        os.path.join(run_root, "client_cluster_assignment.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return digest


# ============================================================
# 5. 模型定义
# ============================================================

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
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def encode(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.encode(x)
        return self.classifier(features)


# ============================================================
# 6. 客户端统计信息提取
# ============================================================

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
                client_prototypes.append(np.zeros((num_classes, 512), dtype=np.float32))
                client_feature_histograms.append(
                    np.zeros((num_classes, FEATURE_HIST_BINS), dtype=np.float32)
                )
                continue

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_images = torch.from_numpy(dataset_data[batch_indices])
                batch_labels = labels[batch_indices]

                batch_images = normalize_cifar_batch(batch_images).to(encoder_device)
                features = encoder(batch_images).cpu().numpy()

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

            feature_dim = prototype_sums.shape[1] if prototype_sums is not None else 512
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


# ============================================================
# 7. 距离计算与 K-Medoids 聚类
# ============================================================

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


def compute_cluster_centers(
    label_histograms,
    feature_histograms,
    client_prototypes,
    cluster_labels,
    num_clusters,
):
    cluster_label_centers = np.zeros(
        (num_clusters, label_histograms.shape[1]), dtype=np.float32
    )

    cluster_feature_centers = np.zeros(
        (num_clusters, feature_histograms.shape[1], feature_histograms.shape[2]),
        dtype=np.float32,
    )

    cluster_prototypes = np.zeros(
        (num_clusters, client_prototypes.shape[1], client_prototypes.shape[2]),
        dtype=np.float32,
    )

    for cluster_id in range(num_clusters):
        cluster_mask = cluster_labels == cluster_id

        if np.any(cluster_mask):
            cluster_label_centers[cluster_id] = label_histograms[cluster_mask].mean(axis=0)
            cluster_feature_centers[cluster_id] = feature_histograms[cluster_mask].mean(axis=0)
            cluster_prototypes[cluster_id] = client_prototypes[cluster_mask].mean(axis=0)

    return cluster_label_centers, cluster_feature_centers, cluster_prototypes


# ============================================================
# 8. 本地训练与 FedAvg
# ============================================================

def local_train(
    model,
    dataloader,
    local_epochs=LOCAL_EPOCHS,
    learning_rate=None,
    proximal_mu=0.0,
):
    model.train()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LR if learning_rate is None else learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )

    criterion = nn.CrossEntropyLoss()
    reference_parameters = (
        [parameter.detach().clone() for parameter in model.parameters()]
        if proximal_mu > 0 else None
    )

    total_loss = 0.0
    total_batches = 0
    total_examples = 0

    for _ in range(local_epochs):
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if reference_parameters is not None:
                proximal_penalty = sum(
                    torch.sum((parameter - reference).pow(2))
                    for parameter, reference in zip(model.parameters(), reference_parameters)
                )
                loss = loss + 0.5 * proximal_mu * proximal_penalty
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            total_examples += int(y.size(0))

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return model.state_dict(), avg_loss, total_batches, total_examples


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


def feddf_server_distill(
    student_model,
    client_states,
    proxy_loader,
    epochs=DISTILL_EPOCHS,
    tau=DISTILL_TAU,
    learning_rate=DISTILL_LR,
):
    """原始 FedDF 风格的无标签服务器蒸馏：均值 logits 教师集成 -> 单一学生。"""
    if proxy_loader is None or len(proxy_loader) == 0:
        raise ValueError("FedDF requires a non-empty unlabeled server proxy loader")
    if not client_states:
        return student_model, {
            "student_update_steps": 0,
            "examples_processed": 0,
            "teacher_forward_batches": 0,
            "mean_kd_loss": float("nan"),
        }

    teacher_model = ResNet18Classifier(
        num_classes=NUM_CLASSES, use_pretrained=False
    ).to(DEVICE)
    teacher_logits_by_batch = [None] * len(proxy_loader)
    teacher_forward_batches = 0

    # 按代理 batch 遍历，逐个加载教师，避免 50 个模型同时常驻 GPU，
    # 也避免每个教师都重新创建一轮 DataLoader worker。
    with torch.no_grad():
        teacher_model.eval()
        for batch_idx, images in enumerate(proxy_loader):
            if isinstance(images, (tuple, list)):
                images = images[0]
            images = images.to(DEVICE)
            batch_logits = None
            for client_state in client_states:
                teacher_model.load_state_dict(client_state)
                logits = teacher_model(images)
                teacher_forward_batches += 1
                if batch_logits is None:
                    batch_logits = logits.detach().clone()
                else:
                    batch_logits.add_(logits.detach())
            teacher_logits_by_batch[batch_idx] = batch_logits.div_(len(client_states))

    student_model.train()
    optimizer = optim.SGD(
        student_model.parameters(), lr=learning_rate,
        momentum=0.9, weight_decay=5e-4,
    )
    update_steps = 0
    examples_processed = 0
    total_kd_loss = 0.0
    for _ in range(epochs):
        for batch_idx, images in enumerate(proxy_loader):
            if isinstance(images, (tuple, list)):
                images = images[0]
            images = images.to(DEVICE)
            student_logits = student_model(images)
            teacher_soft = F.softmax(teacher_logits_by_batch[batch_idx] / tau, dim=1)
            loss = F.kl_div(
                F.log_softmax(student_logits / tau, dim=1),
                teacher_soft,
                reduction="batchmean",
            ).clamp_min(0.0) * (tau ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_steps += 1
            examples_processed += int(images.size(0))
            total_kd_loss += float(loss.detach().item())

    del teacher_model
    return student_model, {
        "student_update_steps": update_steps,
        "examples_processed": examples_processed,
        "teacher_forward_batches": teacher_forward_batches,
        "mean_kd_loss": total_kd_loss / update_steps if update_steps else float("nan"),
    }


# ============================================================
# 9. 评价指标
# ============================================================

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


def get_weak_classes(cluster_label_centers, target_id, ratio=WEAK_CLASS_RATIO):
    q = cluster_label_centers[target_id]
    num_weak = max(1, int(np.ceil(NUM_CLASSES * ratio)))
    weak_classes = np.argsort(q)[:num_weak]
    return weak_classes


def compute_weak_recall(per_class_recall, cluster_label_centers, target_id):
    weak_classes = get_weak_classes(cluster_label_centers, target_id)
    return float(np.mean(per_class_recall[weak_classes]))


# ============================================================
# 10. DKTD 教师选择
# ============================================================

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


def compute_similarity_score(target_id, teacher_id, cluster_label_centers, cluster_prototypes):
    """
    Similarity-TopM 的相似性分数。
    距离越小越相似，因此返回 -distance 作为 score。
    """
    target_q = cluster_label_centers[target_id]
    teacher_q = cluster_label_centers[teacher_id]

    target_proto = cluster_prototypes[target_id]
    teacher_proto = cluster_prototypes[teacher_id]

    d_label = js_distance(target_q, teacher_q)
    d_proto = compute_prototype_distance(target_proto, teacher_proto, target_q, teacher_q)

    distance = 0.5 * d_label + 0.5 * d_proto
    return float(-distance)


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


def compute_dktd_scores(
    target_id,
    candidate_ids,
    strategy,
    cluster_label_centers,
    cluster_prototypes,
    reliability_cache,
):
    rows = []

    for teacher_id in candidate_ids:
        rows.append(
            compute_candidate_components(
                target_id=target_id,
                teacher_id=teacher_id,
                cluster_label_centers=cluster_label_centers,
                cluster_prototypes=cluster_prototypes,
                reliability_cache=reliability_cache,
            )
        )

    if len(rows) == 0:
        return []

    comp_norm = minmax_normalize_values([r["Comp"] for r in rows])
    rel_norm = minmax_normalize_values([r["Rel"] for r in rows])
    risk_norm = minmax_normalize_values([r["Risk"] for r in rows])

    for idx, row in enumerate(rows):
        row["Comp_norm"] = float(comp_norm[idx])
        row["Rel_norm"] = float(rel_norm[idx])
        row["Risk_norm"] = float(risk_norm[idx])

        if strategy == "comp_only":
            score = row["Comp_norm"]
            score_type = "dktd_comp_only"

        elif strategy == "comp_rel":
            score = DKTD_ALPHA * row["Comp_norm"] + DKTD_BETA_D * row["Rel_norm"]
            score_type = "dktd_comp_rel"

        elif strategy == "comp_risk":
            score = DKTD_ALPHA * row["Comp_norm"] - DKTD_GAMMA * row["Risk_norm"]
            score_type = "dktd_comp_risk"

        elif strategy == "dktd_topm_equal":
            score = (
                DKTD_ALPHA * row["Comp_norm"]
                + DKTD_BETA_D * row["Rel_norm"]
                - DKTD_GAMMA * row["Risk_norm"]
            )
            score_type = "dktd_select_equal_weight"

        else:
            score = (
                DKTD_ALPHA * row["Comp_norm"]
                + DKTD_BETA_D * row["Rel_norm"]
                - DKTD_GAMMA * row["Risk_norm"]
            )
            score_type = "dktd_full"

        row["Score"] = float(score)
        row["score_type"] = score_type

    return rows


def select_teachers(
    strategy,
    target_id,
    round_idx,
    cluster_client_ids,
    client_loaders,
    client_weights,
    cluster_label_centers,
    cluster_prototypes,
    reliability_cache,
):
    """
    统一教师选择接口。
    返回 teacher_ids、teacher_weights、score_rows。
    """
    candidate_ids = []
    for cid in range(NUM_CLUSTERS):
        if cid == target_id:
            continue
        has_trainable_client = any(
            client_loaders[client_id] is not None
            and client_weights[client_id] > 0
            and len(client_loaders[client_id]) > 0
            for client_id in cluster_client_ids[cid]
        )
        if has_trainable_client:
            candidate_ids.append(cid)

    if len(candidate_ids) == 0:
        return [], {}, []

    if strategy in {"cfl", "dktd_no_cross_distill"}:
        return [], {}, []

    if strategy == "all_equal":
        teacher_ids = candidate_ids
        teacher_weights = {tid: 1.0 / len(teacher_ids) for tid in teacher_ids}
        score_rows = []

        for tid in candidate_ids:
            row = compute_candidate_components(
                target_id, tid, cluster_label_centers, cluster_prototypes, reliability_cache
            )
            row["Score"] = 0.0
            row["score_type"] = "none_equal_all"
            row["selected"] = int(tid in teacher_ids)
            row["teacher_weight"] = teacher_weights.get(tid, 0.0)
            score_rows.append(row)

        return teacher_ids, teacher_weights, score_rows

    if strategy == "random_m":
        rng = np.random.default_rng(SEED + round_idx * 1000 + target_id)
        teacher_ids = list(
            int(teacher_id)
            for teacher_id in rng.choice(
                candidate_ids,
                size=min(TEACHER_M, len(candidate_ids)),
                replace=False,
            )
        )
        teacher_weights = {tid: 1.0 / len(teacher_ids) for tid in teacher_ids}
        score_rows = []

        for tid in candidate_ids:
            row = compute_candidate_components(
                target_id, tid, cluster_label_centers, cluster_prototypes, reliability_cache
            )
            row["Score"] = 0.0
            row["score_type"] = "none_random"
            row["selected"] = int(tid in teacher_ids)
            row["teacher_weight"] = teacher_weights.get(tid, 0.0)
            score_rows.append(row)

        return teacher_ids, teacher_weights, score_rows

    if strategy == "similarity_topm":
        rows = []

        for tid in candidate_ids:
            row = compute_candidate_components(
                target_id, tid, cluster_label_centers, cluster_prototypes, reliability_cache
            )
            row["Score"] = compute_similarity_score(
                target_id, tid, cluster_label_centers, cluster_prototypes
            )
            row["score_type"] = "similarity"
            rows.append(row)

        rows = sorted(rows, key=lambda x: x["Score"], reverse=True)
        selected_rows = rows[:min(TEACHER_M, len(rows))]
        teacher_ids = [r["candidate_teacher"] for r in selected_rows]
        weights = softmax_np([r["Score"] for r in selected_rows])
        teacher_weights = {tid: float(weight) for tid, weight in zip(teacher_ids, weights)}

        for row in rows:
            tid = row["candidate_teacher"]
            row["selected"] = int(tid in teacher_ids)
            row["teacher_weight"] = teacher_weights.get(tid, 0.0)

        return teacher_ids, teacher_weights, rows

    # DKTD 系列策略：comp_only / comp_rel / comp_risk / dktd_topm_equal / full_dktd
    rows = compute_dktd_scores(
        target_id=target_id,
        candidate_ids=candidate_ids,
        strategy=strategy,
        cluster_label_centers=cluster_label_centers,
        cluster_prototypes=cluster_prototypes,
        reliability_cache=reliability_cache,
    )

    rows = sorted(rows, key=lambda x: x["Score"], reverse=True)
    selected_rows = rows[:min(TEACHER_M, len(rows))]
    teacher_ids = [r["candidate_teacher"] for r in selected_rows]

    if strategy == "dktd_topm_equal":
        teacher_weights = {tid: 1.0 / len(teacher_ids) for tid in teacher_ids}
    else:
        weights = softmax_np([r["Score"] for r in selected_rows])
        teacher_weights = {tid: float(weight) for tid, weight in zip(teacher_ids, weights)}

    for row in rows:
        tid = row["candidate_teacher"]
        row["selected"] = int(tid in teacher_ids)
        row["teacher_weight"] = teacher_weights.get(tid, 0.0)

    return teacher_ids, teacher_weights, rows


# ============================================================
# 11. 多教师蒸馏
# ============================================================

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


# ============================================================
# 12. DataLoader 构建：修复 BatchNorm batch size = 1 报错
# ============================================================

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
    )


class UnlabeledImageSubset(Dataset):
    """包装已有数据索引，只向算法返回图像，不暴露标签。"""

    def __init__(self, dataset, indices):
        self.subset = Subset(dataset, [int(index) for index in indices])

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        sample = self.subset[index]
        return sample[0] if isinstance(sample, (tuple, list)) else sample


def make_unlabeled_proxy_loader(dataset, indices, batch_size=128):
    if indices is None or len(indices) < 2:
        return None
    subset = UnlabeledImageSubset(dataset, indices)
    effective_batch_size = min(batch_size, len(subset))
    drop_last = (len(subset) % effective_batch_size == 1)
    return DataLoader(
        subset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=drop_last,
        persistent_workers=NUM_WORKERS > 0,
    )


def make_client_eval_loaders(dataset, client_indices):
    return [make_eval_loader(dataset, indices) for indices in client_indices]


def make_cluster_eval_loaders(dataset, client_indices, cluster_client_ids):
    loaders = []
    for cluster_id in range(NUM_CLUSTERS):
        indices = []
        for client_id in cluster_client_ids[cluster_id]:
            indices.extend(client_indices[client_id])
        loaders.append(make_eval_loader(dataset, indices))
    return loaders


def make_dynamic_cluster_eval_loaders(dataset, client_indices, assignments):
    dynamic_groups = defaultdict(list)
    for client_id, cluster_id in enumerate(assignments):
        if int(cluster_id) >= 0:
            dynamic_groups[int(cluster_id)].append(client_id)
    cluster_groups = [dynamic_groups[k] for k in range(NUM_CLUSTERS)]
    return make_cluster_eval_loaders(dataset, client_indices, cluster_groups), cluster_groups


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


# ============================================================
# 13. 可视化
# ============================================================

def save_cluster_analysis_figure(
    client_histograms,
    cluster_labels,
    combined_distance_matrix,
    save_path,
):
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(client_histograms.T, aspect="auto", cmap="YlOrRd")
    plt.xlabel("Client ID")
    plt.ylabel("Class")
    plt.title("Client Label Distributions")
    plt.colorbar(label="Probability")

    plt.subplot(1, 3, 2)
    for cluster_id in range(NUM_CLUSTERS):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = client_histograms[cluster_mask]
        if len(cluster_data) > 0:
            plt.plot(cluster_data.T, alpha=0.5, label=f"Cluster {cluster_id}")

    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Cluster-wise Label Distributions")
    plt.legend(fontsize=8)

    plt.subplot(1, 3, 3)
    plt.imshow(combined_distance_matrix, aspect="auto", cmap="viridis")
    plt.xlabel("Client ID")
    plt.ylabel("Client ID")
    plt.title("Combined Client Distance Matrix")
    plt.colorbar(label="Distance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_strategy_curve(round_df, strategy_dir, strategy):
    """
    保存策略收敛曲线。
    这里显式把 pandas Series 转成 numpy array，避免部分 pandas/matplotlib 版本
    在 plt.plot() 内部触发 Series[:, None] 导致报错。
    """
    if round_df is None or len(round_df) == 0:
        print(f"[{strategy}] round_df 为空，跳过曲线绘制")
        return

    rounds = round_df["round"].to_numpy(dtype=float)
    mean_cluster_acc = round_df["mean_cluster_holdout_acc"].to_numpy(dtype=float)
    global_model_acc = round_df["global_model_acc"].to_numpy(dtype=float)
    ensemble_acc = round_df["weighted_cluster_ensemble_acc"].to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))

    plt.plot(
        rounds,
        mean_cluster_acc,
        linewidth=2,
        marker="o",
        markersize=3,
        markevery=10,
        label="Mean cluster-held-out Acc",
    )

    if np.isfinite(global_model_acc).any():
        plt.plot(rounds, global_model_acc, linewidth=2, linestyle="--", label="Global-model test Acc")
    if np.isfinite(ensemble_acc).any():
        plt.plot(rounds, ensemble_acc, linewidth=1.5, linestyle=":", label="Weighted cluster ensemble Acc")

    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(
        f"{strategy} | {EXPERIMENT_GROUP}/{SENSITIVITY_AXIS} | "
        f"{DATASET_NAME}, K={NUM_CLUSTERS}, M={TEACHER_M}, "
        f"beta={DIRICHLET_BETA}, seed={SEED}",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(strategy_dir, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 14. 单个 strategy 训练
# ============================================================

def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def evaluate_cluster_models(
    models,
    test_loader,
    cluster_holdout_loaders,
    cluster_client_ids,
    train_sample_weights,
    round_idx,
    strategy,
    model_scope="cluster",
):
    """同时记录每簇留出分布和官方全局测试集上的结果。"""
    rows = []
    holdout_accs, global_accs, holdout_f1s = [], [], []
    global_metric_cache = {}
    for cluster_id in range(NUM_CLUSTERS):
        active = bool(cluster_client_ids[cluster_id])
        model = models[cluster_id]
        if active:
            cache_key = id(model)
            if cache_key not in global_metric_cache:
                global_metric_cache[cache_key] = evaluate_metrics(model, test_loader, NUM_CLASSES)
            global_metrics = global_metric_cache[cache_key]
        else:
            global_metrics = None
        holdout_loader = cluster_holdout_loaders[cluster_id]
        holdout_metrics = (
            evaluate_metrics(model, holdout_loader, NUM_CLASSES)
            if active and holdout_loader is not None and len(holdout_loader) > 0
            else None
        )
        global_acc = global_metrics["acc"] if global_metrics else float("nan")
        global_f1 = global_metrics["macro_f1"] if global_metrics else float("nan")
        holdout_acc = holdout_metrics["acc"] if holdout_metrics else float("nan")
        holdout_f1 = holdout_metrics["macro_f1"] if holdout_metrics else float("nan")
        if np.isfinite(global_acc):
            global_accs.append(global_acc)
        if np.isfinite(holdout_acc):
            holdout_accs.append(holdout_acc)
            holdout_f1s.append(holdout_f1)
        rows.append({
            "dataset": DATASET_NAME,
            "K": NUM_CLUSTERS,
            "M": TEACHER_M,
            "beta": DIRICHLET_BETA,
            "seed": SEED,
            "round": round_idx,
            "strategy": strategy,
            "cluster_id": cluster_id,
            "model_scope": model_scope,
            "active": int(active),
            "assigned_clients": len(cluster_client_ids[cluster_id]),
            "train_samples": int(train_sample_weights[cluster_id]),
            "holdout_samples": (
                len(holdout_loader.dataset) if holdout_loader is not None else 0
            ),
            "cluster_holdout_acc": holdout_acc,
            "cluster_holdout_macro_f1": holdout_f1,
            "cluster_global_test_acc": global_acc,
            "cluster_global_test_macro_f1": global_f1,
        })
    return rows, {
        "mean_cluster_holdout_acc": safe_mean(holdout_accs),
        "worst_cluster_holdout_acc": (
            float(np.min(holdout_accs)) if holdout_accs else float("nan")
        ),
        "std_cluster_holdout_acc": (
            float(np.std(holdout_accs)) if holdout_accs else float("nan")
        ),
        "mean_cluster_global_test_acc": safe_mean(global_accs),
        "mean_cluster_holdout_macro_f1": safe_mean(holdout_f1s),
        "active_clusters": int(sum(bool(x) for x in cluster_client_ids)),
    }


def make_weighted_cluster_ensemble(models, weights, initial_state):
    valid_ids = [cid for cid, weight in enumerate(weights) if weight > 0]
    if not valid_ids:
        return None
    total_weight = float(sum(weights[cid] for cid in valid_ids))
    normalized_weights = [weights[cid] / total_weight for cid in valid_ids]
    ensemble = ResNet18Classifier(num_classes=NUM_CLASSES, use_pretrained=False).to(DEVICE)
    ensemble.load_state_dict(deepcopy(initial_state))
    ensemble.load_state_dict(fedavg_weighted_state(
        ensemble.state_dict(),
        [models[cid].state_dict() for cid in valid_ids],
        normalized_weights,
    ))
    return ensemble


def model_state_nbytes(state_dict):
    return int(sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values()))


def write_dktd_original_cluster_report(strategy_dir, contexts, events, cluster_logs, round_logs):
    """报告同次KD的M/M'在固定原簇留出集上的变化，不用留出结果选择模型。"""
    event_columns = [
        "dataset", "K", "M", "beta", "seed", "strategy", "event_id", "round",
        "original_cluster_id", "original_client_ids", "holdout_samples", "holdout_indices_sha256",
        "selected_teachers", "evaluation_status", "is_final_round_pair",
        "own_accuracy_before", "own_accuracy_after", "own_accuracy_delta_pp",
        "own_macro_f1_before", "own_macro_f1_after", "own_macro_f1_delta_pp",
        "own_macro_recall_before", "own_macro_recall_after", "own_macro_recall_delta_pp",
        "own_per_class_recall_before", "own_per_class_recall_after",
        "global_accuracy_before", "global_accuracy_after", "global_accuracy_delta_pp",
        "declined_strictly", "declined_beyond_threshold", "decline_threshold_pp",
    ]
    event_frame = stamp_run_metadata(pd.DataFrame(events, columns=event_columns))
    event_frame.to_csv(os.path.join(strategy_dir, "dktd_original_cluster_events.csv"),
                       index=False, encoding="utf-8-sig")

    def finite_or_none(value):
        return float(value) if value is not None and np.isfinite(value) else None

    def mean_or_none(values):
        return finite_or_none(safe_mean(values))

    expected_ids = {cid for cid, context in contexts.items()
                    if context["original_client_count"] > 0 and context["holdout_samples"] > 0}

    def pair_summary(rows, one_event_per_cluster=False):
        valid = [row for row in rows if row["evaluation_status"] == "evaluated"]
        before = [row["own_accuracy_before"] for row in valid]
        after = [row["own_accuracy_after"] for row in valid]
        deltas = [row["own_accuracy_delta_pp"] for row in valid]
        complete = bool(expected_ids) and {row["original_cluster_id"] for row in valid} == expected_ids
        return {
            "recorded_events": len(rows), "eligible_events": len(valid),
            "eligible_original_clusters": len({row["original_cluster_id"] for row in valid}),
            "expected_evaluable_original_clusters": len(expected_ids),
            "complete_original_cluster_coverage": complete,
            "mean_own_accuracy_before": mean_or_none(before),
            "mean_own_accuracy_after": mean_or_none(after),
            "mean_own_accuracy_delta_pp": mean_or_none(deltas),
            "mean_own_macro_f1_delta_pp": mean_or_none([row["own_macro_f1_delta_pp"] for row in valid]),
            "worst_own_accuracy_before": min(before) if before else None,
            "worst_own_accuracy_after": min(after) if after else None,
            "worst_accuracy_delta_pp": min(after) - min(before) if valid else None,
            "minimum_paired_accuracy_delta_pp": min(deltas) if deltas else None,
            "strict_decline_count": sum(row["declined_strictly"] for row in valid),
            "decline_beyond_threshold_count": sum(row["declined_beyond_threshold"] for row in valid),
            "strict_decline_rate": mean_or_none([row["declined_strictly"] for row in valid]),
            "decline_beyond_threshold_rate": mean_or_none([row["declined_beyond_threshold"] for row in valid]),
            "all_original_clusters_within_threshold": (
                all(row["declined_beyond_threshold"] == 0 for row in valid)
                if one_event_per_cluster and complete else None
            ),
        }

    final_round = int(round_logs[-1]["round"]) if round_logs else 0
    final_by_cluster = {int(row["cluster_id"]): row for row in cluster_logs
                        if int(row["round"]) == final_round}
    latest_by_cluster = {}
    for event in sorted(events, key=lambda row: row["round"]):
        latest_by_cluster[event["original_cluster_id"]] = event
    tail_start = max(1, final_round - 9)
    final_rows = []
    for cluster_id, context in contexts.items():
        final = final_by_cluster.get(cluster_id, {})
        pair = latest_by_cluster.get(cluster_id)
        tail = [row for row in cluster_logs if int(row["cluster_id"]) == cluster_id
                and tail_start <= int(row["round"]) <= final_round]
        final_rows.append({
            "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
            "beta": DIRICHLET_BETA, "seed": SEED, "strategy": "full_dktd",
            "original_cluster_id": cluster_id, **context, "final_round": final_round,
            "last_distillation_round": pair["round"] if pair else None,
            "pair_is_final_round": int(pair is not None and pair["round"] == final_round),
            "pair_evaluation_status": pair["evaluation_status"] if pair else "no_distillation_event",
            **{key: pair[key] if pair else None for key in (
                "own_accuracy_before", "own_accuracy_after", "own_accuracy_delta_pp",
                "own_macro_f1_before", "own_macro_f1_after", "own_macro_f1_delta_pp",
                "declined_strictly", "declined_beyond_threshold",
            )},
            "final_model_own_accuracy": finite_or_none(final.get("cluster_holdout_acc")),
            "final_model_own_macro_f1": finite_or_none(final.get("cluster_holdout_macro_f1")),
            "final_model_global_accuracy": finite_or_none(final.get("cluster_global_test_acc")),
            "last10_rounds_own_accuracy": mean_or_none([row["cluster_holdout_acc"] for row in tail]),
            "last10_rounds_observed": len(tail), "decline_threshold_pp": NEG_TRANSFER_EPS_PERCENT,
        })
    stamp_run_metadata(pd.DataFrame(final_rows)).to_csv(
        os.path.join(strategy_dir, "dktd_original_cluster_final.csv"),
        index=False, encoding="utf-8-sig",
    )
    last_planned_round = final_round // DISTILL_INTERVAL * DISTILL_INTERVAL
    tail_rounds = [row for row in round_logs if tail_start <= int(row["round"]) <= final_round]
    summary = {
        "experiment_group": EXPERIMENT_GROUP, "sensitivity_axis": SENSITIVITY_AXIS,
        "run_id": RUN_ID, "dataset": DATASET_NAME, "K": NUM_CLUSTERS,
        "M": TEACHER_M, "beta": DIRICHLET_BETA, "seed": SEED, "strategy": "full_dktd",
        "comparison": "M: after current-round intra-cluster FedAvg; M_prime: after client-local KD and target-cluster student aggregation",
        "evaluation_data": "untouched client holdout data pooled by original fixed K-Medoids membership; same samples/order before and after",
        "metric_units": "accuracy/F1/recall in percent; differences and decline threshold in percentage points",
        "macro_class_policy": "macro F1 and macro recall average all configured classes; zero-support classes contribute zero",
        "decision_use": "reporting only; holdout scores are not used to select teachers, roll back updates, or select checkpoints",
        "decline_threshold_pp": NEG_TRANSFER_EPS_PERCENT,
        "all_distillation_events": pair_summary(events),
        "last_scheduled_distillation_round": last_planned_round if last_planned_round else None,
        "last_scheduled_distillation_pair": pair_summary(
            [row for row in events if row["round"] == last_planned_round], True),
        "final_round": final_round,
        "final_round_distillation_pair": pair_summary(
            [row for row in events if row["round"] == final_round], True),
        "final_model": {
            "mean_own_accuracy": mean_or_none([row["final_model_own_accuracy"] for row in final_rows]),
            "worst_own_accuracy": min((row["final_model_own_accuracy"] for row in final_rows
                                       if row["final_model_own_accuracy"] is not None), default=None),
            "mean_own_macro_f1": mean_or_none([row["final_model_own_macro_f1"] for row in final_rows]),
        },
        "last10_rounds": {
            "round_start": tail_start, "round_end": final_round, "observed_rounds": len(tail_rounds),
            "mean_own_accuracy": mean_or_none([row["mean_cluster_holdout_acc"] for row in tail_rounds]),
            "worst_own_accuracy": mean_or_none([row["worst_cluster_holdout_acc"] for row in tail_rounds]),
        },
        "interpretation": "Paired changes describe the recorded KD updates. Clusters/events within one seed are not independent training runs; across-seed uncertainty must be evaluated separately. Final-model accuracy after later ordinary rounds is not the immediate KD effect.",
    }
    with open(os.path.join(strategy_dir, "dktd_original_cluster_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def write_method_outputs(
    strategy,
    strategy_dir,
    round_logs,
    cluster_logs,
    event_logs,
    candidate_logs,
    assignment_logs=None,
    resource_logs=None,
    total_time_sec=0.0,
):
    round_df = pd.DataFrame(round_logs, columns=ROUND_LOG_COLUMNS)
    cluster_df = pd.DataFrame(cluster_logs, columns=CLUSTER_METRIC_COLUMNS)
    event_df = pd.DataFrame(event_logs, columns=DISTILL_EVENT_COLUMNS)
    candidate_df = pd.DataFrame(candidate_logs, columns=DISTILL_LOG_COLUMNS)
    for frame in (round_df, cluster_df, event_df, candidate_df):
        stamp_run_metadata(frame)
    round_df.to_csv(os.path.join(strategy_dir, "round_metrics.csv"), index=False, encoding="utf-8-sig")
    cluster_df.to_csv(os.path.join(strategy_dir, "cluster_metrics.csv"), index=False, encoding="utf-8-sig")
    event_df.to_csv(os.path.join(strategy_dir, "distill_events.csv"), index=False, encoding="utf-8-sig")
    candidate_df.to_csv(os.path.join(strategy_dir, "distill_candidates.csv"), index=False, encoding="utf-8-sig")
    resource_df = pd.DataFrame(columns=RESOURCE_LOG_COLUMNS)
    if assignment_logs is not None:
        assignment_df = pd.DataFrame(assignment_logs, columns=ASSIGNMENT_COLUMNS)
        stamp_run_metadata(assignment_df)
        assignment_df.to_csv(
            os.path.join(strategy_dir, "assignment_log.csv"), index=False, encoding="utf-8-sig"
        )
    if resource_logs is not None:
        resource_df = pd.DataFrame(resource_logs, columns=RESOURCE_LOG_COLUMNS)
        stamp_run_metadata(resource_df)
        resource_df.to_csv(
            os.path.join(strategy_dir, "resource_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    save_strategy_curve(round_df, strategy_dir, strategy)

    final_row = round_logs[-1] if round_logs else {}
    kd_events = event_df[event_df["event_type"] == "multi_teacher_distillation"] if len(event_df) else event_df
    stability_tail = round_df.sort_values("round").tail(STABILITY_WINDOW_ROUNDS)
    stability_round_count = int(stability_tail["round"].nunique()) if len(stability_tail) else 0
    last20_stability = {}
    for output_metric, source_metric in STABILITY_METRIC_SOURCES.items():
        values = (
            pd.to_numeric(stability_tail[source_metric], errors="coerce").dropna()
            if source_metric in stability_tail else pd.Series(dtype=float)
        )
        last20_stability[output_metric] = (
            float(values.std(ddof=1))
            if stability_round_count == STABILITY_WINDOW_ROUNDS
            and len(values) == STABILITY_WINDOW_ROUNDS
            else None
        )
    summary = {
        "experiment_group": EXPERIMENT_GROUP,
        "sensitivity_axis": SENSITIVITY_AXIS,
        "run_id": RUN_ID,
        "dataset": DATASET_NAME,
        "K": NUM_CLUSTERS,
        "M": TEACHER_M,
        "beta": DIRICHLET_BETA,
        "seed": SEED,
        "strategy": strategy,
        "final_mean_cluster_holdout_acc": final_row.get("mean_cluster_holdout_acc"),
        "final_worst_cluster_holdout_acc": final_row.get("worst_cluster_holdout_acc"),
        "final_std_cluster_holdout_acc": final_row.get("std_cluster_holdout_acc"),
        "final_mean_cluster_global_test_acc": final_row.get("mean_cluster_global_test_acc"),
        "stability_window_rounds_required": STABILITY_WINDOW_ROUNDS,
        "stability_window_rounds_observed": stability_round_count,
        **last20_stability,
        "final_global_model_acc": final_row.get("global_model_acc"),
        "final_weighted_cluster_ensemble_acc": final_row.get("weighted_cluster_ensemble_acc"),
        "num_distillation_events": int(len(kd_events)),
        "negative_transfer_rate_cluster_holdout": (
            float(kd_events["negative_transfer_cluster_holdout"].mean())
            if len(kd_events) else None
        ),
        "negative_transfer_cluster_holdout_denominator": (
            int(kd_events["negative_transfer_cluster_holdout"].notna().sum())
            if len(kd_events) else 0
        ),
        "negative_transfer_rate_global_test": (
            float(kd_events["negative_transfer_global"].mean())
            if len(kd_events) else None
        ),
        "mean_distillation_event_gain_cluster_holdout": (
            float(kd_events["cluster_holdout_acc_delta"].mean())
            if len(kd_events) else None
        ),
        "teacher_set_change_rate": None,
        "total_method_time_sec": total_time_sec,
        "estimated_total_uplink_bytes": (
            int(pd.to_numeric(resource_df["estimated_uplink_bytes"], errors="coerce").sum())
            if len(resource_df) else 0
        ),
        "estimated_total_downlink_bytes": (
            int(pd.to_numeric(resource_df["estimated_downlink_bytes"], errors="coerce").sum())
            if len(resource_df) else 0
        ),
    }
    if len(kd_events) > 0:
        change_flags = []
        for _, group in kd_events.groupby("target_cluster"):
            previous = None
            for selected in group.sort_values("round")["selected_teachers"]:
                current = set(json.loads(selected))
                if previous is not None:
                    change_flags.append(int(current != previous))
                previous = current
        summary["teacher_set_change_rate"] = (
            float(np.mean(change_flags)) if change_flags else None
        )
    with open(os.path.join(strategy_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=True)
    return summary


def compute_assignment_loss(model, dataloader):
    if dataloader is None or len(dataloader) == 0:
        return float("inf")
    model.eval()
    loss_sum, count, batches = 0.0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            count += int(y.size(0))
            batches += 1
    return (loss_sum / count if count else float("inf")), batches


def initial_balanced_tie_assignments(client_ids, seed):
    rng = np.random.default_rng(seed)
    shuffled = list(client_ids)
    rng.shuffle(shuffled)
    return {cid: i % NUM_CLUSTERS for i, cid in enumerate(shuffled)}


def run_fedavg_baseline(
    initial_state, test_loader, client_loaders, client_weights,
    cluster_client_ids, cluster_holdout_loaders, run_root,
    strategy="fedavg", proximal_mu=0.0, feddf_proxy_loader=None,
):
    """运行单全局模型 FedAvg/FedProx/FedDF，不是簇模型权重的事后合并。"""
    if strategy == "feddf" and (feddf_proxy_loader is None or len(feddf_proxy_loader) == 0):
        raise ValueError("FedDF requires the fixed unlabeled server proxy split")
    strategy_dir = os.path.join(run_root, strategy)
    os.makedirs(strategy_dir, exist_ok=True)
    model = ResNet18Classifier(num_classes=NUM_CLASSES, use_pretrained=False).to(DEVICE)
    model.load_state_dict(deepcopy(initial_state))
    train_weights = [
        sum(client_weights[c] for c in cluster_client_ids[k])
        for k in range(NUM_CLUSTERS)
    ]
    round_logs, cluster_logs, resource_logs = [], [], []
    model_bytes = model_state_nbytes(initial_state)
    set_seed(SEED)
    reset_training_loader_rng(client_loaders)
    start_time = time.time()

    for round_idx in range(1, NUM_ROUNDS + 1):
        round_start = time.time()
        states, weights = [], []
        local_steps = local_examples = 0
        feddf_stats = {
            "student_update_steps": 0,
            "examples_processed": 0,
            "teacher_forward_batches": 0,
            "mean_kd_loss": float("nan"),
        }
        for client_id, loader in enumerate(client_loaders):
            if loader is None or len(loader) == 0:
                continue
            local_model = deepcopy(model).to(DEVICE)
            state, _, steps, examples = local_train(
                local_model, loader, LOCAL_EPOCHS, proximal_mu=proximal_mu
            )
            local_steps += steps
            local_examples += examples
            states.append(state)
            weights.append(client_weights[client_id])
            del local_model
        if states:
            total = sum(weights)
            model.load_state_dict(fedavg_weighted_state(
                model.state_dict(), states, [weight / total for weight in weights]
            ))
            if strategy == "feddf":
                model, feddf_stats = feddf_server_distill(
                    model, states, feddf_proxy_loader,
                    epochs=DISTILL_EPOCHS, tau=DISTILL_TAU,
                    learning_rate=DISTILL_LR,
                )

        # The same global model is evaluated on each fixed cluster-held-out set.
        repeated_models = [model] * NUM_CLUSTERS
        rows, aggregate = evaluate_cluster_models(
            repeated_models, test_loader, cluster_holdout_loaders,
            cluster_client_ids, train_weights, round_idx, strategy,
            model_scope="global_model",
        )
        cluster_logs.extend(rows)
        global_metrics = {"acc": aggregate["mean_cluster_global_test_acc"]}
        round_logs.append({
            "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
            "beta": DIRICHLET_BETA, "seed": SEED, "round": round_idx,
            "strategy": strategy, **aggregate,
            "global_model_acc": global_metrics["acc"],
            "weighted_cluster_ensemble_acc": float("nan"),
            "round_time_sec": time.time() - round_start,
        })
        resource_logs.append({
            "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
            "beta": DIRICHLET_BETA, "seed": SEED, "round": round_idx,
            "strategy": strategy, "local_update_steps": local_steps,
            "local_examples_processed": local_examples,
            "extra_student_update_steps": feddf_stats["student_update_steps"],
            "extra_student_examples_processed": feddf_stats["examples_processed"],
            "distillation_student_uploads": 0,
            "distillation_teacher_model_downloads": 0,
            "assignment_forward_batches": 0,
            "teacher_forward_batches": feddf_stats["teacher_forward_batches"],
            "server_distill_kd_loss": feddf_stats["mean_kd_loss"],
            "estimated_uplink_bytes": model_bytes * len(states),
            "estimated_downlink_bytes": model_bytes * len(states),
            "round_time_sec": time.time() - round_start,
            "communication_measurement_kind": "logical estimate; no network transport",
        })
        if round_idx == 1 or round_idx % 10 == 0:
            print(f"{strategy} round {round_idx}/{NUM_ROUNDS}: global test={global_metrics['acc']:.2f}%, "
                  f"cluster-held-out macro={aggregate['mean_cluster_holdout_acc']:.2f}%")

    summary = write_method_outputs(
        strategy, strategy_dir, round_logs, cluster_logs, [], [],
        resource_logs=resource_logs, total_time_sec=time.time() - start_time,
    )
    return summary


def run_ifca_baseline(
    initial_state, test_loader, client_loaders, client_assignment_loaders,
    client_weights, client_holdout_indices, eval_dataset, run_root,
):
    """IFCA：每轮按客户端本地训练样本损失选择簇模型，再簇内聚合。"""
    strategy = "ifca"
    strategy_dir = os.path.join(run_root, strategy)
    os.makedirs(strategy_dir, exist_ok=True)
    models = []
    for _ in range(NUM_CLUSTERS):
        model = ResNet18Classifier(num_classes=NUM_CLASSES, use_pretrained=False).to(DEVICE)
        model.load_state_dict(deepcopy(initial_state))
        models.append(model)
    round_logs, cluster_logs, assignment_logs, resource_logs = [], [], [], []
    model_bytes = model_state_nbytes(initial_state)
    set_seed(SEED)
    reset_training_loader_rng(client_loaders)
    start_time = time.time()

    for round_idx in range(1, NUM_ROUNDS + 1):
        round_start = time.time()
        round_models = [deepcopy(model).to(DEVICE) for model in models]
        client_states = defaultdict(list)
        client_state_weights = defaultdict(list)
        assignments = np.full(NUM_CLIENTS, -1, dtype=np.int64)
        local_steps = local_examples = 0
        valid_clients = [
            cid for cid in range(NUM_CLIENTS)
            if client_loaders[cid] is not None and client_assignment_loaders[cid] is not None
            and len(client_loaders[cid]) > 0 and len(client_assignment_loaders[cid]) > 0
        ]
        balanced_ties = (
            initial_balanced_tie_assignments(valid_clients, SEED + 100000 + round_idx)
            if round_idx == 1 else {}
        )
        assignment_forward_batches = 0

        for client_id in valid_clients:
            losses = []
            for candidate_model in round_models:
                loss, batches = compute_assignment_loss(
                    candidate_model, client_assignment_loaders[client_id]
                )
                losses.append(loss)
                assignment_forward_batches += batches
            min_loss = min(losses)
            tied = [k for k, loss in enumerate(losses) if np.isclose(loss, min_loss, rtol=1e-7, atol=1e-8)]
            if len(tied) > 1 and round_idx == 1:
                selected = balanced_ties[client_id]
            elif len(tied) > 1:
                rng = np.random.default_rng(SEED + round_idx * 100000 + client_id)
                selected = int(rng.choice(tied))
            else:
                selected = int(np.argmin(losses))
            assignments[client_id] = selected
            assignment_logs.append({
                "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
                "beta": DIRICHLET_BETA, "seed": SEED, "round": round_idx,
                "client_id": client_id, "cluster_id": selected,
                "assignment_loss": float(losses[selected]),
                "candidate_losses": json.dumps([float(x) for x in losses]),
            })

            local_model = deepcopy(round_models[selected]).to(DEVICE)
            state, _, steps, examples = local_train(local_model, client_loaders[client_id], LOCAL_EPOCHS)
            local_steps += steps
            local_examples += examples
            client_states[selected].append(state)
            client_state_weights[selected].append(client_weights[client_id])
            del local_model

        groups = [[] for _ in range(NUM_CLUSTERS)]
        for cid, cluster_id in enumerate(assignments):
            if cluster_id >= 0:
                groups[int(cluster_id)].append(cid)
        cluster_train_weights = [
            sum(client_weights[cid] for cid in groups[k]) for k in range(NUM_CLUSTERS)
        ]
        for cluster_id in range(NUM_CLUSTERS):
            if not client_states[cluster_id]:
                continue
            weights = client_state_weights[cluster_id]
            total = sum(weights)
            models[cluster_id].load_state_dict(fedavg_weighted_state(
                models[cluster_id].state_dict(), client_states[cluster_id],
                [w / total for w in weights],
            ))

        holdout_loaders, groups = make_dynamic_cluster_eval_loaders(
            eval_dataset, client_holdout_indices, assignments
        )
        rows, aggregate = evaluate_cluster_models(
            models, test_loader, holdout_loaders, groups, cluster_train_weights,
            round_idx, strategy, model_scope="ifca_dynamic_cluster",
        )
        cluster_logs.extend(rows)
        ensemble = make_weighted_cluster_ensemble(models, cluster_train_weights, initial_state)
        ensemble_acc = (
            evaluate_metrics(ensemble, test_loader, NUM_CLASSES)["acc"]
            if ensemble is not None else float("nan")
        )
        round_logs.append({
            "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
            "beta": DIRICHLET_BETA, "seed": SEED, "round": round_idx,
            "strategy": strategy, **aggregate,
            "global_model_acc": float("nan"),
            "weighted_cluster_ensemble_acc": ensemble_acc,
            "round_time_sec": time.time() - round_start,
        })
        active_clients = len(valid_clients)
        resource_logs.append({
            "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
            "beta": DIRICHLET_BETA, "seed": SEED, "round": round_idx,
            "strategy": strategy, "local_update_steps": local_steps,
            "local_examples_processed": local_examples, "extra_student_update_steps": 0,
            "extra_student_examples_processed": 0,
            "distillation_student_uploads": 0,
            "distillation_teacher_model_downloads": 0,
            "assignment_forward_batches": assignment_forward_batches,
            "teacher_forward_batches": 0,
            "estimated_uplink_bytes": model_bytes * active_clients,
            "estimated_downlink_bytes": model_bytes * NUM_CLUSTERS * active_clients,
            "round_time_sec": time.time() - round_start,
            "communication_measurement_kind": "logical estimate; IFCA downloads K models for assignment; no network transport",
        })
        if round_idx == 1 or round_idx % 10 == 0:
            print(f"IFCA round {round_idx}/{NUM_ROUNDS}: active clusters={aggregate['active_clusters']}, "
                  f"cluster-held-out macro={aggregate['mean_cluster_holdout_acc']:.2f}%")
        del round_models

    final_assignment_rows = []
    for client_id, cluster_id in enumerate(assignments):
        final_assignment_rows.append({
            "experiment_group": EXPERIMENT_GROUP,
            "sensitivity_axis": SENSITIVITY_AXIS,
            "run_id": RUN_ID,
            "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
            "beta": DIRICHLET_BETA, "seed": SEED, "client_id": client_id,
            "cluster_id": int(cluster_id), "train_samples": int(client_weights[client_id]),
            "holdout_samples": len(client_holdout_indices[client_id]),
        })
    final_assignment_df = stamp_run_metadata(pd.DataFrame(final_assignment_rows))
    final_assignment_df.to_csv(
        os.path.join(strategy_dir, "final_assignments.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    summary = write_method_outputs(
        strategy, strategy_dir, round_logs, cluster_logs, [], [],
        assignment_logs=assignment_logs, resource_logs=resource_logs,
        total_time_sec=time.time() - start_time,
    )
    return summary

def run_one_strategy(
    strategy,
    initial_state,
    test_loader,
    client_loaders,
    client_assignment_loaders,
    client_weights,
    client_holdout_indices,
    eval_dataset,
    cluster_client_ids,
    cluster_validation_loaders,
    cluster_holdout_loaders,
    cluster_label_centers,
    cluster_prototypes,
    cluster_sample_weights,
    unlabeled_proxy_loader,
    run_root,
):
    print("\n" + "=" * 70)
    print(f"开始运行策略: {strategy}")
    print("=" * 70)

    if strategy in {"fedavg", "fedprox", "feddf"}:
        return run_fedavg_baseline(
            initial_state, test_loader, client_loaders, client_weights,
            cluster_client_ids, cluster_holdout_loaders, run_root,
            strategy=strategy,
            proximal_mu=FEDPROX_MU if strategy == "fedprox" else 0.0,
            feddf_proxy_loader=(
                unlabeled_proxy_loader if strategy == "feddf" else None
            ),
        )
    if strategy == "ifca":
        return run_ifca_baseline(
            initial_state, test_loader, client_loaders, client_assignment_loaders,
            client_weights, client_holdout_indices, eval_dataset, run_root,
        )

    strategy_dir = os.path.join(run_root, strategy)
    os.makedirs(strategy_dir, exist_ok=True)

    cluster_models = []
    for _ in range(NUM_CLUSTERS):
        model = ResNet18Classifier(num_classes=NUM_CLASSES, use_pretrained=False).to(DEVICE)
        model.load_state_dict(deepcopy(initial_state))
        cluster_models.append(model)
    set_seed(SEED)
    reset_training_loader_rng(client_loaders)

    candidate_logs = []
    event_logs = []
    cluster_logs = []
    resource_logs = []
    round_logs = []
    dktd_own_cluster_events = []
    dktd_original_contexts = {}
    if strategy == "full_dktd":
        for cluster_id in range(NUM_CLUSTERS):
            original_clients = [int(cid) for cid in cluster_client_ids[cluster_id]]
            expected_indices = [int(index) for cid in original_clients
                                for index in client_holdout_indices[cid]]
            loader = cluster_holdout_loaders[cluster_id]
            actual_indices = [int(index) for index in loader.dataset.indices] if loader is not None else []
            if actual_indices != expected_indices:
                raise ValueError(f"C{cluster_id}: holdout loader does not match original cluster membership")
            dktd_original_contexts[cluster_id] = {
                "original_client_ids": json.dumps(original_clients),
                "original_client_count": len(original_clients),
                "holdout_samples": len(expected_indices),
                "holdout_indices_sha256": hashlib.sha256(
                    json.dumps(expected_indices, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
    model_bytes = model_state_nbytes(initial_state)

    start_time = time.time()

    for round_idx in range(1, NUM_ROUNDS + 1):
        round_start = time.time()
        local_steps = local_examples = distill_steps = distill_examples = 0
        teacher_forward_batches = teacher_payload_models = active_client_updates = 0

        # ----------------------------------------------------
        # 1) 簇内 FedAvg 训练
        # ----------------------------------------------------
        for cluster_id in range(NUM_CLUSTERS):
            clients_in_cluster = cluster_client_ids[cluster_id]

            valid_clients = [
                cid for cid in clients_in_cluster
                if client_loaders[cid] is not None
                and client_weights[cid] > 0
                and len(client_loaders[cid]) > 0
            ]

            if len(valid_clients) == 0:
                continue

            client_states = []
            cluster_client_weights = []

            for cid in valid_clients:
                local_model = deepcopy(cluster_models[cluster_id]).to(DEVICE)
                state_dict, _, steps, examples = local_train(
                    local_model, client_loaders[cid], LOCAL_EPOCHS
                )
                local_steps += steps
                local_examples += examples
                active_client_updates += 1
                client_states.append(state_dict)
                cluster_client_weights.append(client_weights[cid])
                del local_model

            total_weight = sum(cluster_client_weights)
            if total_weight <= 0:
                continue

            normalized_weights = [w / total_weight for w in cluster_client_weights]
            new_state = fedavg_weighted_state(
                cluster_models[cluster_id].state_dict(), client_states, normalized_weights
            )
            cluster_models[cluster_id].load_state_dict(new_state)

        # ----------------------------------------------------
        # 2) 多教师蒸馏或学生更新量匹配的纯 CE 对照
        # ----------------------------------------------------
        distillation_student_uploads = 0

        if round_idx % DISTILL_INTERVAL == 0 and strategy == "dktd_step_matched":
            for target_id in range(NUM_CLUSTERS):
                valid_clients = [
                    cid for cid in cluster_client_ids[target_id]
                    if client_loaders[cid] is not None
                    and client_weights[cid] > 0
                    and len(client_loaders[cid]) > 0
                ]
                if not valid_clients:
                    continue
                event_start = time.time()
                before_global = evaluate_metrics(cluster_models[target_id], test_loader, NUM_CLASSES)
                before_holdout = evaluate_metrics(
                    cluster_models[target_id], cluster_holdout_loaders[target_id], NUM_CLASSES
                )
                control_states = []
                control_weights = []
                steps = examples = 0
                for cid in valid_clients:
                    control_model = deepcopy(cluster_models[target_id]).to(DEVICE)
                    client_state, _, client_steps, client_examples = local_train(
                        control_model, client_loaders[cid], DISTILL_EPOCHS,
                        learning_rate=DISTILL_LR,
                    )
                    control_states.append(deepcopy(client_state))
                    control_weights.append(client_weights[cid])
                    steps += client_steps
                    examples += client_examples
                    del control_model
                control_state = fedavg_weighted_state(
                    cluster_models[target_id].state_dict(), control_states, control_weights
                )
                cluster_models[target_id].load_state_dict(control_state)
                distill_steps += steps
                distill_examples += examples
                distillation_student_uploads += len(valid_clients)
                after_global = evaluate_metrics(cluster_models[target_id], test_loader, NUM_CLASSES)
                after_holdout = evaluate_metrics(
                    cluster_models[target_id], cluster_holdout_loaders[target_id], NUM_CLASSES
                )
                holdout_delta = after_holdout["acc"] - before_holdout["acc"]
                global_delta = after_global["acc"] - before_global["acc"]
                weak_before_metrics = before_holdout if np.isfinite(before_holdout["acc"]) else before_global
                weak_after_metrics = after_holdout if np.isfinite(after_holdout["acc"]) else after_global
                weak_before = compute_weak_recall(
                    weak_before_metrics["per_class_recall"], cluster_label_centers, target_id
                )
                weak_after = compute_weak_recall(
                    weak_after_metrics["per_class_recall"], cluster_label_centers, target_id
                )
                event_logs.append({
                    "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
                    "beta": DIRICHLET_BETA, "seed": SEED,
                    "event_id": f"{strategy}:r{round_idx}:c{target_id}",
                    "round": round_idx, "strategy": strategy,
                    "event_type": "ce_compute_match", "target_cluster": target_id,
                    "selected_teachers": "[]", "selected_teacher_count": 0,
                    "global_acc_before": before_global["acc"],
                    "global_acc_after": after_global["acc"], "global_acc_delta": global_delta,
                    "cluster_holdout_acc_before": before_holdout["acc"],
                    "cluster_holdout_acc_after": after_holdout["acc"],
                    "cluster_holdout_acc_delta": holdout_delta,
                    "negative_transfer_global": int(global_delta < -NEG_TRANSFER_EPS_PERCENT),
                    "negative_transfer_cluster_holdout": int(holdout_delta < -NEG_TRANSFER_EPS_PERCENT)
                    if np.isfinite(holdout_delta) else np.nan,
                    "weak_recall_before": weak_before, "weak_recall_after": weak_after,
                    "weak_recall_delta": weak_after - weak_before,
                    "student_update_steps": steps, "examples_processed": examples,
                    "teacher_forward_batches": 0, "event_time_sec": time.time() - event_start,
                })

        if round_idx % DISTILL_INTERVAL == 0 and strategy not in {
            "dktd_no_cross_distill", "dktd_step_matched"
        }:
            print(f"\n[蒸馏阶段] Round {round_idx}, strategy={strategy}")

            reliability_cache = {}
            for cid in range(NUM_CLUSTERS):
                val_loader = cluster_validation_loaders[cid]
                if val_loader is None or len(val_loader) == 0:
                    reliability_cache[cid] = 0.0
                else:
                    reliability_cache[cid] = compute_confidence(cluster_models[cid], val_loader)

            for target_id in range(NUM_CLUSTERS):
                if len(cluster_client_ids[target_id]) == 0:
                    continue
                valid_clients = [
                    cid for cid in cluster_client_ids[target_id]
                    if client_loaders[cid] is not None
                    and client_weights[cid] > 0
                    and len(client_loaders[cid]) > 0
                ]
                if not valid_clients:
                    continue

                event_start = time.time()
                before_metrics = evaluate_metrics(cluster_models[target_id], test_loader, NUM_CLASSES)
                before_holdout = evaluate_metrics(
                    cluster_models[target_id], cluster_holdout_loaders[target_id], NUM_CLASSES
                )
                weak_metrics_before = (
                    before_holdout if np.isfinite(before_holdout["acc"]) else before_metrics
                )
                weak_recall_before = compute_weak_recall(
                    weak_metrics_before["per_class_recall"], cluster_label_centers, target_id
                )

                teacher_ids, teacher_weights, score_rows = select_teachers(
                    strategy=strategy,
                    target_id=target_id,
                    round_idx=round_idx,
                    cluster_client_ids=cluster_client_ids,
                    client_loaders=client_loaders,
                    client_weights=client_weights,
                    cluster_label_centers=cluster_label_centers,
                    cluster_prototypes=cluster_prototypes,
                    reliability_cache=reliability_cache,
                )

                if len(teacher_ids) == 0:
                    continue

                print(
                    f"  Target C{target_id}: teachers={teacher_ids}, "
                    f"weights={[round(teacher_weights[t], 3) for t in teacher_ids]}"
                )

                teacher_models = [cluster_models[tid] for tid in teacher_ids]
                teacher_weight_list = [teacher_weights[tid] for tid in teacher_ids]

                client_student_states = []
                client_student_weights = []
                distill_stats = {
                    "student_update_steps": 0,
                    "examples_processed": 0,
                    "teacher_forward_batches": 0,
                }
                for cid in valid_clients:
                    local_student, local_stats = distill_cluster_multiteacher(
                        student_model=cluster_models[target_id],
                        teacher_models=teacher_models,
                        teacher_weights=teacher_weight_list,
                        dataloader=client_loaders[cid],
                        epochs=DISTILL_EPOCHS,
                        tau=DISTILL_TAU,
                        lambda_kd=KD_LAMBDA,
                        lr=DISTILL_LR,
                        return_stats=True,
                    )
                    client_student_states.append(deepcopy(local_student.state_dict()))
                    client_student_weights.append(client_weights[cid])
                    for key in distill_stats:
                        distill_stats[key] += local_stats[key]
                    del local_student

                aggregated_student_state = fedavg_weighted_state(
                    cluster_models[target_id].state_dict(),
                    client_student_states,
                    client_student_weights,
                )
                cluster_models[target_id].load_state_dict(aggregated_student_state)
                distill_steps += distill_stats["student_update_steps"]
                distill_examples += distill_stats["examples_processed"]
                teacher_forward_batches += distill_stats["teacher_forward_batches"]
                distillation_student_uploads += len(valid_clients)
                teacher_payload_models += len(teacher_ids) * len(valid_clients)

                after_metrics = evaluate_metrics(cluster_models[target_id], test_loader, NUM_CLASSES)
                after_holdout = evaluate_metrics(
                    cluster_models[target_id], cluster_holdout_loaders[target_id], NUM_CLASSES
                )
                weak_metrics_after = (
                    after_holdout if np.isfinite(after_holdout["acc"]) else after_metrics
                )
                weak_recall_after = compute_weak_recall(
                    weak_metrics_after["per_class_recall"], cluster_label_centers, target_id
                )
                global_delta = after_metrics["acc"] - before_metrics["acc"]
                holdout_delta = after_holdout["acc"] - before_holdout["acc"]
                event_id = f"{strategy}:r{round_idx}:c{target_id}"

                event_logs.append({
                    "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
                    "beta": DIRICHLET_BETA, "seed": SEED, "event_id": event_id,
                    "round": round_idx, "strategy": strategy,
                    "event_type": "multi_teacher_distillation", "target_cluster": target_id,
                    "selected_teachers": json.dumps(teacher_ids),
                    "selected_teacher_count": len(teacher_ids),
                    "global_acc_before": before_metrics["acc"],
                    "global_acc_after": after_metrics["acc"], "global_acc_delta": global_delta,
                    "cluster_holdout_acc_before": before_holdout["acc"],
                    "cluster_holdout_acc_after": after_holdout["acc"],
                    "cluster_holdout_acc_delta": holdout_delta,
                    "negative_transfer_global": int(global_delta < -NEG_TRANSFER_EPS_PERCENT),
                    "negative_transfer_cluster_holdout": int(
                        holdout_delta < -NEG_TRANSFER_EPS_PERCENT
                    ) if np.isfinite(holdout_delta) else np.nan,
                    "weak_recall_before": weak_recall_before,
                    "weak_recall_after": weak_recall_after,
                    "weak_recall_delta": weak_recall_after - weak_recall_before,
                    **distill_stats, "event_time_sec": time.time() - event_start,
                })

                if strategy == "full_dktd":
                    context = dktd_original_contexts[target_id]
                    eligible = context["holdout_samples"] > 0 and np.isfinite(holdout_delta)
                    recall_before = np.asarray(before_holdout["per_class_recall"], dtype=float)
                    recall_after = np.asarray(after_holdout["per_class_recall"], dtype=float)
                    dktd_own_cluster_events.append({
                        "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
                        "beta": DIRICHLET_BETA, "seed": SEED, "strategy": strategy,
                        "event_id": event_id, "round": round_idx,
                        "original_cluster_id": target_id, **context,
                        "selected_teachers": json.dumps(teacher_ids),
                        "evaluation_status": "evaluated" if eligible else "no_valid_original_holdout",
                        "is_final_round_pair": int(round_idx == NUM_ROUNDS),
                        "own_accuracy_before": before_holdout["acc"],
                        "own_accuracy_after": after_holdout["acc"],
                        "own_accuracy_delta_pp": holdout_delta,
                        "own_macro_f1_before": before_holdout["macro_f1"],
                        "own_macro_f1_after": after_holdout["macro_f1"],
                        "own_macro_f1_delta_pp": after_holdout["macro_f1"] - before_holdout["macro_f1"],
                        "own_macro_recall_before": float(np.mean(recall_before)),
                        "own_macro_recall_after": float(np.mean(recall_after)),
                        "own_macro_recall_delta_pp": float(np.mean(recall_after - recall_before)),
                        "own_per_class_recall_before": json.dumps([float(v) if np.isfinite(v) else None for v in recall_before]),
                        "own_per_class_recall_after": json.dumps([float(v) if np.isfinite(v) else None for v in recall_after]),
                        "global_accuracy_before": before_metrics["acc"],
                        "global_accuracy_after": after_metrics["acc"],
                        "global_accuracy_delta_pp": global_delta,
                        "declined_strictly": int(holdout_delta < 0) if eligible else None,
                        "declined_beyond_threshold": int(holdout_delta < -NEG_TRANSFER_EPS_PERCENT) if eligible else None,
                        "decline_threshold_pp": NEG_TRANSFER_EPS_PERCENT,
                    })
                    if eligible:
                        print(f"    Original C{target_id} holdout: "
                              f"M={before_holdout['acc']:.2f}% -> M'={after_holdout['acc']:.2f}%; "
                              f"delta={holdout_delta:+.2f} pp; "
                              f"declined={holdout_delta < 0}; "
                              f"drop>{NEG_TRANSFER_EPS_PERCENT:g} pp={holdout_delta < -NEG_TRANSFER_EPS_PERCENT}")
                    else:
                        print(f"    Original C{target_id} holdout: unavailable; excluded from paired decline rates")

                for row in score_rows:
                    log_row = {
                        "dataset": DATASET_NAME,
                        "K": NUM_CLUSTERS,
                        "M": TEACHER_M,
                        "beta": DIRICHLET_BETA,
                        "seed": SEED,
                        "event_id": event_id,
                        "round": round_idx,
                        "strategy": strategy,
                        "score_type": row.get("score_type", None),
                        "target_cluster": target_id,
                        "candidate_teacher": row["candidate_teacher"],
                        "Comp": row.get("Comp", None),
                        "Rel": row.get("Rel", None),
                        "Risk": row.get("Risk", None),
                        "Comp_norm": row.get("Comp_norm", None),
                        "Rel_norm": row.get("Rel_norm", None),
                        "Risk_norm": row.get("Risk_norm", None),
                        "Score": row.get("Score", None),
                        "selected": row.get("selected", 0),
                        "teacher_weight": row.get("teacher_weight", 0.0),
                    }
                    candidate_logs.append(log_row)

        # ----------------------------------------------------
        # 3) 逐簇评价；加权合并模型只作为辅助量，明确不称作 FedAvg。
        # ----------------------------------------------------
        rows, aggregate = evaluate_cluster_models(
            cluster_models, test_loader, cluster_holdout_loaders,
            cluster_client_ids, cluster_sample_weights, round_idx, strategy,
        )
        cluster_logs.extend(rows)
        ensemble = make_weighted_cluster_ensemble(
            cluster_models, cluster_sample_weights, initial_state
        )
        ensemble_acc = (
            evaluate_metrics(ensemble, test_loader, NUM_CLASSES)["acc"]
            if ensemble is not None else float("nan")
        )
        del ensemble
        round_time = time.time() - round_start
        round_logs.append({
            "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
            "beta": DIRICHLET_BETA, "seed": SEED, "round": round_idx,
            "strategy": strategy, **aggregate,
            "global_model_acc": float("nan"),
            "weighted_cluster_ensemble_acc": ensemble_acc,
            "round_time_sec": round_time,
        })
        resource_logs.append({
            "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
            "beta": DIRICHLET_BETA, "seed": SEED, "round": round_idx,
            "strategy": strategy, "local_update_steps": local_steps,
            "local_examples_processed": local_examples,
            "extra_student_update_steps": distill_steps,
            "extra_student_examples_processed": distill_examples,
            "distillation_student_uploads": distillation_student_uploads,
            "distillation_teacher_model_downloads": teacher_payload_models,
            "assignment_forward_batches": 0,
            "teacher_forward_batches": teacher_forward_batches,
            "estimated_uplink_bytes": model_bytes * (
                active_client_updates + distillation_student_uploads
            ),
            "estimated_downlink_bytes": model_bytes * (
                active_client_updates + distillation_student_uploads + teacher_payload_models
            ),
            "round_time_sec": round_time,
            "communication_measurement_kind": "logical estimate; includes one model per client update and selected teacher payloads; no network transport",
        })

        if round_idx == 1 or round_idx % 10 == 0 or round_idx % DISTILL_INTERVAL == 0:
            print(f"Round [{round_idx:03d}/{NUM_ROUNDS}] | strategy={strategy}")
            print(f"  Mean cluster-held-out Acc : {aggregate['mean_cluster_holdout_acc']:.2f}%")
            print(f"  Worst cluster-held-out Acc: {aggregate['worst_cluster_holdout_acc']:.2f}%")
            print(f"  Weighted cluster ensemble (global test): {ensemble_acc:.2f}%")

    total_time = time.time() - start_time
    summary = write_method_outputs(
        strategy, strategy_dir, round_logs, cluster_logs, event_logs,
        candidate_logs, resource_logs=resource_logs, total_time_sec=total_time,
    )
    if strategy == "full_dktd":
        own_report = write_dktd_original_cluster_report(
            strategy_dir, dktd_original_contexts, dktd_own_cluster_events, cluster_logs, round_logs,
        )
        final_pair = own_report["final_round_distillation_pair"]
        print("\nDKTD original-cluster evaluation:")
        print(f"  Final own-cluster mean accuracy: {own_report['final_model']['mean_own_accuracy']}")
        print(f"  Final-round paired clusters: {final_pair['eligible_original_clusters']}/"
              f"{final_pair['expected_evaluable_original_clusters']}")
        print(f"  Final-round paired mean delta (pp): {final_pair['mean_own_accuracy_delta_pp']}")
        print(f"  Final-round strict decline rate: {final_pair['strict_decline_rate']}")
        print(f"  Reports: {strategy_dir}/dktd_original_cluster_*.csv/json")

    print("\n策略运行完成：")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ============================================================
# 15. 主函数
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified CIFAR-10 runner for DKTD variants, IFCA, FedAvg, FedProx and FedDF."
    )
    parser.add_argument("--dataset", choices=["CIFAR-10", "cifar10"], default="CIFAR-10")
    parser.add_argument("--clusters", "--K", dest="clusters", type=int, default=NUM_CLUSTERS)
    parser.add_argument("--teachers", "--M", dest="teachers", type=int, default=TEACHER_M)
    parser.add_argument("--beta", type=float, default=DIRICHLET_BETA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument(
        "--experiment-group", choices=["main", "km_sensitivity"], default="main",
        help="实验批次；结果将写入独立目录并记录在每个日志文件中。",
    )
    parser.add_argument(
        "--sensitivity-axis", choices=["K", "M"], default=None,
        help="km_sensitivity 必填：K 扫描时固定 M；M 扫描时固定 K。",
    )
    parser.add_argument(
        "--methods", default=None,
        help="Comma-separated methods, or 'all'. Omit to use the group-specific default method list.",
    )
    parser.add_argument("--result-root", default=RESULT_ROOT)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--distill-interval", type=int, default=DISTILL_INTERVAL)
    parser.add_argument("--distill-epochs", type=int, default=DISTILL_EPOCHS)
    parser.add_argument("--proximal-mu", type=float, default=FEDPROX_MU)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pretrained",dest="pretrained",action="store_true",help="Use pretrained encoder")
    parser.add_argument("--no-pretrained",dest="pretrained",action="store_false",help="Do not use pretrained encoder")
    parser.set_defaults(pretrained=USE_PRETRAINED_ENCODER)
    #parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=USE_PRETRAINED_ENCODER)
    parser.add_argument("--aggregate", action="store_true", help="仅汇总 result-root 下已有的多 seed 结果")
    parser.add_argument(
        "--aggregate-group", choices=["all", "main", "km_sensitivity"], default="all",
        help="汇总全部实验组，或只汇总主实验/KM 敏感性实验。",
    )
    parser.add_argument("--reference-method", default="full_dktd", help="配对差值参考方法")
    args = parser.parse_args(argv)

    if not np.isfinite(args.beta) or args.beta <= 0:
        parser.error("--beta must be a finite number greater than 0")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not 2 <= args.clusters <= args.clients:
        parser.error("require 2 <= K <= number of clients")
    if not 1 <= args.teachers < args.clusters:
        parser.error("require 1 <= M < K")
    if args.rounds <= 0 or args.clients <= 0 or args.workers < 0:
        parser.error("rounds/clients must be positive and workers non-negative")
    if args.local_epochs <= 0 or args.batch_size <= 0 or args.distill_interval <= 0 or args.distill_epochs <= 0:
        parser.error("epochs, batch size and distillation interval must be positive")
    if not np.isfinite(args.proximal_mu) or args.proximal_mu < 0:
        parser.error("--proximal-mu must be finite and non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    if args.aggregate:
        return args

    if args.experiment_group == "km_sensitivity" and args.sensitivity_axis is None:
        parser.error("--experiment-group km_sensitivity requires --sensitivity-axis K or M")
    if args.experiment_group == "main" and args.sensitivity_axis is not None:
        parser.error("--sensitivity-axis is only valid for --experiment-group km_sensitivity")

    if args.methods is None:
        if args.experiment_group == "main":
            args.methods = RUN_STRATEGIES + [
                "dktd_step_matched", "ifca", "fedavg", "fedprox", "feddf"
            ]
        elif args.sensitivity_axis == "K":
            args.methods = RUN_STRATEGIES + ["ifca"]
        else:
            args.methods = [
                "dktd_no_cross_distill", "all_equal", "random_m", "similarity_topm",
                "dktd_topm_equal", "full_dktd",
            ]

    if isinstance(args.methods, str) and args.methods.strip().lower() == "all":
        methods = list(ALL_METHODS)
    elif isinstance(args.methods, str):
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    else:
        methods = list(args.methods)
    methods = [METHOD_ALIASES.get(item, item) for item in methods]
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        parser.error(f"unknown method(s): {', '.join(unknown)}; choices: {', '.join(ALL_METHODS)}")
    if not methods:
        parser.error("at least one method must be selected")
    args.methods = list(dict.fromkeys(methods))
    args.dataset = "CIFAR-10"
    return args


def current_git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def current_worktree_dirty():
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            check=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def current_script_sha256():
    try:
        digest = hashlib.sha256()
        with open(os.path.abspath(__file__), "rb") as script_file:
            for block in iter(lambda: script_file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception:
        return None


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


def bootstrap_mean_ci(values, seed=0, n_boot=10000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_sign_flip_pvalue(differences, seed=0, n_perm=20000):
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return float("nan")
    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(differences)))
    permuted = np.abs((signs * differences).mean(axis=1))
    return float((1 + np.sum(permuted >= observed)) / (n_perm + 1))


def write_cross_seed_convergence_outputs(curve_records, aggregate_dir):
    """Write per-round seed means/SDs and convergence plots for Figure 3."""
    if not curve_records:
        return 0, 0

    frame = pd.DataFrame(curve_records)
    group_columns = [
        "experiment_group", "sensitivity_axis", "dataset", "K", "M", "beta",
        "strategy", "metric", "round", "script_sha256",
    ]
    curve_summary = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            seeds=("seed", lambda values: ",".join(
                str(int(value)) for value in sorted(set(values))
            )),
            mean=("value", "mean"),
            std=("value", "std"),
        )
        .reset_index()
    )
    curve_summary.to_csv(
        os.path.join(aggregate_dir, "cross_seed_convergence.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    figure_dir = os.path.join(aggregate_dir, "convergence_figures")
    os.makedirs(figure_dir, exist_ok=True)
    config_columns = [
        "experiment_group", "sensitivity_axis", "dataset", "K", "M", "beta",
        "metric", "script_sha256",
    ]
    figures_written = 0
    for config_key, config_frame in curve_summary.groupby(config_columns, dropna=False):
        group, axis, dataset, clusters, teachers, beta, metric, script_hash = config_key
        plotted = config_frame[
            config_frame["strategy"].isin(CONVERGENCE_CURVE_METHODS)
        ]
        if plotted.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for method in CONVERGENCE_CURVE_METHODS:
            method_frame = plotted[plotted["strategy"] == method].sort_values("round")
            if method_frame.empty:
                continue
            rounds = method_frame["round"].to_numpy(dtype=float)
            means = method_frame["mean"].to_numpy(dtype=float)
            stds = method_frame["std"].to_numpy(dtype=float)
            minimum_seed_count = int(method_frame["n_seeds"].min())
            display_name = CONVERGENCE_CURVE_LABELS.get(method, method)
            ax.plot(
                rounds, means, linewidth=2,
                label=f"{display_name} (n≥{minimum_seed_count})",
            )
            valid = np.isfinite(stds) & (method_frame["n_seeds"].to_numpy() >= 2)
            if valid.any():
                ax.fill_between(
                    rounds[valid], means[valid] - stds[valid], means[valid] + stds[valid],
                    alpha=0.16,
                )

        metric_label = (
            "Average Cluster Accuracy on Global Test Set (%)"
            if metric == "mean_cluster_global_test_acc"
            else "Mean Cluster-Held-Out Accuracy (%)"
        )
        ax.set_xlabel("Communication Rounds")
        ax.set_ylabel(metric_label)
        ax.set_title(
            f"{group}/{axis} | {dataset}, K={clusters}, M={teachers}, beta={beta:g}"
        )
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        metric_slug = (
            "global_test_average_cluster_accuracy"
            if metric == "mean_cluster_global_test_acc"
            else "cluster_holdout_accuracy"
        )
        safe_hash = str(script_hash)[:8] if script_hash else "nohash"
        filename = (
            f"convergence_{group}_{axis}_{dataset}_K{clusters}_M{teachers}_"
            f"beta{beta:g}_{metric_slug}_{safe_hash}.png"
        )
        fig.savefig(os.path.join(figure_dir, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures_written += 1

    return len(curve_summary), figures_written


def aggregate_seed_runs(result_root, reference_method="full_dktd", group_filter="all"):
    """汇总主实验及 K/M 扫描，并按共同 seed/数据划分计算配对差值。"""
    reference_method = METHOD_ALIASES.get(reference_method, reference_method)
    metric_names = [
        "mean_cluster_holdout_acc", "worst_cluster_holdout_acc",
        "mean_cluster_global_test_acc", "global_model_acc",
        "weighted_cluster_ensemble_acc",
    ] + list(STABILITY_METRIC_SOURCES)
    per_seed, duplicate_paths = {}, defaultdict(list)
    curve_records = []
    config_paths = []
    for root, dirs, files in os.walk(result_root):
        dirs.sort()
        if "run_config.json" in files:
            config_paths.append(os.path.join(root, "run_config.json"))

    for config_path in sorted(config_paths):
        run_dir = os.path.dirname(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Older completed runs may not have a status field. Explicitly
        # incomplete/failed runs must never enter cross-seed statistics.
        if config.get("status", "completed") != "completed":
            continue
        experiment_group = config.get("experiment_group", "legacy")
        sensitivity_axis = config.get("sensitivity_axis") or "not_applicable"
        include_as_km_anchor = (
            group_filter == "km_sensitivity" and experiment_group == "main"
        )
        if group_filter != "all" and experiment_group != group_filter and not include_as_km_anchor:
            continue
        group = (
            experiment_group, sensitivity_axis, config.get("dataset"),
            int(config.get("K")), int(config.get("M")), float(config.get("beta")),
        )
        seed = int(config.get("seed"))
        split_info = config.get("split") or {}
        partition_hash = split_info.get(
            "client_data_partition_sha256",
            split_info.get("partition_sha256", config.get("partition_sha256", "")),
        )
        for method in config.get("methods", []):
            metrics_path = os.path.join(run_dir, method, "round_metrics.csv")
            if not os.path.isfile(metrics_path):
                continue
            frame = pd.read_csv(metrics_path)
            if frame.empty:
                continue
            expected_rounds = int(config.get("rounds", 0))
            if expected_rounds > 0 and (
                frame["round"].nunique() < expected_rounds
                or int(frame["round"].max()) < expected_rounds
            ):
                continue
            ordered_frame = frame.sort_values("round")
            tail = ordered_frame.tail(min(10, len(ordered_frame)))
            stability_tail = ordered_frame.tail(STABILITY_WINDOW_ROUNDS)
            stability_round_count = (
                int(stability_tail["round"].nunique()) if len(stability_tail) else 0
            )
            record = {
                "experiment_group": group[0], "sensitivity_axis": group[1],
                "dataset": group[2], "K": group[3], "M": group[4], "beta": group[5],
                "seed": seed, "strategy": method, "partition_sha256": partition_hash,
                "script_sha256": config.get("script_sha256", ""),
                "run_id": config.get("run_id", os.path.basename(run_dir)),
                "final_window_start_round": int(tail["round"].min()),
                "final_window_end_round": int(tail["round"].max()),
                "stability_window_start_round": (
                    int(stability_tail["round"].min()) if stability_round_count else None
                ),
                "stability_window_end_round": (
                    int(stability_tail["round"].max()) if stability_round_count else None
                ),
                "stability_window_rounds_observed": stability_round_count,
            }
            for metric in metric_names:
                if metric in STABILITY_METRIC_SOURCES:
                    source_metric = STABILITY_METRIC_SOURCES[metric]
                    values = (
                        pd.to_numeric(stability_tail[source_metric], errors="coerce").dropna()
                        if source_metric in stability_tail else pd.Series(dtype=float)
                    )
                    record[metric] = (
                        float(values.std(ddof=1))
                        if stability_round_count == STABILITY_WINDOW_ROUNDS
                        and len(values) == STABILITY_WINDOW_ROUNDS
                        else float("nan")
                    )
                else:
                    record[metric] = (
                        float(pd.to_numeric(tail[metric], errors="coerce").mean())
                        if metric in tail else float("nan")
                    )
            key = (group, method, seed)
            # The K/M sensitivity aggregate may read main runs as shared anchors;
            # do not duplicate those anchors as sensitivity convergence figures.
            include_curve = not (
                group_filter == "km_sensitivity" and experiment_group == "main"
            )
            if include_curve:
                for metric in CONVERGENCE_CURVE_METRICS:
                    if metric not in frame:
                        continue
                    values = pd.to_numeric(frame[metric], errors="coerce")
                    for round_value, value in zip(frame["round"], values):
                        if not np.isfinite(value):
                            continue
                        curve_records.append({
                            "experiment_group": group[0],
                            "sensitivity_axis": group[1],
                            "dataset": group[2], "K": group[3], "M": group[4],
                            "beta": group[5], "strategy": method,
                            "metric": metric, "round": int(round_value),
                            "seed": seed, "value": float(value),
                            "script_sha256": config.get("script_sha256", ""),
                            "run_id": config.get("run_id", os.path.basename(run_dir)),
                        })
            if key in per_seed:
                duplicate_paths[key].extend([per_seed[key]["source_run_dir"], run_dir])
            else:
                record["source_run_dir"] = run_dir
                per_seed[key] = record

    if duplicate_paths:
        examples = list(duplicate_paths.items())[:5]
        details = "\n".join(
            f"{key}: {sorted(set(paths))}" for key, paths in examples
        )
        raise ValueError(
            "发现同一实验组/配置/方法/seed 的多个完整运行。为避免静默选错结果，"
            "请先移走或明确整理重复目录：\n" + details
        )
    has_requested_group = (
        group_filter == "all"
        or any(record["experiment_group"] == group_filter for record in per_seed.values())
    )
    if not per_seed or not has_requested_group:
        raise FileNotFoundError(
            f"No completed runs for aggregate-group={group_filter} under {result_root}"
        )

    grouped = defaultdict(list)
    for record in per_seed.values():
        if group_filter == "km_sensitivity" and record["experiment_group"] != "km_sensitivity":
            continue
        for metric in metric_names:
            grouped[(record["experiment_group"], record["sensitivity_axis"],
                     record["dataset"], record["K"], record["M"], record["beta"],
                     record["strategy"], metric)].append(record)
    summary_rows = []
    for key, records in sorted(grouped.items(), key=lambda item: str(item[0])):
        metric = key[-1]
        finite_records = [r for r in records if np.isfinite(r[metric])]
        values = np.asarray([r[metric] for r in finite_records], dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(values)
        summary_rows.append({
            "experiment_group": key[0], "sensitivity_axis": key[1],
            "dataset": key[2], "K": key[3], "M": key[4], "beta": key[5],
            "strategy": key[6], "metric": metric, "n_seeds": len(values),
            "seeds": ",".join(str(r["seed"]) for r in finite_records),
            "within_run_window_rounds": (
                STABILITY_WINDOW_ROUNDS if metric in STABILITY_METRIC_SOURCES else min(
                    10, int(records[0].get("final_window_end_round", 0))
                )
            ),
            "aggregation_statistic": (
                "within-seed sample SD over final 20 rounds"
                if metric in STABILITY_METRIC_SOURCES
                else "within-seed mean over final 10 rounds"
            ),
            "mean": float(values.mean()) if len(values) else float("nan"),
            "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high,
        })

    # 方法间配对：只比较同一实验组、同一 K/M/beta 下共享 seed 的方法。
    pair_rows = []
    groups = sorted({
        key[0] for key in per_seed
        if group_filter == "all" or key[0][0] == group_filter
    })
    methods = sorted({key[1] for key in per_seed})
    for group in groups:
        for method in methods:
            if method == reference_method:
                continue
            for metric in metric_names:
                differences, paired_seeds, mismatched_splits, mismatched_scripts = [], [], 0, 0
                for (candidate_group, candidate_method, seed), candidate in per_seed.items():
                    if candidate_group != group or candidate_method != method:
                        continue
                    reference = per_seed.get((group, reference_method, seed))
                    if reference is None:
                        continue
                    if (
                        candidate["partition_sha256"]
                        and reference["partition_sha256"]
                        and candidate["partition_sha256"] != reference["partition_sha256"]
                    ):
                        mismatched_splits += 1
                        continue
                    if (
                        candidate["script_sha256"]
                        and reference["script_sha256"]
                        and candidate["script_sha256"] != reference["script_sha256"]
                    ):
                        mismatched_scripts += 1
                        continue
                    value, ref_value = candidate[metric], reference[metric]
                    if np.isfinite(value) and np.isfinite(ref_value):
                        differences.append(value - ref_value)
                        paired_seeds.append(seed)
                if not differences:
                    continue
                pair_token = (group, method, reference_method, metric)
                bootstrap_seed = int.from_bytes(
                    hashlib.sha256(str(pair_token).encode()).digest()[:4], "big"
                )
                ci_low, ci_high = bootstrap_mean_ci(differences, seed=bootstrap_seed)
                pair_rows.append({
                    "experiment_group": group[0], "sensitivity_axis": group[1],
                    "dataset": group[2], "K": group[3], "M": group[4], "beta": group[5],
                    "method": method, "reference_method": reference_method,
                    "metric": metric, "n_paired_seeds": len(differences),
                    "paired_seeds": ",".join(str(x) for x in paired_seeds),
                    "partition_hash_mismatch_count": mismatched_splits,
                    "script_hash_mismatch_count": mismatched_scripts,
                    "mean_paired_difference": float(np.mean(differences)),
                    "paired_difference_std": float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan"),
                    "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high,
                    "p_sign_flip": paired_sign_flip_pvalue(differences, seed=bootstrap_seed),
                    "p_holm": float("nan"),
                })

    pair_groups = defaultdict(list)
    for index, row in enumerate(pair_rows):
        pair_groups[(row["experiment_group"], row["sensitivity_axis"], row["dataset"],
                     row["K"], row["M"], row["beta"], row["metric"])].append(index)
    for indices in pair_groups.values():
        ordered = sorted(
            (i for i in indices if np.isfinite(pair_rows[i]["p_sign_flip"])),
            key=lambda i: pair_rows[i]["p_sign_flip"],
        )
        running_max, count = 0.0, len(ordered)
        for rank, row_index in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * pair_rows[row_index]["p_sign_flip"])
            running_max = max(running_max, adjusted)
            pair_rows[row_index]["p_holm"] = running_max

    # K/M 敏感性：在同一 beta、同一 method、同一 seed 和同一客户端数据划分下，
    # 两两比较只改变 K 或只改变 M 的条件。
    km_families = defaultdict(lambda: defaultdict(dict))
    for record in per_seed.values():
        axis = record["sensitivity_axis"]
        if record["experiment_group"] != "km_sensitivity" or axis not in ("K", "M"):
            continue
        for metric in metric_names:
            value = record[metric]
            if not np.isfinite(value):
                continue
            if axis == "K":
                family = (record["dataset"], record["beta"], axis, record["strategy"], record["M"], metric)
                condition = record["K"]
            else:
                family = (record["dataset"], record["beta"], axis, record["strategy"], record["K"], metric)
                condition = record["M"]
            km_families[family][condition][record["seed"]] = record

    # 主实验若恰好采用了扫描中的 K/M，可作为该扫描的共享锚点，避免重复训练。
    # 它只参与明确匹配的敏感性配对；仍保留在主实验独立汇总中。
    for family, conditions in km_families.items():
        dataset, beta, axis, method, fixed_value, metric = family
        for record in per_seed.values():
            if (
                record["experiment_group"] != "main"
                or record["dataset"] != dataset
                or record["beta"] != beta
                or record["strategy"] != method
                or not np.isfinite(record[metric])
            ):
                continue
            if axis == "K" and record["M"] == fixed_value:
                condition = record["K"]
            elif axis == "M" and record["K"] == fixed_value:
                condition = record["M"]
            else:
                continue
            existing = conditions[condition].get(record["seed"])
            if existing is not None:
                partition_conflict = (
                    existing["partition_sha256"] and record["partition_sha256"]
                    and existing["partition_sha256"] != record["partition_sha256"]
                )
                script_conflict = (
                    existing["script_sha256"] and record["script_sha256"]
                    and existing["script_sha256"] != record["script_sha256"]
                )
                if partition_conflict or script_conflict:
                    raise ValueError(
                        "主实验与 K/M 锚点的划分或代码版本不一致："
                        f"{dataset=}, {beta=}, {axis=}, {method=}, {condition=}, {record['seed']=}"
                    )
            else:
                conditions[condition][record["seed"]] = record

    km_pair_rows = []
    for family, conditions in sorted(km_families.items(), key=lambda item: str(item[0])):
        dataset, beta, axis, method, fixed_value, metric = family
        condition_values = sorted(conditions)
        for index_a in range(len(condition_values)):
            for index_b in range(index_a + 1, len(condition_values)):
                condition_a, condition_b = condition_values[index_a], condition_values[index_b]
                runs_a, runs_b = conditions[condition_a], conditions[condition_b]
                paired_seeds, differences, mismatched_splits, mismatched_scripts = [], [], 0, 0
                for seed in sorted(set(runs_a) & set(runs_b)):
                    run_a, run_b = runs_a[seed], runs_b[seed]
                    hash_a, hash_b = run_a["partition_sha256"], run_b["partition_sha256"]
                    if hash_a and hash_b and hash_a != hash_b:
                        mismatched_splits += 1
                        continue
                    script_a, script_b = run_a["script_sha256"], run_b["script_sha256"]
                    if script_a and script_b and script_a != script_b:
                        mismatched_scripts += 1
                        continue
                    differences.append(run_b[metric] - run_a[metric])
                    paired_seeds.append(seed)
                if not differences:
                    continue
                pair_token = (family, condition_a, condition_b)
                bootstrap_seed = int.from_bytes(
                    hashlib.sha256(str(pair_token).encode()).digest()[:4], "big"
                )
                ci_low, ci_high = bootstrap_mean_ci(differences, seed=bootstrap_seed)
                row = {
                    "experiment_group": "km_sensitivity", "sensitivity_axis": axis,
                    "dataset": dataset, "beta": beta, "method": method, "metric": metric,
                    "source_groups_a": ",".join(sorted({runs_a[s]["experiment_group"] for s in paired_seeds})),
                    "source_groups_b": ",".join(sorted({runs_b[s]["experiment_group"] for s in paired_seeds})),
                    "run_ids_a": ";".join(runs_a[s]["run_id"] for s in paired_seeds),
                    "run_ids_b": ";".join(runs_b[s]["run_id"] for s in paired_seeds),
                    "fixed_K": fixed_value if axis == "M" else None,
                    "fixed_M": fixed_value if axis == "K" else None,
                    "condition_a": condition_a, "condition_b": condition_b,
                    "K_a": condition_a if axis == "K" else fixed_value,
                    "M_a": fixed_value if axis == "K" else condition_a,
                    "K_b": condition_b if axis == "K" else fixed_value,
                    "M_b": fixed_value if axis == "K" else condition_b,
                    "n_paired_seeds": len(differences),
                    "paired_seeds": ",".join(str(seed) for seed in paired_seeds),
                    "partition_hash_mismatch_count": mismatched_splits,
                    "script_hash_mismatch_count": mismatched_scripts,
                    "mean_paired_difference_b_minus_a": float(np.mean(differences)),
                    "paired_difference_std": float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan"),
                    "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high,
                    "p_sign_flip": paired_sign_flip_pvalue(differences, seed=bootstrap_seed),
                    "p_holm": float("nan"),
                }
                km_pair_rows.append(row)

    km_correction_families = defaultdict(list)
    for index, row in enumerate(km_pair_rows):
        km_correction_families[(row["dataset"], row["beta"], row["sensitivity_axis"],
                                row["method"], row["fixed_K"], row["fixed_M"], row["metric"])].append(index)
    for indices in km_correction_families.values():
        ordered = sorted(
            (i for i in indices if np.isfinite(km_pair_rows[i]["p_sign_flip"])),
            key=lambda i: km_pair_rows[i]["p_sign_flip"],
        )
        running_max, count = 0.0, len(ordered)
        for rank, row_index in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * km_pair_rows[row_index]["p_sign_flip"])
            running_max = max(running_max, adjusted)
            km_pair_rows[row_index]["p_holm"] = running_max

    aggregate_dir = os.path.join(result_root, "_aggregates", group_filter)
    os.makedirs(aggregate_dir, exist_ok=True)
    stability_rows = []
    for record in per_seed.values():
        if group_filter == "km_sensitivity" and record["experiment_group"] != "km_sensitivity":
            continue
        for metric in STABILITY_METRIC_SOURCES:
            stability_rows.append({
                "experiment_group": record["experiment_group"],
                "sensitivity_axis": record["sensitivity_axis"],
                "dataset": record["dataset"], "K": record["K"], "M": record["M"],
                "beta": record["beta"], "strategy": record["strategy"],
                "seed": record["seed"], "metric": metric,
                "window_rounds_required": STABILITY_WINDOW_ROUNDS,
                "window_rounds_observed": record["stability_window_rounds_observed"],
                "window_start_round": record["stability_window_start_round"],
                "window_end_round": record["stability_window_end_round"],
                "temporal_sd_percentage_points": record[metric],
                "run_id": record["run_id"],
                "partition_sha256": record["partition_sha256"],
                "script_sha256": record["script_sha256"],
            })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(aggregate_dir, "cross_seed_summary.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(stability_rows).to_csv(
        os.path.join(aggregate_dir, "per_seed_last20_stability.csv"),
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(aggregate_dir, "paired_seed_differences.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(km_pair_rows).to_csv(
        os.path.join(aggregate_dir, "paired_km_sensitivity.csv"), index=False, encoding="utf-8-sig"
    )
    curve_rows, convergence_figures = write_cross_seed_convergence_outputs(
        curve_records, aggregate_dir
    )
    print(
        f"汇总完成：{aggregate_dir} | group={group_filter} | "
        f"seed 汇总={len(summary_rows)} 行，方法配对={len(pair_rows)} 行，"
        f"K/M 配对={len(km_pair_rows)} 行，逐轮曲线={curve_rows} 行/图={convergence_figures} 张"
    )
    return summary_rows, pair_rows, km_pair_rows


def main(argv=None):
    global DATASET_NAME, DATA_ROOT, DIRICHLET_BETA, NUM_CLIENTS, NUM_CLUSTERS
    global NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, SEED, NUM_WORKERS, DEVICE
    global TEACHER_M, DISTILL_INTERVAL, DISTILL_EPOCHS, RESULT_ROOT, USE_PRETRAINED_ENCODER
    global PRETRAINED_LOAD_STATUS, FEDPROX_MU
    global EXPERIMENT_GROUP, SENSITIVITY_AXIS, RUN_ID

    args = parse_args(argv)
    if args.aggregate:
        aggregate_seed_runs(args.result_root, args.reference_method, args.aggregate_group)
        return args.result_root
    DATASET_NAME = args.dataset
    DATA_ROOT = args.data_root
    DIRICHLET_BETA = args.beta
    NUM_CLIENTS = args.clients
    NUM_CLUSTERS = args.clusters
    NUM_ROUNDS = args.rounds
    LOCAL_EPOCHS = args.local_epochs
    BATCH_SIZE = args.batch_size
    SEED = args.seed
    NUM_WORKERS = args.workers
    TEACHER_M = args.teachers
    DISTILL_INTERVAL = args.distill_interval
    DISTILL_EPOCHS = args.distill_epochs
    FEDPROX_MU = args.proximal_mu
    RESULT_ROOT = args.result_root
    EXPERIMENT_GROUP = args.experiment_group
    SENSITIVITY_AXIS = args.sensitivity_axis or "not_applicable"
    USE_PRETRAINED_ENCODER = args.pretrained
    PRETRAINED_LOAD_STATUS = "pending" if USE_PRETRAINED_ENCODER else "not_requested"
    DEVICE = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    pipeline_start = time.time()
    set_seed(SEED)
    os.makedirs(RESULT_ROOT, exist_ok=True)
    dataset_slug = "cifar10"
    run_name = f"{dataset_slug}_K{NUM_CLUSTERS}_M{TEACHER_M}_beta{DIRICHLET_BETA:g}_seed{SEED}"
    if EXPERIMENT_GROUP == "km_sensitivity":
        group_root = os.path.join(RESULT_ROOT, EXPERIMENT_GROUP, SENSITIVITY_AXIS)
    else:
        group_root = os.path.join(RESULT_ROOT, EXPERIMENT_GROUP)
    os.makedirs(group_root, exist_ok=True)
    run_root = os.path.join(group_root, run_name)
    if os.path.isdir(run_root) and os.listdir(run_root):
        run_root = f"{run_root}_repeat{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}"
    RUN_ID = os.path.relpath(run_root, RESULT_ROOT).replace(os.sep, "/")
    os.makedirs(run_root, exist_ok=True)
    with open(os.path.join(run_root, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "status": "preprocessing_started",
            "experiment_group": EXPERIMENT_GROUP,
            "sensitivity_axis": SENSITIVITY_AXIS,
            "run_id": RUN_ID,
            "dataset": DATASET_NAME, "K": NUM_CLUSTERS, "M": TEACHER_M,
            "beta": DIRICHLET_BETA, "seed": SEED, "rounds": NUM_ROUNDS,
            "methods": args.methods,
        }, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print(f"设备: {DEVICE} | 数据集: {DATASET_NAME} | 客户端: {NUM_CLIENTS}")
    print(f"K={NUM_CLUSTERS}, M={TEACHER_M}, beta={DIRICHLET_BETA}, seed={SEED}, rounds={NUM_ROUNDS}")
    print(f"实验组: {EXPERIMENT_GROUP} | 敏感性轴: {SENSITIVITY_AXIS}")
    print(f"方法: {args.methods}")
    print(f"结果目录: {run_root}")
    print("=" * 72)

    stage_times = {}
    stage_start = time.time()
    print("加载 CIFAR-10 数据集...")
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train
    )
    eval_trainset = copy(trainset)
    eval_trainset.transform = transform_test
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_test
    )
    labels = np.asarray(trainset.targets, dtype=np.int64)
    stage_times["dataset_loading_sec"] = time.time() - stage_start
    print(f"训练集大小: {len(trainset)} | 官方全局测试集大小: {len(testset)}")

    stage_start = time.time()
    original_client_indices = dirichlet_split_noniid(
        labels, NUM_CLIENTS, NUM_CLASSES, DIRICHLET_BETA, seed=SEED
    )
    client_train_indices, client_validation_indices, client_holdout_indices = (
        split_client_indices_for_validation_and_holdout(
            original_client_indices, labels, SEED,
            validation_fraction=VAL_FRACTION,
            holdout_fraction=CLUSTER_HOLDOUT_FRACTION,
        )
    )
    verify_client_partitions(
        original_client_indices, client_train_indices, client_validation_indices,
        client_holdout_indices, len(trainset),
    )
    partition_hash = save_client_partition_artifacts(
        run_root, original_client_indices, client_train_indices,
        client_validation_indices, client_holdout_indices, labels,
        np.full(NUM_CLIENTS, -1, dtype=np.int64),
    )
    stage_times["partitioning_sec"] = time.time() - stage_start

    stage_start = time.time()
    print("构建共享 encoder 并提取训练集统计信息...")
    base_model = ResNet18Classifier(
        num_classes=NUM_CLASSES, use_pretrained=USE_PRETRAINED_ENCODER
    ).to(DEVICE)
    client_histograms, client_prototypes, client_feature_histograms = extract_client_statistics(
        client_indices=client_train_indices,
        dataset_data=trainset.data,
        labels=labels,
        num_classes=NUM_CLASSES,
        encoder=base_model.backbone,
        batch_size=ENCODER_BATCH_SIZE,
    )
    stage_times["statistics_extraction_sec"] = time.time() - stage_start

    stage_start = time.time()
    combined_distance_matrix = build_combined_distance_matrix(
        label_histograms=client_histograms,
        client_prototypes=client_prototypes,
        feature_histograms=client_feature_histograms,
        lambda_label=LAMBDA_LABEL,
        lambda_feat=LAMBDA_FEAT,
        alpha=PROTO_HIST_ALPHA,
    )
    cluster_labels, medoid_indices = kmedoids_precomputed(
        distance_matrix=combined_distance_matrix,
        num_clusters=NUM_CLUSTERS,
        random_state=SEED,
    )
    cluster_label_centers, _, cluster_prototypes = compute_cluster_centers(
        label_histograms=client_histograms,
        feature_histograms=client_feature_histograms,
        client_prototypes=client_prototypes,
        cluster_labels=cluster_labels,
        num_clusters=NUM_CLUSTERS,
    )
    cluster_client_ids = defaultdict(list)
    for client_id, cluster_id in enumerate(cluster_labels):
        cluster_client_ids[int(cluster_id)].append(client_id)
    stage_times["clustering_sec"] = time.time() - stage_start
    cluster_assignment_hash = hashlib.sha256(
        json.dumps([int(x) for x in cluster_labels], separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    partition_hash = save_client_partition_artifacts(
        run_root, original_client_indices, client_train_indices,
        client_validation_indices, client_holdout_indices, labels, cluster_labels,
    )
    cluster_summary_rows = []
    for cluster_id in range(NUM_CLUSTERS):
        clients = cluster_client_ids[cluster_id]
        row = {
            "experiment_group": EXPERIMENT_GROUP,
            "sensitivity_axis": SENSITIVITY_AXIS,
            "run_id": RUN_ID,
            "dataset": DATASET_NAME,
            "K": NUM_CLUSTERS,
            "M": TEACHER_M,
            "beta": DIRICHLET_BETA,
            "seed": SEED,
            "cluster_id": cluster_id,
            "medoid_client_id": int(medoid_indices[cluster_id]),
            "num_clients": len(clients),
            "train_samples": int(sum(len(client_train_indices[c]) for c in clients)),
            "validation_samples": int(sum(len(client_validation_indices[c]) for c in clients)),
            "holdout_samples": int(sum(len(client_holdout_indices[c]) for c in clients)),
        }
        for class_id in range(NUM_CLASSES):
            row[f"train_class_{class_id}"] = int(sum(
                np.sum(labels[np.asarray(client_train_indices[c], dtype=np.int64)] == class_id)
                for c in clients
            ))
        cluster_summary_rows.append(row)
    cluster_summary_df = stamp_run_metadata(pd.DataFrame(cluster_summary_rows))
    cluster_summary_df.to_csv(
        os.path.join(run_root, "cluster_summary.csv"), index=False, encoding="utf-8-sig"
    )
    save_cluster_analysis_figure(
        client_histograms, cluster_labels, combined_distance_matrix,
        os.path.join(run_root, "cluster_analysis.png"),
    )

    stage_start = time.time()
    client_loaders, client_weights = make_client_loaders(trainset, client_train_indices)
    client_assignment_loaders = make_client_eval_loaders(eval_trainset, client_train_indices)
    cluster_validation_loaders = make_cluster_eval_loaders(
        eval_trainset, client_validation_indices, cluster_client_ids
    )
    cluster_holdout_loaders = make_cluster_eval_loaders(
        eval_trainset, client_holdout_indices, cluster_client_ids
    )
    # Keep this proxy fixed for the whole run: the same client validation
    # images/order are pooled at the server, with labels hidden by the wrapper.
    server_proxy_indices = [
        int(index)
        for client_indices in client_validation_indices
        for index in client_indices
    ]
    server_proxy_indices_sha256 = hashlib.sha256(
        json.dumps(server_proxy_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    unlabeled_proxy_loader = make_unlabeled_proxy_loader(
        eval_trainset, server_proxy_indices, batch_size=128
    )
    cluster_sample_weights = [
        sum(client_weights[cid] for cid in cluster_client_ids[cluster_id])
        for cluster_id in range(NUM_CLUSTERS)
    ]
    test_loader = DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=NUM_WORKERS, drop_last=False,
    )
    initial_state = deepcopy(base_model.state_dict())
    del base_model
    stage_times["loader_and_initial_model_sec"] = time.time() - stage_start

    preprocessing_time_sec = time.time() - pipeline_start
    run_config = {
        "status": "training_started",
        "experiment_group": EXPERIMENT_GROUP,
        "sensitivity_axis": SENSITIVITY_AXIS,
        "run_id": RUN_ID,
        "sensitivity_design": {
            "changed_parameter": SENSITIVITY_AXIS if EXPERIMENT_GROUP == "km_sensitivity" else None,
            "fixed_parameter": (
                {"K": NUM_CLUSTERS, "M": TEACHER_M}["M" if SENSITIVITY_AXIS == "K" else "K"]
                if EXPERIMENT_GROUP == "km_sensitivity" else None
            ),
        },
        "dataset": DATASET_NAME,
        "data_root": os.path.abspath(DATA_ROOT),
        "num_clients": NUM_CLIENTS,
        "num_classes": NUM_CLASSES,
        "K": NUM_CLUSTERS,
        "M": TEACHER_M,
        "beta": DIRICHLET_BETA,
        "seed": SEED,
        "rounds": NUM_ROUNDS,
        "methods": args.methods,
        "teacher_count_rule": {
            "top_m_methods": "min(M, number of active candidate clusters)",
            "all_equal": "all active candidate clusters (K-1 when all clusters are active)",
            "fedavg_ifca_no_distillation": True,
            "feddf": "all available client models are ensemble teachers for one global student",
        },
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "model_architecture": "torchvision ResNet-18 feature backbone with a newly initialized CIFAR-10 linear classifier",
        "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 5e-4},
        "num_workers": NUM_WORKERS,
        "fedprox_mu": FEDPROX_MU,
        "distillation": {
            "interval": DISTILL_INTERVAL, "epochs": DISTILL_EPOCHS,
            "tau": DISTILL_TAU, "lambda_kd": KD_LAMBDA, "learning_rate": DISTILL_LR,
        },
        "dktd_distillation_protocol": {
            "placement": "client-local within each target cluster",
            "training_data": "each participating target-cluster client's own local training partition",
            "student_initialization": "current target-cluster model copied to each client",
            "teacher_delivery": "selected teacher cluster models and weights are sent to each participating target client",
            "aggregation": "sample-weighted FedAvg of locally distilled student models at the server",
            "aggregation_weight": "number of local training examples held by each participating client",
            "round_schedule": f"every {DISTILL_INTERVAL} communication rounds after intra-cluster FedAvg",
            "epochs_per_client": DISTILL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": DISTILL_LR,
            "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 5e-4},
            "objective": "local cross-entropy plus lambda_kd times temperature-scaled KL divergence",
            "lambda_kd": KD_LAMBDA,
            "temperature": DISTILL_TAU,
            "server_pooled_target_cluster_training_data": False,
        },
        "batch_normalization": {
            "architecture": "torchvision ResNet-18",
            "fedbn": False,
            "floating_parameters_and_running_stats": "sample-weighted FedAvg, including BatchNorm running_mean and running_var",
            "num_batches_tracked": "copied from the participating client with the largest aggregation weight",
            "evaluation_and_teacher_forward": "BatchNorm in eval mode using the aggregated running statistics",
        },
        "feddf": {
            "proxy_protocol_id": FEDDF_PROXY_PROTOCOL,
            "student_scope": "single global ResNet-18 student",
            "round_protocol": "sample-weighted FedAvg initialization followed by server-side ensemble distillation each round",
            "teacher_ensemble": "arithmetic mean of raw logits from all participating client models",
            "objective": "temperature-scaled KL divergence only; no server ground-truth labels or CE loss",
            "proxy_dataset": "CIFAR-10 public benchmark training split",
            "proxy_source": "union of the per-client validation indices from the CIFAR-10 training split, pooled on the server as an unlabeled proxy",
            "proxy_selection_rule": "nominal 10% per-client validation split, performed class-wise; sparse client/class cells may use the existing minimum-count adjustment",
            "proxy_validation_fraction_nominal": VAL_FRACTION,
            "proxy_index_order": "concatenated by client_id, preserving each client's saved validation-index order",
            "proxy_indices_sha256": server_proxy_indices_sha256,
            "proxy_refresh_policy": "fixed indices, order, and deterministic evaluation transform reused every communication round",
            "proxy_samples": len(unlabeled_proxy_loader.dataset) if unlabeled_proxy_loader else 0,
            "proxy_labels_exposed_to_training": False,
            "proxy_is_client_validation_data": True,
            "proxy_is_external_auxiliary_dataset": False,
            "proxy_disjoint_from_client_train_and_cluster_holdout": True,
            "proxy_overlaps_cluster_validation": True,
            "proxy_images_also_used_for_dktd_teacher_reliability": True,
            "proxy_batch_size": 128,
            "proxy_num_workers": NUM_WORKERS,
            "epochs_per_round": DISTILL_EPOCHS,
            "temperature": DISTILL_TAU,
            "learning_rate": DISTILL_LR,
        },
        "scoring": {
            "rho": DKTD_RHO, "alpha_comp": DKTD_ALPHA,
            "beta_rel": DKTD_BETA_D, "gamma_risk": DKTD_GAMMA,
        },
        "split": {
            "dirichlet_seed": SEED, "validation_fraction": VAL_FRACTION,
            "cluster_holdout_fraction": CLUSTER_HOLDOUT_FRACTION,
            "train_samples": int(sum(len(x) for x in client_train_indices)),
            "validation_samples": int(sum(len(x) for x in client_validation_indices)),
            "cluster_holdout_samples": int(sum(len(x) for x in client_holdout_indices)),
            "partition_sha256": partition_hash,
            "client_data_partition_sha256": partition_hash,
            "cluster_assignment_sha256": cluster_assignment_hash,
            "assignment_file": "client_splits.json",
        },
        "transforms": {
            "train": "RandomCrop(32,padding=4) + RandomHorizontalFlip + ToTensor + ImageNet normalization",
            "validation_and_holdout": "ToTensor + ImageNet normalization; no stochastic augmentation",
        },
        "evaluation_protocol": {
            "global_test": "official CIFAR-10 test set; secondary metric",
            "cluster_holdout": "client-held-out samples; grouped by fixed K-Medoids clusters for DKTD/FedAvg/FedProx and by current assignments each round for IFCA",
            "teacher_reliability": "mean maximum softmax confidence on cluster validation split; no augmentation",
            "global_acc_semantics": "global_model_acc is the trained single model for FedAvg/FedProx/FedDF; cluster-model weighted averages are logged separately",
            "feddf_semantics": "one global student evaluated on the same official test set and each fixed cluster-held-out split; not a personalized K-model method",
            "dktd_original_cluster_pair": "same original-cluster untouched holdout before KD (after intra-cluster FedAvg) and after client-local KD/student aggregation; separate immediate paired effects, last-ten-round metrics, and final model",
            "dktd_original_cluster_reports": ["full_dktd/dktd_original_cluster_events.csv", "full_dktd/dktd_original_cluster_final.csv", "full_dktd/dktd_original_cluster_summary.json"],
        },
        "cross_seed_summary_protocol": {
            "per_seed_value": "mean over the last min(10, rounds) communication rounds",
            "uncertainty": "seed-level bootstrap percentile 95% CI",
            "paired_test": "two-sided sign-flip permutation test and bootstrap 95% CI on paired per-seed differences for every finite logged metric",
            "multiplicity": "Holm correction across method comparisons separately for each metric and configuration",
            "stability": "within-seed sample SD (ddof=1) over the final 20 communication rounds; unavailable when fewer than 20 complete rounds are present; then summarized across seeds",
        },
        "negative_transfer_threshold_percentage_points": NEG_TRANSFER_EPS_PERCENT,
        "weak_class_ratio": WEAK_CLASS_RATIO,
        "pretrained_requested": USE_PRETRAINED_ENCODER,
        "pretrained_load_status": PRETRAINED_LOAD_STATUS,
        "device": str(DEVICE),
        "cuda_name": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "git_revision": current_git_revision(),
        "git_worktree_dirty": current_worktree_dirty(),
        "script_sha256": current_script_sha256(),
        "stage_times_sec": stage_times,
        "preprocessing_time_sec": preprocessing_time_sec,
        "communication_note": "Local simulation; byte counts are not physical network measurements.",
        "training_rng_policy": "Reset client train-loader random streams to the same seed before each method for paired comparisons.",
    }
    with open(os.path.join(run_root, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, allow_nan=True)

    summaries = []
    run_config["completed_methods"] = []
    for strategy in args.methods:
        run_config["current_method"] = strategy
        with open(os.path.join(run_root, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2, allow_nan=True)
        set_seed(SEED)
        try:
            summary = run_one_strategy(
                strategy=strategy,
                initial_state=initial_state,
                test_loader=test_loader,
                client_loaders=client_loaders,
                client_assignment_loaders=client_assignment_loaders,
                client_weights=client_weights,
                client_holdout_indices=client_holdout_indices,
                eval_dataset=eval_trainset,
                cluster_client_ids=cluster_client_ids,
                cluster_validation_loaders=cluster_validation_loaders,
                cluster_holdout_loaders=cluster_holdout_loaders,
                cluster_label_centers=cluster_label_centers,
                cluster_prototypes=cluster_prototypes,
                cluster_sample_weights=cluster_sample_weights,
                unlabeled_proxy_loader=unlabeled_proxy_loader,
                run_root=run_root,
            )
        except BaseException as exc:
            run_config["status"] = (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
            run_config["failed_method"] = strategy
            run_config["failure_type"] = type(exc).__name__
            run_config["failure_message"] = str(exc)
            run_config["completed_methods"] = list(run_config["completed_methods"])
            with open(os.path.join(run_root, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(run_config, f, ensure_ascii=False, indent=2, allow_nan=True)
            raise
        summary["preprocessing_time_sec"] = preprocessing_time_sec
        summary["pipeline_elapsed_at_method_end_sec"] = time.time() - pipeline_start
        summaries.append(summary)
        run_config["completed_methods"].append(strategy)
        run_config["current_method"] = None
        with open(os.path.join(run_root, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2, allow_nan=True)

    pd.DataFrame(summaries).to_csv(
        os.path.join(run_root, "summary.csv"), index=False, encoding="utf-8-sig"
    )
    run_config["status"] = "completed"
    run_config["current_method"] = None
    run_config["total_pipeline_time_sec"] = time.time() - pipeline_start
    with open(os.path.join(run_root, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, allow_nan=True)
    print("\n全部方法运行完成：")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"结果目录: {run_root}")
    print(f"总流水线耗时: {time.time() - pipeline_start:.1f} 秒")
    return run_root


if __name__ == "__main__":
    main()
