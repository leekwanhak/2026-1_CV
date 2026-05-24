"""utils.py — MLP 실험 공통 유틸리티 (HW#3)

포함 항목 (claude.md 섹션 12.1):
  1. set_seed          — 재현성을 위한 전역 seed 고정
  2. get_*             — 데이터로더 (Fashion-MNIST, Digits, make_moons, make_circles)
  3. train / train_one_epoch — 학습 루프 직접 구현 (for epoch in range(…) 형태)
  4. evaluate          — 평가 함수 (model.eval() + torch.no_grad() 필수)
  5. plot_*            — 시각화 유틸 (Loss/Acc 곡선, 활성화 분포, Dead ReLU 히트맵,
                         Gradient Flow 히트맵)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms

from sklearn.datasets import load_digits, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. 재현성 — Random Seed 고정
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Python / NumPy / PyTorch 전체 seed를 고정하여 실험 재현성을 보장한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 결정론적 CUDA 연산 강제 (속도 감소를 감수하고 재현성 최우선)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 2. 데이터로더
# ─────────────────────────────────────────────────────────────────────────────

def get_fashion_mnist(
    batch_size: int = 64,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader, int]:
    """Fashion-MNIST 데이터셋을 로드한다.

    반환: (train_loader, test_loader, num_classes=10)
    노트: 이미지 shape = (1, 28, 28) → 모델 첫 레이어에 nn.Flatten() 필요.
    """
    # 픽셀값을 [-1, 1] 범위로 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = torchvision.datasets.FashionMNIST(
        data_root, train=True,  download=True, transform=transform
    )
    test_ds = torchvision.datasets.FashionMNIST(
        data_root, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, 10


def get_digits(
    batch_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int]:
    """scikit-learn Digits 데이터셋 (8×8 픽셀, 64-dim)을 로드한다.

    반환: (train_loader, test_loader, num_classes=10)
    """
    digits = load_digits()
    X = digits.data.astype(np.float32)   # shape: (N, 64)
    y = digits.target.astype(np.int64)

    # StandardScaler로 특성 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, 10


def get_make_moons(
    n_samples: int = 2000,
    noise: float = 0.2,
    batch_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int]:
    """make_moons 2D 이진 분류 데이터를 생성·로드한다.

    실험 B에서 Dead ReLU 및 결정 경계 시각화에 사용.
    반환: (train_loader, test_loader, num_classes=2)
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, 2


def get_make_circles(
    n_samples: int = 2000,
    noise: float = 0.1,
    factor: float = 0.5,
    batch_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int]:
    """make_circles 2D 이진 분류 데이터를 생성·로드한다.

    실험 B에서 비선형 결정 경계 학습 능력 비교에 사용.
    반환: (train_loader, test_loader, num_classes=2)
    """
    X, y = make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=seed
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. 학습 루프 — for epoch in range(…) 형태로 직접 구현 (필수 요건)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_softmax_output: bool = False,
) -> tuple[float, float]:
    """단일 에폭 학습을 수행한다.

    Args:
        use_softmax_output: True이면 출력층에 softmax를 적용 후 MSE 계산.
                            실험 A의 'MSE + softmax' 조건에서 사용.
                            CrossEntropy 실험에서는 False (내부적으로 softmax 포함).
    반환: (avg_loss, accuracy)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        if use_softmax_output:
            # MSE 실험: softmax로 확률 분포 변환 후 one-hot 타깃과 비교
            # CrossEntropy와 달리 MSE는 자동으로 softmax를 적용하지 않으므로 명시 필요
            probs = torch.softmax(outputs, dim=1)
            targets_onehot = torch.zeros_like(probs)
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
            loss = loss_fn(probs, targets_onehot)
            preds = probs.argmax(dim=1)
        else:
            # CrossEntropy 실험: logit을 그대로 전달 (내부에서 log-softmax 수행)
            loss = loss_fn(outputs, targets)
            preds = outputs.argmax(dim=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct    += (preds == targets).sum().item()
        total      += inputs.size(0)

    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    use_softmax_output: bool = False,
    scheduler=None,
    verbose_every: int = 10,
    grad_history: dict | None = None,
) -> tuple[list, list, list, list]:
    """전체 학습을 에폭 루프로 수행한다.

    Args:
        scheduler:     LR 스케줄러 (ExponentialLR 등). None이면 미적용.
        grad_history:  dict를 전달하면 에폭마다 gradient norm을 누적 기록.
                       Gradient Flow 히트맵 생성에 활용.
    반환: (train_losses, train_accs, test_losses, test_accs)
    """
    train_losses, train_accs = [], []
    test_losses,  test_accs  = [], []

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, use_softmax_output
        )
        te_loss, te_acc = evaluate(
            model, test_loader, loss_fn, device, use_softmax_output
        )

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        # gradient norm 히스토리 수집 (Gradient Flow 히트맵에 활용)
        if grad_history is not None:
            collect_gradient_norms(grad_history, model)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % verbose_every == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:>3}/{epochs}] "
                f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
                f"Test  Loss: {te_loss:.4f}  Acc: {te_acc:.4f}"
            )

    return train_losses, train_accs, test_losses, test_accs


# ─────────────────────────────────────────────────────────────────────────────
# 4. 평가 함수 — model.eval() + torch.no_grad() 필수 사용
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    use_softmax_output: bool = False,
) -> tuple[float, float]:
    """평가 세트에 대한 loss와 accuracy를 계산한다.

    model.eval()로 Dropout·BatchNorm을 평가 모드로 전환하고,
    torch.no_grad()로 gradient 연산을 비활성화하여 메모리·속도 최적화.
    반환: (avg_loss, accuracy)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if use_softmax_output:
                probs = torch.softmax(outputs, dim=1)
                targets_onehot = torch.zeros_like(probs)
                targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
                loss = loss_fn(probs, targets_onehot)
                preds = probs.argmax(dim=1)
            else:
                loss = loss_fn(outputs, targets)
                preds = outputs.argmax(dim=1)

            total_loss += loss.item() * inputs.size(0)
            correct    += (preds == targets).sum().item()
            total      += inputs.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 5. 시각화 유틸
# ─────────────────────────────────────────────────────────────────────────────

# ── 5-A. Loss / Accuracy 곡선 ──────────────────────────────────────────────

def plot_loss_acc(
    train_losses: list,
    train_accs: list,
    test_losses: list,
    test_accs: list,
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Loss vs Epoch, Accuracy vs Epoch 곡선을 2패널로 출력한다 (필수 그래프)."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, test_losses,  label="Test Loss", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss Curve  {title}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, test_accs,  label="Test Acc", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy Curve  {title}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ── 5-B. 중간 레이어 활성화 분포 ──────────────────────────────────────────

def collect_layer_activations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 5,
) -> dict[str, torch.Tensor]:
    """Forward hook으로 각 활성화 함수 레이어의 출력값을 수집한다.

    수집 대상: nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh
    반환: {layer_name: Tensor(total_samples, n_neurons)}
    """
    activations: dict[str, list] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, input, output):
            activations.setdefault(name, []).append(output.detach().cpu())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= max_batches:
                break
            model(inputs.to(device))

    for h in hooks:
        h.remove()

    # 배치 차원을 합쳐 하나의 텐서로 반환
    return {k: torch.cat(v, dim=0) for k, v in activations.items()}


def plot_activation_distributions(
    activations: dict[str, torch.Tensor],
    plot_type: str = "histogram",
    title_prefix: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """레이어별 활성화 분포를 히스토그램 또는 BoxPlot으로 시각화한다.

    Args:
        plot_type: "histogram" 또는 "boxplot"
    분석 포인트:
      - ReLU: 출력이 0에 집중되어 있으면 Dead ReLU 의심
      - Sigmoid: 출력이 0 또는 1 근처에 포화되면 Vanishing Gradient 의심
    """
    n = len(activations)
    if n == 0:
        print("[Warning] No activations collected. Check if the model has ReLU/Sigmoid layers.")
        return None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, activations.items()):
        flat = values.numpy().flatten()
        if plot_type == "histogram":
            ax.hist(flat, bins=60, density=True, alpha=0.75, color="steelblue",
                    edgecolor="white", linewidth=0.3)
            ax.set_xlabel("Activation value")
            ax.set_ylabel("Density")
        elif plot_type == "boxplot":
            # 뉴런 단위 분포 — Neuron × Samples 행렬을 boxplot으로 표현
            data = values.numpy()
            step = max(1, data.shape[1] // 30)  # 뉴런 수가 많으면 샘플링
            ax.boxplot(data[:, ::step], vert=True, patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.6),
                       flierprops=dict(marker=".", markersize=2))
            ax.set_xlabel("Neuron index (sampled)")
            ax.set_ylabel("Activation value")
        ax.set_title(f"{title_prefix}  {name}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ── 5-C. Dead ReLU 뉴런 비율 히트맵 ──────────────────────────────────────

def compute_dead_relu_ratio(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> dict[str, float]:
    """각 ReLU/LeakyReLU 레이어에서 항상 비활성(≤0)인 뉴런의 비율을 계산한다.

    Dead ReLU: 데이터 전체에서 단 한 번도 양수 출력을 내지 못한 뉴런.
    반환: {layer_name: dead_ratio}  (0 ~ 1)
    """
    layer_pos_counts: dict[str, torch.Tensor] = {}  # 뉴런별 양수 출력 누적 횟수
    layer_sample_counts: dict[str, int] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, input, output):
            out = output.detach().cpu().reshape(output.size(0), -1)  # (B, N)
            pos = (out > 0).float().sum(dim=0)                       # (N,)
            if name not in layer_pos_counts:
                layer_pos_counts[name] = torch.zeros(out.size(1))
                layer_sample_counts[name] = 0
            layer_pos_counts[name]   += pos
            layer_sample_counts[name] += out.size(0)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= max_batches:
                break
            model(inputs.to(device))

    for h in hooks:
        h.remove()

    # 전체 샘플에서 한 번도 양수가 되지 않은 뉴런 → Dead
    return {
        name: (layer_pos_counts[name] == 0).float().mean().item()
        for name in layer_pos_counts
    }


def plot_dead_relu_heatmap(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
    save_path: str | None = None,
) -> plt.Figure:
    """Dead ReLU 뉴런의 위치를 레이어별 히트맵으로 시각화한다 (필수 히트맵).

    붉을수록 해당 뉴런이 자주(또는 항상) 비활성임을 의미.
    분석 포인트: 특정 레이어에 dead 뉴런이 집중되면 해당 층이 학습에 기여하지 못함.
    """
    layer_active: dict[str, list] = {}  # {name: [Tensor(B, N), ...]}
    hooks = []

    def make_hook(name: str):
        def hook(module, input, output):
            out = output.detach().cpu().reshape(output.size(0), -1)
            layer_active.setdefault(name, []).append((out > 0).float())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= max_batches:
                break
            model(inputs.to(device))

    for h in hooks:
        h.remove()

    layer_names = list(layer_active.keys())
    if not layer_names:
        print("[Warning] No ReLU/LeakyReLU layers found.")
        return None

    # 각 레이어: 샘플 전체에서 뉴런별 활성 빈도 (0 = always dead, 1 = always active)
    max_n = max(torch.cat(v).size(1) for v in layer_active.values())
    heatmap_rows = []
    for name in layer_names:
        acts = torch.cat(layer_active[name], dim=0)          # (total, N)
        active_rate = acts.mean(dim=0).numpy()               # (N,) ∈ [0,1]
        dead_rate   = 1.0 - active_rate                      # 높을수록 dead
        padded = np.full(max_n, np.nan)
        padded[:len(dead_rate)] = dead_rate
        heatmap_rows.append(padded)

    matrix = np.array(heatmap_rows)  # (num_layers, max_neurons)

    fig_w = min(max(8, max_n // 8), 24)
    fig_h = max(3, len(layer_names) + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=10)
    ax.set_xlabel("Neuron Index")
    ax.set_title("Dead ReLU Neuron Ratio per Layer  (Red = always dead)")
    plt.colorbar(im, ax=ax, label="Dead Ratio  (1 = never activated)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ── 5-D. Gradient Flow 시각화 ─────────────────────────────────────────────

def plot_gradient_flow(
    model: nn.Module,
    title: str = "Gradient Flow",
    save_path: str | None = None,
) -> plt.Figure:
    """loss.backward() 직후 각 레이어의 gradient norm을 막대그래프로 시각화.

    사용법:
        loss.backward()           # 먼저 backward 호출
        plot_gradient_flow(model) # 그 후 이 함수 호출

    분석 포인트:
      - 특정 레이어에서 gradient가 0에 가까우면 → Vanishing Gradient
      - 특정 레이어에서 gradient가 폭발적으로 크면 → Exploding Gradient
    """
    named_params = [
        (n, p) for n, p in model.named_parameters()
        if p.grad is not None and "weight" in n
    ]
    if not named_params:
        print("[Warning] No gradients found. Call loss.backward() first.")
        return None

    layer_labels = [n.replace(".weight", "") for n, _ in named_params]
    grad_norms   = [p.grad.abs().mean().item() for _, p in named_params]

    fig, ax = plt.subplots(figsize=(max(6, len(layer_labels) * 1.2), 4))
    bars = ax.bar(range(len(layer_labels)), grad_norms, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean |Gradient|")
    ax.set_title(title)
    ax.set_yscale("log")  # log scale: vanishing/exploding gradient를 한눈에 확인
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def collect_gradient_norms(
    grad_history: dict,
    model: nn.Module,
) -> dict:
    """에폭마다 호출하여 레이어별 gradient norm 히스토리를 누적한다.

    사용법:
        grad_history = {}
        for epoch in range(epochs):
            ...
            loss.backward()
            collect_gradient_norms(grad_history, model)  # backward 직후 호출
    반환: {layer_name: [norm_ep0, norm_ep1, ...]}
    """
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name:
            norm = param.grad.abs().mean().item()
            grad_history.setdefault(name, []).append(norm)
    return grad_history


def plot_gradient_flow_heatmap(
    grad_history: dict,
    title: str = "Gradient Flow over Epochs",
    save_path: str | None = None,
) -> plt.Figure:
    """에폭 × 레이어 gradient norm 히트맵을 출력한다 (필수 히트맵).

    어느 층에서 어느 시점에 gradient가 소멸/폭발하는지 시각적으로 확인.
    분석 포인트:
      - 초기 레이어에서 gradient가 지속적으로 낮으면 → Vanishing Gradient
      - 후반 에폭에서 gradient가 갑자기 커지면 → 학습 불안정 (Exploding)
    """
    if not grad_history:
        print("[Warning] Gradient history is empty.")
        return None

    layers = list(grad_history.keys())
    matrix = np.array([grad_history[l] for l in layers])  # (L, E)

    fig_w = min(max(8, matrix.shape[1] // 4 + 2), 24)
    fig_h = max(3, len(layers) + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([l.replace(".weight", "") for l in layers], fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Mean |Gradient|")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
