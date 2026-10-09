from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    def __init__(self, n_classes: int = 1, lstm_hidden: int = 32):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden, num_layers=3, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)


class ECG_CNN_Transformer(nn.Module):
    def __init__(
        self,
        seq_len: int = 156,
        window_size: int = 13,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.seq_after_cnn = seq_len // 4
        self.window_size = window_size
        if self.seq_after_cnn % self.window_size != 0:
            raise ValueError(f"window_size={window_size} does not divide seq_after_cnn={self.seq_after_cnn}")
        self.n_tokens = self.seq_after_cnn // self.window_size
        self.embedding = nn.Linear(64 * self.window_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_tokens, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        batch_size, channels, length = x.shape
        x = x.view(batch_size, channels, self.n_tokens, self.window_size)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, self.n_tokens, channels * self.window_size)
        x = self.embedding(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


class LTCCell(nn.Module):
    """Liquid Time-Constant cell: τ(x,I) * dx/dt = -x + σ(Wx + UI + b)"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_f = nn.Linear(input_size, hidden_size)
        self.W_tau = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_tau = nn.Linear(input_size, hidden_size)

    def forward(self, inp: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_target = torch.sigmoid(self.W_f(h) + self.U_f(inp))
        tau = torch.exp(self.W_tau(h) + self.U_tau(inp)).clamp(min=0.1, max=10.0)
        return h + (-h + h_target) / tau


class CNN_LTC(nn.Module):
    """CNN feature extractor followed by a Liquid Time-Constant Network."""

    def __init__(self, n_classes: int = 1, hidden_size: int = 64):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(), nn.MaxPool1d(2),
        )
        self.ltc = LTCCell(input_size=32, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)
        B, T, _ = x.shape
        h = torch.zeros(B, self.ltc.hidden_size, device=x.device)
        for t in range(T):
            h = self.ltc(x[:, t, :], h)
        return self.fc(h)


class CTRNNCell(nn.Module):
    """Continuous-time RNN: τ * dx/dt = -x + tanh(Wx + UI + b)"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(input_size, hidden_size)
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inp: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        tau = self.log_tau.exp().clamp(min=0.1, max=10.0)
        return h + (-h + torch.tanh(self.W(h) + self.U(inp))) / tau


class CNN_CTRNN(nn.Module):
    """CNN feature extractor followed by a Continuous-time RNN."""

    def __init__(self, n_classes: int = 1, hidden_size: int = 64):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(), nn.MaxPool1d(2),
        )
        self.ctrnn = CTRNNCell(input_size=32, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)
        B, T, _ = x.shape
        h = torch.zeros(B, self.ctrnn.hidden_size, device=x.device)
        for t in range(T):
            h = self.ctrnn(x[:, t, :], h)
        return self.fc(h)


class _ODEFunc(nn.Module):
    """Learnable ODE right-hand side: dh/dt = f_θ(h)"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class NeuralODECell(nn.Module):
    """Neural ODE cell: Euler-integrates dh/dt = f(h) after input injection."""

    def __init__(self, input_size: int, hidden_size: int, n_steps: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.ode_func = _ODEFunc(hidden_size)

    def forward(self, inp: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h = h + self.input_proj(inp)
        dt = 1.0 / self.n_steps
        for _ in range(self.n_steps):
            h = h + dt * self.ode_func(h)
        return h


class CNN_NeuralODE(nn.Module):
    """CNN feature extractor followed by a Neural ODE (Euler integration)."""

    def __init__(self, n_classes: int = 1, hidden_size: int = 64):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(), nn.MaxPool1d(2),
        )
        self.ode_cell = NeuralODECell(input_size=32, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)
        B, T, _ = x.shape
        h = torch.zeros(B, self.ode_cell.hidden_size, device=x.device)
        for t in range(T):
            h = self.ode_cell(x[:, t, :], h)
        return self.fc(h)


class CTGRUCell(nn.Module):
    """Continuous-time GRU: reset gate + input-dependent time constant."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_h = nn.Linear(input_size, hidden_size)
        self.U_tau = nn.Linear(input_size, hidden_size)

    def forward(self, inp: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.W_r(h) + self.U_r(inp))
        h_cand = torch.tanh(self.W_h(r * h) + self.U_h(inp))
        tau = self.U_tau(inp).exp().clamp(min=0.1, max=10.0)
        return h + (-h + h_cand) / tau


class CNN_CTGRU(nn.Module):
    """CNN feature extractor followed by a Continuous-time GRU."""

    def __init__(self, n_classes: int = 1, hidden_size: int = 64):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(), nn.MaxPool1d(2),
        )
        self.ctgru = CTGRUCell(input_size=32, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)
        B, T, _ = x.shape
        h = torch.zeros(B, self.ctgru.hidden_size, device=x.device)
        for t in range(T):
            h = self.ctgru(x[:, t, :], h)
        return self.fc(h)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

