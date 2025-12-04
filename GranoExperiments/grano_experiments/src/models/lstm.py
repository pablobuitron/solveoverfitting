import torch
import torch.nn as nn


class YieldLSTM(nn.Module):
    """
    LSTM model for yield prediction with optional static input features.

    The model expects:
      - x_dynamic: [batch, seq_len, n_dynamic_features]
      - x_static: [batch, static_size] or None
    """

    def __init__(self, input_size, hidden_size, num_layers, static_size=0):
        super().__init__()
        self.static_size = static_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size + static_size, 1)

    def forward(self, x_dynamic, x_static=None):
        """
        Forward pass.

        Args:
            x_dynamic (Tensor): [batch, seq_len, n_features]
            x_static (Tensor or None): [batch, static_size] or [batch, 1, static_size]

        Returns:
            y_pred (Tensor): [batch, 1]
        """

        # ---------------------------------------------------------
        # Validate dynamic input
        # ---------------------------------------------------------
        if x_dynamic.dim() != 3:
            raise ValueError(
                f"x_dynamic must have shape [batch, seq_len, features], but got {x_dynamic.shape}"
            )

        batch_size = x_dynamic.size(0)

        # ---------------------------------------------------------
        # Validate and format static input
        # ---------------------------------------------------------
        if self.static_size == 0:
            if x_static is not None:
                raise ValueError(
                    "x_static was provided but static_size is 0. "
                    "Either remove x_static or initialize YieldLSTM with static_size > 0."
                )
        else:
            if x_static is None:
                raise ValueError(
                    "x_static is required because static_size > 0, but got None."
                )

            # Accept shapes [batch, static_size] and [batch, 1, static_size]
            if x_static.dim() == 3:
                # Expect [batch, 1, static_size]
                if x_static.size(1) != 1:
                    raise ValueError(
                        f"x_static with dim==3 must be [batch, 1, static_size], but got {x_static.shape}"
                    )
                x_static = x_static[:, 0, :]  # -> [batch, static_size]

            elif x_static.dim() != 2:
                raise ValueError(
                    f"x_static must have shape [batch, static_size] or [batch, 1, static_size], but got {x_static.shape}"
                )

            if x_static.size(1) != self.static_size:
                raise ValueError(
                    f"x_static has wrong feature dimension. Expected {self.static_size}, got {x_static.size(1)}"
                )

            if x_static.size(0) != batch_size:
                raise ValueError(
                    f"x_static batch size mismatch: x_dynamic batch={batch_size}, x_static batch={x_static.size(0)}"
                )

        # ---------------------------------------------------------
        # LSTM forward
        # ---------------------------------------------------------
        out, (h_n, c_n) = self.lstm(x_dynamic)

        # Get last timestep hidden output
        out_last = out[:, -1, :]  # [batch, hidden_size]

        # ---------------------------------------------------------
        # Append static features (if provided)
        # ---------------------------------------------------------
        if self.static_size > 0:
            out_last = torch.cat([out_last, x_static], dim=1)

        # Final regression head
        y_pred = self.fc(out_last)
        return y_pred
