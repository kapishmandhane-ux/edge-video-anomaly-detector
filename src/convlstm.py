"""
ConvLSTM Implementation in Pure PyTorch.

Provides:
- ConvLSTMCell: Fused 2D Convolutional LSTM cell.
- ConvLSTM: Multi-layer stacked spatio-temporal recurrent network.
Optimized for ONNX and TensorRT graph tracing with dynamic batch support.
"""

from typing import List, Tuple, Union, Optional
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell with fused gate convolutions.

    Formula:
        [i_t, f_t, g_t, o_t]^T = [sigmoid, sigmoid, tanh, sigmoid]^T (W * [X_t, H_{t-1}] + b)
        C_t = f_t * C_{t-1} + i_t * g_t
        H_t = o_t * tanh(C_t)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.bias = bias

        # Fused 4-gate 2D convolution: input (i), forget (f), cell (g), output (o)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution weights using Xavier normal, and forget gate bias to 1."""
        nn.init.xavier_normal_(self.conv.weight)
        if self.bias and self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
            # Initialize forget gate bias to 1.0 to prevent vanishing gradients early in training
            # Forget gate is the second chunk: [hidden_dim : 2 * hidden_dim]
            with torch.no_grad():
                self.conv.bias[self.hidden_dim : 2 * self.hidden_dim].fill_(1.0)

    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single time step.

        Args:
            input_tensor: Tensor of shape (B, C_in, H, W)
            cur_state: Tuple of (h_cur, c_cur), each of shape (B, C_hidden, H, W).
                       If None, initialized to zeros.

        Returns:
            Tuple of (h_next, c_next), each of shape (B, C_hidden, H, W)
        """
        b, _, h, w = input_tensor.shape

        if cur_state is None:
            h_cur = torch.zeros(
                (b, self.hidden_dim, h, w),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            c_cur = torch.zeros(
                (b, self.hidden_dim, h, w),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
        else:
            h_cur, c_cur = cur_state

        # Concatenate along channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv(combined)

        # Chunk into [i, f, g, o]
        i_gate, f_gate, g_gate, o_gate = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        spatial_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Explicitly allocate initial zero state."""
        h, w = spatial_size
        return (
            torch.zeros((batch_size, self.hidden_dim, h, w), device=device, dtype=dtype),
            torch.zeros((batch_size, self.hidden_dim, h, w), device=device, dtype=dtype),
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer stacked ConvLSTM module.

    Supports:
    - Multiple stacked recurrent layers
    - Dynamic batch dimension
    - Sequence-first or batch-first inputs (batch-first by default)
    - Clean static graph generation for ONNX and TensorRT engines
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, List[int]],
        kernel_size: Union[int, Tuple[int, int], List[Union[int, Tuple[int, int]]]] = 3,
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        # Standardize hidden dimensions across layers
        if isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim] * num_layers
        else:
            if len(hidden_dim) != num_layers:
                raise ValueError(
                    f"Length of hidden_dim ({len(hidden_dim)}) must match num_layers ({num_layers})"
                )
            self.hidden_dim = hidden_dim

        # Standardize kernel sizes across layers
        if isinstance(kernel_size, (int, tuple)):
            self.kernel_size = [kernel_size] * num_layers
        else:
            if len(kernel_size) != num_layers:
                raise ValueError(
                    f"Length of kernel_size ({len(kernel_size)}) must match num_layers ({num_layers})"
                )
            self.kernel_size = kernel_size

        self.num_layers = num_layers
        cell_list: List[ConvLSTMCell] = []

        for i in range(num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through all stacked ConvLSTM layers.

        Args:
            input_tensor: Tensor of shape:
                - If batch_first: (B, T, C, H, W)
                - If not batch_first: (T, B, C, H, W)
            hidden_state: Optional list of (h, c) tuples for each layer.

        Returns:
            layer_output:
                - If return_all_layers: List of Tensors, each (B, T, C_l, H, W)
                - Else: Tensor of shape (B, T, C_last, H, W)
            last_state_list: List of (h, c) for each layer at final time step.
        """
        if not self.batch_first:
            # Convert (T, B, C, H, W) -> (B, T, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.shape

        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w), input_tensor.device, input_tensor.dtype)

        layer_output_list: List[torch.Tensor] = []
        last_state_list: List[Tuple[torch.Tensor, torch.Tensor]] = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            cell = self.cell_list[layer_idx]
            h_state, c_state = hidden_state[layer_idx]
            output_inner: List[torch.Tensor] = []

            # Unroll temporally (statically exportable to ONNX)
            time_slices = torch.unbind(cur_layer_input, dim=1)
            for t_input in time_slices:
                h_state, c_state = cell(t_input, (h_state, c_state))
                output_inner.append(h_state)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h_state, c_state))

        if not self.return_all_layers:
            output = layer_output_list[-1]
        else:
            output = layer_output_list

        if not self.batch_first:
            if self.return_all_layers:
                output = [out.permute(1, 0, 2, 3, 4) for out in output]
            else:
                output = output.permute(1, 0, 2, 3, 4)

        return output, last_state_list

    def _init_hidden(
        self,
        batch_size: int,
        spatial_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, spatial_size, device=device, dtype=dtype)
            )
        return init_states

