# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import warnings

import torch
from torch import nn

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


class ActorCriticRecurrentRay(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        obs_ranges: dict,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)

        self.obs_ranges = obs_ranges

        self.obs_proprio_range = obs_ranges["proprio"]
        self.obs_priv_range = obs_ranges["priv"]
        self.obs_actor_ray_range = obs_ranges["actor_ray"]
        self.obs_critic_ray_range = obs_ranges["critic_ray"]

        self.obs_proprio_dim = self.obs_proprio_range[1] - self.obs_proprio_range[0]
        self.obs_priv_dim = self.obs_priv_range[1] - self.obs_priv_range[0]
        self.obs_actor_ray_dim = self.obs_actor_ray_range[1] - self.obs_actor_ray_range[0]
        self.obs_critic_ray_dim = self.obs_critic_ray_range[1] - self.obs_critic_ray_range[0]

        self.ray_latent_dim_actor = 64
        self.ray_latent_dim_critic = 128
        self.num_actions = num_actions

        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.memory_a = Memory(
            self.obs_proprio_dim + self.ray_latent_dim_actor,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )
        self.memory_c = Memory(
            self.obs_proprio_dim + self.ray_latent_dim_critic + self.obs_priv_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        self.ray_encoder_actor = RayEncoderCNN1x180(
            activation=activation,
            ray_latent_dim=self.ray_latent_dim_actor,
        )
        self.ray_encoder_critic = RayEncoderCNN3x180(
            activation=activation,
            ray_latent_dim=self.ray_latent_dim_critic,
        )

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        print(f"Ray Encoder Actor: {self.ray_encoder_actor}")
        print(f"Ray Encoder Critic: {self.ray_encoder_critic}")

    def get_policy_params(self):
        params = []
        params.extend(self.actor.parameters())
        params.extend(self.memory_a.parameters())
        params.extend(self.ray_encoder_actor.parameters())
        return params

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def reset_policy_hidden_states(self, hidden_states):
        self.memory_a.hidden_states = hidden_states

    def detach_policy_hidden_states(self, dones=None):
        self.memory_a.detach_hidden_states(dones)

    def get_policy_hidden_states(self):
        return self.memory_a.hidden_states

    def act(self, observations, masks=None, hidden_states=None):
        obs = torch.cat(
            [
                observations[..., self.obs_proprio_range[0] : self.obs_proprio_range[1]],
                self.ray_encoder_actor(observations[..., self.obs_actor_ray_range[0] : self.obs_actor_ray_range[1]]),
            ],
            dim=-1,
        )
        input_a = self.memory_a(obs, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        obs = torch.cat(
            [
                observations[..., self.obs_proprio_range[0] : self.obs_proprio_range[1]],
                self.ray_encoder_actor(observations[..., self.obs_actor_ray_range[0] : self.obs_actor_ray_range[1]]),
            ],
            dim=-1,
        )
        input_a = self.memory_a(obs)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        obs = torch.cat(
            [
                critic_observations[..., self.obs_proprio_range[0] : self.obs_proprio_range[1]],
                critic_observations[..., self.obs_priv_range[0] : self.obs_priv_range[1]],
                self.ray_encoder_critic(
                    critic_observations[..., self.obs_critic_ray_range[0] : self.obs_critic_ray_range[1]]
                ),
            ],
            dim=-1,
        )
        input_c = self.memory_c(obs, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class RayEncoderCNN1x180(nn.Module):
    def __init__(self, activation: nn.Module, ray_latent_dim: int = 64):
        super().__init__()

        self.input_dim = 180  # 180 rays with 2-degree separation

        # Circular 1D CNN backbone for processing ray distances
        self.cnn = nn.Sequential(
            # Input: (N, 1, 180)
            nn.Conv1d(1, 4, kernel_size=5, stride=1, padding=2, padding_mode="circular"),
            activation,
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Output: (N, 4, 90)
            nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1, padding_mode="circular"),
            activation,
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Output: (N, 8, 45)
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1, padding_mode="circular"),
            activation,
            nn.MaxPool1d(kernel_size=3, stride=3),
            # Output: (N, 16, 15)
        )

        # MLP head for final processing
        self.mlp = nn.Sequential(
            nn.Linear(240, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, ray_latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the circular ray encoder.

        Args:
            x (torch.Tensor): A tensor of shape (..., 180) containing ray distances.
        """
        # Store original shape for reshaping output
        original_shape = x.shape[:-1]
        batch_size = x.shape[:-1].numel()

        # Reshape and add channel dimension for Conv1D
        x = x.reshape(batch_size, 1, self.input_dim)  # (N, 1, 180)

        # Process with CNN
        features = self.cnn(x)  # (N, 16, 15)

        # Flatten for MLP
        flattened = features.view(batch_size, -1)  # (N, 240)

        # Process with MLP
        output = self.mlp(flattened)

        # Reshape back to original batch dimensions
        return output.view(*original_shape, -1)


class RayEncoderCNN3x180(nn.Module):
    def __init__(self, activation: nn.Module, ray_latent_dim: int = 64):
        super().__init__()

        self.input_dim = 180  # 180 rays with 2-degree separation

        # Circular 1D CNN backbone for processing ray distances
        self.cnn = nn.Sequential(
            # Input: (N, 3, 180)
            nn.Conv1d(3, 8, kernel_size=5, stride=1, padding=2, padding_mode="circular"),
            activation,
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Output: (N, 8, 90)
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1, padding_mode="circular"),
            activation,
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Output: (N, 16, 45)
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, padding_mode="circular"),
            activation,
            nn.MaxPool1d(kernel_size=3, stride=3),
            # Output: (N, 32, 15)
        )

        # MLP head for final processing
        self.mlp = nn.Sequential(
            nn.Linear(480, 512),
            activation,
            nn.Linear(512, 512),
            activation,
            nn.Linear(512, ray_latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the circular ray encoder.

        Args:
            x (torch.Tensor): A tensor of shape (..., 180) containing ray distances.
        """
        # Store original shape for reshaping output
        original_shape = x.shape[:-1]
        batch_size = x.shape[:-1].numel()

        # Reshape and add channel dimension for Conv1D
        x = x.reshape(batch_size, 3, self.input_dim)  # (N, 3, 180)

        # Process with CNN
        features = self.cnn(x)  # (N, 32, 15)

        # Flatten for MLP
        flattened = features.view(batch_size, -1)  # (N, 480)

        # Process with MLP
        output = self.mlp(flattened)

        # Reshape back to original batch dimensions
        return output.view(*original_shape, -1)


def export_actor_onnx(
    model: ActorCriticRecurrentRay,
    onnx_path: str,
    normalizer=None,
    verbose: bool = False,
):
    """
    Export the actor model to ONNX format with separate inputs for deployment.

    Args:
        model: The ActorCriticRecurrentHigh model to export
        onnx_path: Path where to save the ONNX model
        normalizer: Optional observation normalizer
        verbose: Whether to print verbose ONNX export information
    """
    import torch.onnx

    class _ActorONNXExporter(torch.nn.Module):
        """ONNX exporter for ActorCriticRecurrentHigh actor."""

        def __init__(self, policy, normalizer=None):
            super().__init__()
            # Copy actor components
            self.actor = copy.deepcopy(policy.actor)
            self.ray_encoder_actor = copy.deepcopy(policy.ray_encoder_actor)
            self.rnn = copy.deepcopy(policy.memory_a.rnn)

            # Store dimensions
            self.obs_proprio_dim = policy.obs_proprio_dim
            self.obs_actor_ray_dim = policy.obs_actor_ray_dim

            # Copy normalizer if exists
            if normalizer:
                self.normalizer = copy.deepcopy(normalizer)
            else:
                self.normalizer = torch.nn.Identity()

        def forward(self, proprio_obs, ray_obs, h_in, c_in):
            # Encode ray observations
            ray_features = self.ray_encoder_actor(ray_obs)

            # Concatenate all observations
            obs = torch.cat([proprio_obs, ray_features], dim=-1)

            # Apply normalization
            obs = self.normalizer(obs)

            # Process through RNN
            x, (h_out, c_out) = self.rnn(obs.unsqueeze(0), (h_in, c_in))
            x = x.squeeze(0)

            # Get actions from actor
            actions = self.actor(x)

            return actions, h_out, c_out

    # Create exporter
    exporter = _ActorONNXExporter(model, normalizer)
    exporter.to("cpu")

    # Create dummy inputs with batch size 1
    proprio_obs = torch.zeros(1, model.obs_proprio_dim)
    ray_obs = torch.zeros(1, model.obs_actor_ray_dim)
    h_in = torch.zeros(model.rnn_num_layers, 1, model.rnn_hidden_dim)
    c_in = torch.zeros(model.rnn_num_layers, 1, model.rnn_hidden_dim)

    # Test forward pass
    actions, h_out, c_out = exporter(proprio_obs, ray_obs, h_in, c_in)

    # Export to ONNX
    torch.onnx.export(
        exporter,
        (proprio_obs, ray_obs, h_in, c_in),
        onnx_path,
        export_params=True,
        opset_version=11,
        verbose=verbose,
        input_names=[
            "proprio_obs",
            "ray_obs",
            "h_in",
            "c_in",
        ],
        output_names=[
            "actions",
            "h_out",
            "c_out",
        ],
        dynamic_axes={},
    )

    print(f"Actor model exported to {onnx_path}")
    print("Input shapes:")
    print(f"  - proprio_obs: (1, {model.obs_proprio_dim})")
    print(f"  - ray_obs: (1, {model.obs_actor_ray_dim})")
    print(f"  - h_in/c_in: ({model.rnn_num_layers}, 1, {model.rnn_hidden_dim})")
    print("Output shapes:")
    print(f"  - actions: (1, {model.num_actions})")
    print(f"  - h_out/c_out: ({model.rnn_num_layers}, 1, {model.rnn_hidden_dim})")


def export_actor_torchscript(
    model: ActorCriticRecurrentRay,
    torchscript_path: str,
    normalizer=None,
):
    """
    Export the actor model to TorchScript format with internal hidden state buffers.

    Args:
        model: The ActorCriticRecurrentRay model to export
        torchscript_path: Path where to save the TorchScript model
        normalizer: Optional observation normalizer
        verbose: Whether to print verbose export information
    """
    import torch.jit

    class _RayEncoderExporter(torch.nn.Module):
        """TorchScript exporter for ray encoder with single batch dimension."""

        def __init__(self, ray_encoder):
            super().__init__()
            self.cnn = copy.deepcopy(ray_encoder.cnn)
            self.mlp = copy.deepcopy(ray_encoder.mlp)
            self.input_dim = ray_encoder.input_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Process ray observations with single batch dimension."""
            # x shape: (batch_size, ray_dim)
            batch_size = x.shape[0]

            # Reshape for CNN: (batch_size, channels, ray_dim)
            x = x.reshape(batch_size, 1, self.input_dim)

            # Process with CNN
            features = self.cnn(x)

            # Flatten for MLP
            flattened = features.view(batch_size, -1)

            # Process with MLP
            output = self.mlp(flattened)

            return output

    class _ActorTorchScriptExporter(torch.nn.Module):
        """TorchScript exporter for ActorCriticRecurrentRay actor with internal state."""

        def __init__(self, policy, normalizer=None):
            super().__init__()
            # Copy actor components
            self.actor = copy.deepcopy(policy.actor)
            self.ray_encoder_actor = _RayEncoderExporter(policy.ray_encoder_actor)
            self.rnn = copy.deepcopy(policy.memory_a.rnn)

            # Store dimensions
            self.obs_proprio_dim = policy.obs_proprio_dim
            self.obs_actor_ray_dim = policy.obs_actor_ray_dim
            self.rnn_num_layers = policy.rnn_num_layers
            self.rnn_hidden_dim = policy.rnn_hidden_dim

            # Copy normalizer if exists
            if normalizer:
                self.normalizer = copy.deepcopy(normalizer)
            else:
                self.normalizer = torch.nn.Identity()

            # Register internal buffers for hidden states
            self.register_buffer("h_state", torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim))
            self.register_buffer("c_state", torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim))

        def forward(self, proprio_obs: torch.Tensor, ray_obs: torch.Tensor) -> torch.Tensor:
            # Encode ray observations
            ray_features = self.ray_encoder_actor(ray_obs)

            # Concatenate all observations
            obs = torch.cat([proprio_obs, ray_features], dim=-1)

            # Apply normalization
            obs = self.normalizer(obs)

            # Process through RNN with internal state
            x, (h_out, c_out) = self.rnn(obs.unsqueeze(0), (self.h_state, self.c_state))
            x = x.squeeze(0)

            # Update internal state
            self.h_state = h_out
            self.c_state = c_out

            # Get actions from actor
            actions = self.actor(x)

            return actions

        def reset_hidden_states(self):
            """Reset internal hidden states to zero."""
            self.h_state.zero_()
            self.c_state.zero_()

    # Create exporter
    exporter = _ActorTorchScriptExporter(model, normalizer)
    exporter.to("cpu")
    exporter.eval()

    # Create dummy inputs with batch size 1
    proprio_obs = torch.zeros(1, model.obs_proprio_dim)
    ray_obs = torch.zeros(1, model.obs_actor_ray_dim)

    # Try torch.jit.script first
    try:
        print("Attempting export with torch.jit.script...")
        scripted_model = torch.jit.script(exporter)
        scripted_model.save(torchscript_path)
        export_method = "torch.jit.script"
    except Exception as e:
        print(f"torch.jit.script failed with error: {e}")
        print("Falling back to torch.jit.trace...")

        # Fall back to torch.jit.trace
        try:
            traced_model = torch.jit.trace(exporter, (proprio_obs, ray_obs))
            traced_model.save(torchscript_path)
            export_method = "torch.jit.trace"
        except Exception as trace_error:
            raise RuntimeError(
                f"Both torch.jit.script and torch.jit.trace failed. Script error: {e}. Trace error: {trace_error}"
            )

    print(f"Actor model exported to {torchscript_path} using {export_method}")
    print("Input shapes:")
    print(f"  - proprio_obs: (1, {model.obs_proprio_dim})")
    print(f"  - ray_obs: (1, {model.obs_actor_ray_dim})")
    print("Output shape:")
    print(f"  - actions: (1, {model.num_actions})")
    print("Note: Hidden states are managed internally. Call reset_hidden_states() to reset.")
