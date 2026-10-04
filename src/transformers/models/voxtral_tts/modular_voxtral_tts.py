# Copyright 2025 Mistral AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...modeling_outputs import CausalLMOutputWithPast
from ..mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralMLP,
    MistralModel,
    MistralPreTrainedModel,
    MistralRMSNorm,
    MistralRotaryEmbedding,
    repeat_kv,
)
from .configuration_voxtral_tts import VoxtralTtsCodecConfig, VoxtralTtsConfig, VoxtralTtsFlowMatchingConfig
from .generation_voxtral_tts import VoxtralTtsGenerationMixin


class VoxtralTtsMLP(MistralMLP):
    pass


class VoxtralTtsAttention(MistralAttention):
    pass


class VoxtralTtsRMSNorm(MistralRMSNorm):
    pass


class VoxtralTtsDecoderLayer(MistralDecoderLayer):
    pass


class VoxtralTtsRotaryEmbedding(MistralRotaryEmbedding):
    pass


class VoxtralTtsAudioEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.embed_audio_tokens = nn.Embedding(config.audio_vocab_size, config.hidden_size)

        self._semantic_codebook_size = config.semantic_codebook_size
        self._acoustic_codebook_size = config.acoustic_codebook_size
        self._n_acoustic_codebook = config.n_acoustic_codebook
        self._audio_vocab_size = config.audio_vocab_size
        self._num_special_audio_tokens = config.num_special_audio_tokens

    def _get_offsets(self, device: torch.device) -> torch.LongTensor:
        """Row at which each codebook's block starts inside the shared embedding table.

        Every codebook reserves `num_special_audio_tokens` rows below its own entries, so the
        blocks are `semantic_codebook_size + num_special` followed by `n_acoustic_codebook` blocks
        of `acoustic_codebook_size + num_special`. `audio_vocab_size` pads that total up to a
        multiple of 128; the padding trails the last block and belongs to no codebook.
        """
        sizes = torch.full(
            (self.num_codebooks,),
            self._acoustic_codebook_size + self._num_special_audio_tokens,
            dtype=torch.long,
            device=device,
        )
        sizes[0] = self._semantic_codebook_size + self._num_special_audio_tokens
        return torch.cumsum(sizes, dim=0) - sizes

    def forward(self, input_ids):
        offsets = self._get_offsets(input_ids.device)
        inputs_embeds = self.embed_audio_tokens(input_ids + offsets)
        inputs_embeds = inputs_embeds.sum(dim=2)
        return inputs_embeds


class VoxtralTtsPreTrainedModel(MistralPreTrainedModel):
    _no_split_modules = [
        "VoxtralTtsDecoderLayer",
        "VoxtralTtsFlowMatchingDecoderLayer",
        "VoxtralTtsCodecTransformerLayer",
    ]

    def _init_weights(self, module):
        """Defer entirely to Mistral's initialiser.

        This hook runs *after* `from_pretrained` materialises the checkpoint, so anything it writes
        overwrites trained values that loaded correctly. It must therefore never touch state the
        checkpoint supplies. The two tempting candidates are both handled in `__init__` instead, and
        both are recorded below because initialising them here silently destroyed the model twice.
        """
        super()._init_weights(module)

        # `VoxtralTtsCodecTransformerLayer`'s layer-scale vectors are deliberately NOT initialised
        # here either. They are trained parameters present in the checkpoint, and this hook runs
        # after materialisation -- filling them with `layer_scale_init` replaced every trained
        # value with the constant 0.01, which drives each transformer block's residual branch to
        # nearly zero and collapses the codec decoder onto its convolution stack. Their `__init__`
        # already seeds them with `layer_scale_init` for randomly initialised models.

        # `VoxtralTtsSemanticCodebook`'s `cluster_usage` / `embedding_sum` are deliberately NOT
        # touched here. They are persistent buffers carrying the trained VQ centroids, and this hook
        # runs after the checkpoint is materialised -- initialising them replaces the codebook with
        # noise, silently, in every `from_pretrained` model. Their `__init__` defaults (ones/zeros)
        # already give a finite state for randomly initialised models.


class VoxtralTtsBackboneModel(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = VoxtralTtsAudioEmbeddings(config)


class VoxtralTtsFlowMatchingAttention(nn.Module):
    """Bidirectional attention over the flow-matching transformer's three input tokens.

    The sequence is `[noisy acoustic sample, timestep, backbone hidden state]`. These are three
    heterogeneous conditioning signals rather than a temporal sequence, so no positional embedding
    is applied (the model uses sinusoidal encoding for the timestep *value* instead) and attention
    is unmasked.
    """

    def __init__(self, config: VoxtralTtsFlowMatchingConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=None, scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class VoxtralTtsFlowMatchingDecoderLayer(nn.Module):
    def __init__(self, config: VoxtralTtsFlowMatchingConfig, layer_idx: int):
        super().__init__()
        self.self_attn = VoxtralTtsFlowMatchingAttention(config=config, layer_idx=layer_idx)
        self.mlp = VoxtralTtsMLP(config)
        self.input_layernorm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class VoxtralTtsFlowMatchingTransformer(nn.Module):
    """Predicts the acoustic velocity field for one audio frame.

    Each call forms a three-token sequence from separately projected inputs -- the noisy acoustic
    sample, the timestep embedding, and the backbone hidden state -- runs bidirectional attention
    over it, and reads the velocity off the sample's position. The separate projections exist
    because the three inputs differ in activation scale.

    `semantic_codebook_output` lives here to mirror the checkpoint layout, but it is *not* part of
    this forward pass: the semantic codebook is predicted directly from the backbone hidden state
    via [`~VoxtralTtsFlowMatchingTransformer.predict_semantic_logits`].
    """

    def __init__(self, config: VoxtralTtsFlowMatchingConfig):
        super().__init__()
        self.config = config

        self.llm_projection = nn.Linear(config.input_dim, config.hidden_size, bias=False)
        self.time_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.input_projection = nn.Linear(config.acoustic_dim, config.hidden_size, bias=False)

        self.layers = nn.ModuleList(
            [VoxtralTtsFlowMatchingDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.semantic_codebook_output = nn.Linear(config.hidden_size, config.semantic_vocab_size, bias=False)
        self.acoustic_codebook_output = nn.Linear(config.hidden_size, config.acoustic_dim, bias=False)

    def _inv_freq(self, device: torch.device) -> torch.Tensor:
        """Inverse frequencies of the timestep embedding.

        Computed on demand rather than held in a buffer. A non-persistent buffer is absent from the
        checkpoint, so `from_pretrained` -- which builds the model on the meta device and then
        materialises it from the checkpoint -- would leave it pointing at uninitialised memory. That
        failure is near-silent: the embedding degenerates to `cos(0) = 1` for every timestep, so the
        flow-matching ODE loses its time conditioning while still producing plausible-looking output.
        """
        half_dim = self.config.hidden_size // 2
        exponent = torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim
        return torch.exp(-math.log(self.config.time_embedding_theta) * exponent)

    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding of `timesteps`, shape `(batch_size,)` -> `(batch_size, hidden_size)`."""
        freqs = timesteps.float().unsqueeze(-1) * self._inv_freq(timesteps.device).unsqueeze(0)
        return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    def predict_semantic_logits(self, backbone_hidden: torch.Tensor) -> torch.Tensor:
        """Semantic codebook logits, read directly off the backbone hidden state."""
        return self.semantic_codebook_output(backbone_hidden)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, input_dim)`):
                Backbone hidden state for the frame being generated.
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                Current ODE time in `[0, 1]`.
            acoustic_embeddings (`torch.Tensor` of shape `(batch_size, acoustic_dim)`):
                Current noisy acoustic sample `x_t`.

        Returns:
            `torch.Tensor` of shape `(batch_size, acoustic_dim)`: the velocity `v_t`.
        """
        timestep_embedding = self.get_timestep_embedding(timesteps).to(hidden_states.dtype)

        sequence = torch.stack(
            [
                self.input_projection(acoustic_embeddings),
                self.time_projection(timestep_embedding),
                self.llm_projection(hidden_states),
            ],
            dim=1,
        )

        for layer in self.layers:
            sequence = layer(sequence)

        sequence = self.norm(sequence)
        return self.acoustic_codebook_output(sequence[:, 0, :])


# ==================== Codec Components ====================


class VoxtralTtsSemanticCodebook(nn.Module):
    """Euclidean VQ codebook, stored as running sums so the effective table is a ratio."""

    epsilon: float = 1e-5

    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__()
        self.register_buffer("cluster_usage", torch.ones(config.semantic_codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(config.semantic_codebook_size, config.semantic_dim))

    @property
    def embeddings(self):
        return self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon).unsqueeze(-1)

    def decode(self, indices: torch.LongTensor) -> torch.Tensor:
        return nn.functional.embedding(indices, self.embeddings)


class VoxtralTtsQuantizer(nn.Module):
    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__()
        self.semantic_codebook = VoxtralTtsSemanticCodebook(config)
        self.acoustic_codebook_size = config.acoustic_codebook_size

    def decode_semantic(self, indices: torch.LongTensor) -> torch.Tensor:
        return self.semantic_codebook.decode(indices)

    def decode_acoustic(self, codes: torch.LongTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Map FSQ levels in `[0, acoustic_codebook_size - 1]` back onto `[-1, 1]`.

        Inverse of the quantisation applied to the flow-matching output, so the decoder sees values
        on the scale it was trained against rather than the raw continuous sample.
        """
        return codes.to(dtype) * (2.0 / (self.acoustic_codebook_size - 1)) - 1.0


def build_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Per-head ALiBi slopes: a geometric sequence with ratio `2 ** (-8 / num_heads)`.

    Head counts that are not powers of two fall back to interleaving the slopes of the next power
    of two, following the original ALiBi formulation.
    """

    def geometric(n: int) -> torch.Tensor:
        ratio = 2.0 ** (-8.0 / n)
        return torch.tensor([ratio**i for i in range(n)], dtype=torch.float32)

    if math.log2(num_heads).is_integer():
        return geometric(num_heads)
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    return torch.cat(
        [geometric(closest_power_of_2), geometric(2 * closest_power_of_2)[::2][: num_heads - closest_power_of_2]]
    )


class VoxtralTtsCodecAttention(nn.Module):
    """Sliding-window causal attention with ALiBi position bias.

    The codec carries no rotary or learned positional embedding; ALiBi is its only source of
    position information. `sliding_window` is per-block: it tracks the resampling factor, so the
    decoder widens it (2 -> 4 -> 8 -> 16) as it upsamples back towards the waveform.
    """

    def __init__(self, config: VoxtralTtsCodecConfig, sliding_window: int):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.is_causal = config.causal
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.use_biases)

        self.use_qk_norm = config.qk_norm
        if config.qk_norm:
            self.q_norm = VoxtralTtsRMSNorm(self.num_attention_heads * self.head_dim, eps=config.qk_norm_eps)
            self.k_norm = VoxtralTtsRMSNorm(self.num_key_value_heads * self.head_dim, eps=config.qk_norm_eps)

    def _build_attention_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """ALiBi bias combined with the causal and sliding-window masks, shape `(1, heads, q, kv)`.

        The slopes are recomputed here rather than cached in a buffer. A non-persistent buffer is
        absent from the checkpoint, so `from_pretrained` would leave it uninitialised and the codec
        would decode against a garbage position bias -- the same near-silent failure the
        flow-matching timestep embedding hit. There are only `num_attention_heads` of them, so
        there is nothing to gain from caching.
        """
        positions = torch.arange(seq_len, device=device)
        relative_position = positions.unsqueeze(0) - positions.unsqueeze(1)

        alibi_slopes = build_alibi_slopes(self.num_attention_heads).to(device=device, dtype=dtype)
        attn_bias = alibi_slopes.view(-1, 1, 1) * relative_position.to(dtype)

        if self.is_causal:
            attn_bias = attn_bias.masked_fill(relative_position.unsqueeze(0) > 0, torch.finfo(dtype).min)

        window_right = 0 if self.is_causal else self.sliding_window
        outside_window = (relative_position < -self.sliding_window) | (relative_position > window_right)
        attn_bias = attn_bias.masked_fill(outside_window.unsqueeze(0), torch.finfo(dtype).min)
        return attn_bias.unsqueeze(0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_bias = self._build_attention_bias(seq_len, hidden_states.device, query_states.dtype)
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attn_bias
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class VoxtralTtsCodecMLP(nn.Module):
    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_biases)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class VoxtralTtsCodecTransformerLayer(nn.Module):
    def __init__(self, config: VoxtralTtsCodecConfig, sliding_window: int, layer_idx: int = 0):
        super().__init__()
        self.self_attn = VoxtralTtsCodecAttention(config, sliding_window=sliding_window)
        self.mlp = VoxtralTtsCodecMLP(config)
        self.input_layernorm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = VoxtralTtsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_layer_scale = config.layer_scale
        if config.layer_scale:
            self.self_attn_layer_scale = nn.Parameter(torch.full((config.hidden_size,), config.layer_scale_init))
            self.mlp_layer_scale = nn.Parameter(torch.full((config.hidden_size,), config.layer_scale_init))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        if self.use_layer_scale:
            hidden_states = self.self_attn_layer_scale * hidden_states
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.use_layer_scale:
            hidden_states = self.mlp_layer_scale * hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


class VoxtralTtsCodecTransformerBlock(nn.Module):
    def __init__(self, config: VoxtralTtsCodecConfig, num_layers: int, sliding_window: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                VoxtralTtsCodecTransformerLayer(config, sliding_window=sliding_window, layer_idx=i)
                for i in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states.transpose(1, 2)


class VoxtralTtsCodecConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = True,
        use_weight_norm: bool = True,
        pad_mode: str = "replicate",
    ):
        super().__init__()
        self.causal = causal
        self.is_transpose = stride > 1
        self.pad_mode = pad_mode

        if self.is_transpose:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, bias=False)
            self.trim_right = kernel_size - stride
            self.left_pad = 0
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
            self.trim_right = 0
            self.left_pad = kernel_size - 1 if causal else (kernel_size - 1) // 2

        if use_weight_norm:
            nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.is_transpose:
            hidden_states = self.conv(hidden_states)
            if self.trim_right > 0 and self.causal:
                hidden_states = hidden_states[..., : -self.trim_right]
        else:
            if self.left_pad > 0:
                hidden_states = nn.functional.pad(hidden_states, (self.left_pad, 0), mode=self.pad_mode)
            hidden_states = self.conv(hidden_states)
        return hidden_states


class VoxtralTtsCodecModel(nn.Module):
    def __init__(self, config: VoxtralTtsCodecConfig):
        super().__init__()
        self.config = config
        self.quantizer = VoxtralTtsQuantizer(config)

        decoder_blocks = []
        in_channels = config.semantic_dim + config.acoustic_dim

        # The attention window tracks the resampling factor. The encoder narrows it on the way down
        # (16 -> 8 -> 4 -> 2), so the decoder starts at the narrow end and widens symmetrically.
        total_upsampling = math.prod(config.decoder_conv_strides)
        sliding_window = max(1, config.attn_sliding_window_size // total_upsampling)

        for kernel, stride, n_layers in zip(
            config.decoder_conv_kernels,
            config.decoder_conv_strides,
            config.decoder_transformer_lengths,
        ):
            decoder_blocks.append(
                VoxtralTtsCodecConvBlock(
                    in_channels,
                    config.hidden_size,
                    kernel,
                    stride,
                    causal=config.causal,
                    use_weight_norm=config.conv_weight_norm,
                )
            )
            if stride > 1 and config.half_attn_window_upon_downsampling:
                sliding_window *= stride
            decoder_blocks.append(VoxtralTtsCodecTransformerBlock(config, n_layers, sliding_window=sliding_window))
            in_channels = config.hidden_size

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.output_proj = VoxtralTtsCodecConvBlock(
            config.hidden_size,
            config.patch_size * config.channels,
            config.patch_proj_kernel_size,
            stride=1,
            causal=config.causal,
            use_weight_norm=config.conv_weight_norm,
            pad_mode="reflect",
        )
        self.patch_size = config.patch_size
        self.channels = config.channels

    def decode(
        self,
        semantic_token_ids: torch.LongTensor,
        acoustic_codes: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Decode one frame sequence into a waveform.

        Args:
            semantic_token_ids (`torch.LongTensor` of shape `(batch_size, num_frames)`):
                Semantic VQ indices, with the audio special-token offset already removed.
            acoustic_codes (`torch.LongTensor` of shape `(batch_size, num_frames, acoustic_dim)`):
                Discrete FSQ levels in `[0, acoustic_codebook_size - 1]`, offset already removed.

        Returns:
            `torch.Tensor` of shape `(batch_size, channels, num_samples)`: the waveform.
        """
        semantic_embeddings = self.quantizer.decode_semantic(semantic_token_ids)
        acoustic_values = self.quantizer.decode_acoustic(acoustic_codes, dtype=semantic_embeddings.dtype)
        decoder_input = torch.cat([semantic_embeddings, acoustic_values], dim=-1)

        hidden_states = decoder_input.transpose(1, 2)

        for block in self.decoder_blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.output_proj(hidden_states)

        batch_size = hidden_states.shape[0]
        num_frames = hidden_states.shape[2]
        hidden_states = hidden_states.view(batch_size, self.channels, self.patch_size, num_frames)
        hidden_states = hidden_states.permute(0, 1, 3, 2).contiguous()
        waveform = hidden_states.view(batch_size, self.channels, num_frames * self.patch_size)

        return waveform


class VoxtralTtsForTextToSpeech(VoxtralTtsPreTrainedModel, VoxtralTtsGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "embed_text_tokens.weight"}

    def __init__(self, config: VoxtralTtsConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size

        self.embed_text_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.backbone_model = VoxtralTtsBackboneModel._from_config(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.flow_matching_transformer = VoxtralTtsFlowMatchingTransformer(config.flow_matching_config)
        self.codec_model = VoxtralTtsCodecModel(config.codec_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_text_tokens

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        voice_embeddings: torch.FloatTensor,
    ) -> torch.Tensor:
        """Mask selecting the reference-voice placeholder positions in `inputs_embeds`.

        Mirrors the idiom used across the library's multimodal models, and checks that the number of
        placeholders matches the number of voice embedding rows -- a mismatch means the prompt and
        the preset disagree about the reference length, which would otherwise scatter silently.
        """
        if input_ids is None:
            special_audio_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_audio_mask = special_audio_mask.all(-1)
        else:
            special_audio_mask = input_ids == self.config.audio_token_id

        n_audio_tokens = int(special_audio_mask.sum())
        n_voice_frames = voice_embeddings.shape[-2]
        if n_audio_tokens != n_voice_frames:
            raise ValueError(
                f"Voice embeddings do not match the prompt: {n_audio_tokens} audio placeholder "
                f"tokens but {n_voice_frames} embedding frames."
            )
        return special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    def set_input_embeddings(self, value):
        self.embed_text_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        audio_codes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            parts = []
            if audio_codes is not None:
                parts.append(self.backbone_model.embed_tokens(audio_codes))
            if input_ids is not None:
                parts.append(self.embed_text_tokens(input_ids))
            if parts:
                inputs_embeds = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        backbone_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        if output_attentions is not None:
            backbone_kwargs["output_attentions"] = output_attentions
        if output_hidden_states is not None:
            backbone_kwargs["output_hidden_states"] = output_hidden_states
        if return_dict is not None:
            backbone_kwargs["return_dict"] = return_dict

        backbone_outputs = self.backbone_model(**backbone_kwargs, **kwargs)

        hidden_states = backbone_outputs[0]
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,) + backbone_outputs[1:]

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=backbone_outputs.past_key_values,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
        )


__all__ = [
    "VoxtralTtsPreTrainedModel",
    "VoxtralTtsBackboneModel",
    "VoxtralTtsCodecModel",
    "VoxtralTtsForTextToSpeech",
]
