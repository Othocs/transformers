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

from dataclasses import dataclass

import torch

from ...generation import GenerationMixin
from ...utils import ModelOutput, logging


logger = logging.get_logger(__name__)


@dataclass
class VoxtralTtsGenerateOutput(ModelOutput):
    """
    Output of [`VoxtralTtsForTextToSpeech.generate`].

    Args:
        audio (`list[torch.Tensor]`, *optional*):
            Generated waveforms, one per batch entry, each of shape `(num_samples,)`. Entries are
            trimmed independently at their own `[END_AUDIO]` frame, so lengths differ.
        audio_codes (`list[torch.LongTensor]`, *optional*):
            Generated frames per batch entry, each of shape `(num_frames, num_codebooks)`. Codes
            carry the audio special-token offset, matching what the backbone consumes.
        semantic_tokens (`list[torch.LongTensor]`, *optional*):
            Semantic codebook ids per batch entry, offset removed, so in
            `[0, semantic_codebook_size)`.
    """

    audio: list[torch.Tensor] | None = None
    audio_codes: list[torch.LongTensor] | None = None
    semantic_tokens: list[torch.LongTensor] | None = None


class VoxtralTtsGenerationMixin(GenerationMixin):
    """
    Generation for Voxtral TTS, which does not fit the standard `generate` loop: each step samples a
    37-integer audio frame rather than a token id, produced by two modules and embedded through a
    separate table. The pipeline per frame is:

    1. **Backbone step** -- a causal decoder over text + voice-reference frames produces a hidden
       state, with KV caching.
    2. **Semantic codebook** -- a linear head reads the semantic id directly off that hidden state,
       greedily. Predicting `[END_AUDIO]` terminates the sequence.
    3. **Acoustic codebooks** -- a flow-matching ODE, conditioned on the same hidden state and
       guided by classifier-free guidance against a zeroed condition, produces 36 continuous values
       that are scalar-quantised to FSQ levels.
    4. **Codec decode** -- semantic ids plus acoustic levels become a 24 kHz waveform.
    """

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        voice_embeddings: torch.FloatTensor | None = None,
        audio_codes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 2048,
        num_flow_steps: int | None = None,
        guidance_scale: float | None = None,
        output_audio: bool = True,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> VoxtralTtsGenerateOutput:
        r"""
        Generate speech from text and an optional voice reference.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Prompt token ids from [`VoxtralTtsProcessor`], containing a run of
                `config.audio_token_id` placeholders for the reference voice.
            voice_embeddings (`torch.FloatTensor` of shape `(num_ref_frames, hidden_size)`, *optional*):
                Reference-voice embeddings, scattered onto the placeholder positions in `input_ids`.
                The shipped voice presets are pre-computed embeddings rather than codes, so this --
                not `audio_codes` -- is the working path for this checkpoint.
            audio_codes (`torch.LongTensor` of shape `(batch_size, num_ref_frames, num_codebooks)`, *optional*):
                Reference frames as discrete codes, for references produced by a codec encoder. The
                released checkpoint ships no encoder, so this path is unused in practice; prefer
                `voice_embeddings`.
            attention_mask (`torch.Tensor`, *optional*):
                Only all-ones masks are currently supported; padded batches raise.
            max_new_tokens (`int`, *optional*, defaults to 2048):
                Frame budget. At 12.5 Hz, 2048 frames is roughly 164 seconds. Generation usually
                stops earlier, when the semantic codebook predicts `[END_AUDIO]`.
            num_flow_steps (`int`, *optional*):
                Euler steps for the flow-matching ODE. Defaults to `config.num_flow_steps`.
            guidance_scale (`float`, *optional*):
                Classifier-free guidance scale on the acoustic velocity. Defaults to
                `config.guidance_scale`. `1.0` disables guidance.
            output_audio (`bool`, *optional*, defaults to `True`):
                Whether to run the codec and return waveforms.
            generator (`torch.Generator`, *optional*):
                Generator for the flow-matching noise, the only stochastic part of the pipeline.

        Returns:
            [`VoxtralTtsGenerateOutput`]

        Example:
        ```python
        >>> from transformers import VoxtralTtsProcessor, VoxtralTtsForTextToSpeech

        >>> processor = VoxtralTtsProcessor.from_pretrained("mistralai/Voxtral-4B-TTS-2603")
        >>> model = VoxtralTtsForTextToSpeech.from_pretrained("mistralai/Voxtral-4B-TTS-2603")

        >>> inputs = processor("Hello, my name is Voxtral!", voice_preset="neutral_female")
        >>> output = model.generate(**inputs)
        >>> processor.save_audio(output.audio[0], "output.wav")
        ```
        """
        config = self.config
        if input_ids is None and audio_codes is None:
            raise ValueError("At least one of `input_ids` or `audio_codes` must be provided.")
        if voice_embeddings is not None and input_ids is None:
            raise ValueError("`voice_embeddings` needs `input_ids` to carry the placeholder positions.")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise NotImplementedError(
                "Voxtral TTS generation does not yet support padded batches. Generate one sequence "
                "at a time, or group inputs of equal length."
            )

        num_flow_steps = num_flow_steps if num_flow_steps is not None else config.num_flow_steps
        guidance_scale = guidance_scale if guidance_scale is not None else config.guidance_scale

        # The prompt already carries the reference-voice run as `audio_token_id` placeholders, so
        # the embeddings are scattered into position rather than prepended.
        if input_ids is not None:
            inputs_embeds = self.embed_text_tokens(input_ids)
            if voice_embeddings is not None:
                voice_embeddings = voice_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
                mask = self.get_placeholder_mask(input_ids, inputs_embeds, voice_embeddings)
                inputs_embeds = inputs_embeds.masked_scatter(mask, voice_embeddings)
            if audio_codes is not None:
                inputs_embeds = torch.cat([self.backbone_model.embed_tokens(audio_codes), inputs_embeds], dim=1)
        else:
            inputs_embeds = self.backbone_model.embed_tokens(audio_codes)

        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        outputs = self.backbone_model(inputs_embeds=inputs_embeds, use_cache=True)
        past_key_values = outputs.past_key_values
        backbone_hidden = outputs.last_hidden_state[:, -1, :]

        offset = config.num_special_audio_tokens
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        collected_frames = []

        for _ in range(max_new_tokens):
            semantic_code = self._sample_semantic_code(backbone_hidden)
            acoustic_codes = self._sample_acoustic_codes(
                backbone_hidden,
                num_flow_steps=num_flow_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            # Once a sequence has ended, its acoustic codebooks carry [EMPTY_AUDIO]. The reference
            # assigns the token *before* applying the special-token offset, so the stored code is
            # `empty_audio_token_id + num_special_audio_tokens`, not the bare token id.
            ended = semantic_code == config.end_audio_token_id
            empty_audio_code = config.empty_audio_token_id + config.num_special_audio_tokens
            acoustic_codes = torch.where(
                ended.unsqueeze(-1),
                torch.full_like(acoustic_codes, empty_audio_code),
                acoustic_codes,
            )

            frame_codes = torch.cat([semantic_code.unsqueeze(-1), acoustic_codes], dim=-1)
            collected_frames.append(frame_codes)

            unfinished = unfinished & ~ended
            if not bool(unfinished.any()):
                break

            outputs = self.backbone_model(
                inputs_embeds=self.backbone_model.embed_tokens(frame_codes.unsqueeze(1)),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            backbone_hidden = outputs.last_hidden_state[:, -1, :]

        if not collected_frames:
            empty = torch.zeros(batch_size, 0, config.num_codebooks, dtype=torch.long, device=device)
            return VoxtralTtsGenerateOutput(
                audio=[torch.zeros(0, device=device)] * batch_size if output_audio else None,
                audio_codes=list(empty.unbind(0)),
                semantic_tokens=[frame[:, 0] for frame in empty.unbind(0)],
            )

        all_frames = torch.stack(collected_frames, dim=1)

        # Trim each sequence at its own [END_AUDIO] frame.
        per_sequence_codes, per_sequence_semantic = [], []
        for i in range(batch_size):
            semantic = all_frames[i, :, 0]
            end_positions = (semantic == config.end_audio_token_id).nonzero()
            cutoff = int(end_positions[0]) if end_positions.numel() > 0 else all_frames.shape[1]
            per_sequence_codes.append(all_frames[i, :cutoff])
            per_sequence_semantic.append((all_frames[i, :cutoff, 0] - offset).clamp(min=0))

        audio = None
        if output_audio:
            audio = []
            for codes, semantic in zip(per_sequence_codes, per_sequence_semantic):
                if codes.shape[0] == 0:
                    audio.append(torch.zeros(0, device=device))
                    continue
                acoustic = (codes[:, 1:] - offset).clamp(min=0)
                waveform = self.codec_model.decode(semantic.unsqueeze(0), acoustic.unsqueeze(0))
                audio.append(waveform[0, 0])

        return VoxtralTtsGenerateOutput(
            audio=audio,
            audio_codes=per_sequence_codes,
            semantic_tokens=per_sequence_semantic,
        )

    def _sample_semantic_code(self, backbone_hidden: torch.Tensor) -> torch.LongTensor:
        """Greedy semantic codebook prediction straight off the backbone hidden state.

        `[EMPTY_AUDIO]` is never a legal prediction, and the head's output is padded up to a
        multiple of 128, so the padding tail is masked out too -- otherwise the argmax could land on
        a row that was never trained.
        """
        config = self.config
        logits = self.flow_matching_transformer.predict_semantic_logits(backbone_hidden).float()
        logits[:, config.empty_audio_token_id] = float("-inf")
        logits[:, config.num_special_audio_tokens + config.semantic_codebook_size :] = float("-inf")
        return logits.argmax(dim=-1)

    def _sample_acoustic_codes(
        self,
        backbone_hidden: torch.Tensor,
        num_flow_steps: int,
        guidance_scale: float,
        generator: torch.Generator | None = None,
    ) -> torch.LongTensor:
        """Integrate the flow-matching ODE for one frame and scalar-quantise the result.

        Conditional and unconditional velocities are evaluated in a single batched forward pass; the
        unconditional branch is the same network given a zeroed hidden state.
        """
        config = self.config
        batch_size = backbone_hidden.shape[0]
        device, dtype = backbone_hidden.device, backbone_hidden.dtype
        acoustic_dim = config.flow_matching_config.acoustic_dim

        sample = torch.randn(batch_size, acoustic_dim, device=device, dtype=dtype, generator=generator)
        # Built in fp32 then cast: step sizes such as 1/7 are not representable in bf16.
        timesteps = torch.linspace(0, 1, num_flow_steps + 1, device=device, dtype=torch.float32).to(dtype)

        use_guidance = guidance_scale != 1.0
        conditioning = (
            torch.cat([backbone_hidden, torch.zeros_like(backbone_hidden)]) if use_guidance else backbone_hidden
        )

        for step in range(num_flow_steps):
            dt = timesteps[step + 1] - timesteps[step]
            t = timesteps[step].expand(batch_size)

            if use_guidance:
                velocity = self.flow_matching_transformer(
                    hidden_states=conditioning,
                    timesteps=torch.cat([t, t]),
                    acoustic_embeddings=torch.cat([sample, sample]),
                )
                cond, uncond = velocity[:batch_size], velocity[batch_size:]
                velocity = guidance_scale * cond + (1.0 - guidance_scale) * uncond
            else:
                velocity = self.flow_matching_transformer(
                    hidden_states=conditioning, timesteps=t, acoustic_embeddings=sample
                )

            sample = sample + velocity * dt

        levels = config.acoustic_codebook_size
        scaled = ((sample.clamp(-1.0, 1.0) + 1.0) / 2.0) * (levels - 1)
        return scaled.round().long() + config.num_special_audio_tokens


__all__ = ["VoxtralTtsGenerateOutput", "VoxtralTtsGenerationMixin"]
