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

import json
from pathlib import Path

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...utils import (
    auto_docstring,
    is_mistral_common_available,
    is_soundfile_available,
    is_torch_available,
    logging,
    requires_backends,
)
from ...utils.hub import cached_file
from ..auto import AutoTokenizer


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf

if is_mistral_common_available():
    from mistral_common.protocol.speech.request import SpeechRequest

    from ...tokenization_mistral_common import MistralCommonBackend


# Fallback id for the audio placeholder token, used only when the tokenizer does not expose its
# audio encoder. Matches `audio_model_args.audio_token_id` in the checkpoint's params.json.
DEFAULT_AUDIO_TOKEN_ID = 24

# Fallback preset list. `from_pretrained` prefers the `voice:` mapping in the checkpoint's
# params.json, which is authoritative; this covers checkpoints that do not ship one.
KNOWN_VOICE_PRESETS = [
    "ar_male",
    "casual_female",
    "casual_male",
    "cheerful_female",
    "de_female",
    "de_male",
    "es_female",
    "es_male",
    "fr_female",
    "fr_male",
    "hi_female",
    "hi_male",
    "it_female",
    "it_male",
    "neutral_female",
    "neutral_male",
    "nl_female",
    "nl_male",
    "pt_female",
    "pt_male",
]


@auto_docstring
class VoxtralTtsProcessor(ProcessorMixin):
    r"""
    Constructs a Voxtral TTS processor which wraps a tokenizer and provides voice preset loading
    for the Voxtral TTS text-to-speech model.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            An instance of a tokenizer (e.g., `MistralCommonBackend` / Tekken).
        voice_presets (`dict[str, str]`, *optional*):
            Dictionary mapping voice preset names to their file paths relative to the checkpoint directory,
            plus a `"repo_or_path"` key pointing to the model repository or local directory.
    """

    VOICE_PRESET_DIR = "voice_embedding"

    def __init__(self, tokenizer, voice_presets=None):
        r"""
        voice_presets (`dict[str, str]`, *optional*):
            Dictionary mapping voice preset names to their file paths relative to the checkpoint directory,
            plus a `"repo_or_path"` key pointing to the model repository or local directory. Typically populated
            automatically by [`~VoxtralTtsProcessor.from_pretrained`].
        """
        super().__init__(tokenizer)
        self.voice_presets = voice_presets or {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a Voxtral TTS processor from a pretrained model checkpoint.

        Loads the tokenizer and discovers available voice presets from the `voice_embedding/` directory.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Either a model id on huggingface.co or a local directory path.
            **kwargs:
                Additional keyword arguments passed to the tokenizer's `from_pretrained` method.
        """
        token = kwargs.get("token")
        # The released checkpoint ships `tekken.json` but no `tokenizer_config.json`, so
        # `AutoTokenizer` cannot infer `model_type` and would fall back to a plain tokenizers
        # backend -- which has no speech-request API. Ask for the mistral-common backend directly.
        if is_mistral_common_available():
            tokenizer = MistralCommonBackend.from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            logger.warning(
                "`mistral_common` is not installed, so prompts cannot be built for Voxtral TTS. "
                "Install it with `pip install mistral-common`."
            )
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        voice_presets = {"repo_or_path": pretrained_model_name_or_path}
        for name in cls._candidate_voice_names(pretrained_model_name_or_path, token=token):
            rel_path = f"{cls.VOICE_PRESET_DIR}/{name}.pt"
            resolved = cached_file(
                pretrained_model_name_or_path,
                rel_path,
                token=token,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            if resolved is not None:
                voice_presets[name] = rel_path

        if len(voice_presets) == 1:
            logger.warning(
                f"No voice presets found in `{pretrained_model_name_or_path}/{cls.VOICE_PRESET_DIR}/`. "
                "Voice preset features will not be available."
            )

        return cls(tokenizer=tokenizer, voice_presets=voice_presets)

    @staticmethod
    def _candidate_voice_names(pretrained_model_name_or_path, token=None) -> list[str]:
        """Voice names from the checkpoint's `params.json`, falling back to the known list."""
        params_path = cached_file(
            pretrained_model_name_or_path,
            "params.json",
            token=token,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if params_path is not None:
            try:
                with open(params_path) as handle:
                    params = json.load(handle)
                voices = params["multimodal"]["audio_tokenizer_args"]["voice"]
                if voices:
                    return sorted(voices)
            except (KeyError, TypeError, ValueError, OSError):
                logger.warning("Could not read the voice list from params.json; using the built-in list.")
        return KNOWN_VOICE_PRESETS

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the processor's tokenizer to the specified directory.

        Voice presets are not re-saved since they are loaded directly from the original checkpoint.
        """
        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)

    def _load_voice_preset(self, voice_preset, **kwargs):
        """
        Load a voice preset tensor from a preset name, file path, or Hub-relative path.

        Args:
            voice_preset (`str`):
                Either a known voice preset name (e.g. `"neutral_female"`), a path to a local `.pt` file,
                or a Hub-relative path.

        Returns:
            `torch.Tensor`: Audio codes tensor of shape `(num_frames, num_codebooks)`.
        """
        token = kwargs.get("token")

        if voice_preset in self.voice_presets:
            rel_path = self.voice_presets[voice_preset]
            resolved = cached_file(
                self.voice_presets.get("repo_or_path", "."),
                rel_path,
                token=token,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            if resolved is None:
                raise ValueError(
                    f"Voice preset '{voice_preset}' was registered but could not be loaded from "
                    f"`{self.voice_presets.get('repo_or_path')}/{rel_path}`."
                )
        elif Path(voice_preset).exists():
            resolved = voice_preset
        else:
            rel_path = f"{self.VOICE_PRESET_DIR}/{voice_preset}.pt"
            resolved = cached_file(
                self.voice_presets.get("repo_or_path", "."),
                rel_path,
                token=token,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            if resolved is None:
                raise ValueError(
                    f"Voice preset '{voice_preset}' not found. Available presets: {self.available_voice_presets}. "
                    "You can also pass a path to a local `.pt` file."
                )

        return torch.load(resolved, map_location="cpu", weights_only=True)

    @property
    def available_voice_presets(self) -> list[str]:
        """Returns a list of available voice preset names."""
        return [k for k in self.voice_presets if k != "repo_or_path"]

    @property
    def audio_token_id(self) -> int:
        """Id of the audio placeholder token that reference-voice embeddings are scattered onto.

        Taken from the tokenizer's audio encoder when it exposes one, which is authoritative; the
        constant is the value shipped with `mistral_common` for this model family.
        """
        mistral_tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        encoder = getattr(mistral_tokenizer, "instruct", None) or getattr(
            mistral_tokenizer, "instruct_tokenizer", None
        )
        audio_encoder = getattr(encoder, "audio_encoder", None)
        special_ids = getattr(audio_encoder, "special_ids", None)
        if special_ids is not None and getattr(special_ids, "audio", None) is not None:
            return special_ids.audio
        return DEFAULT_AUDIO_TOKEN_ID

    def _speech_request_encoder(self):
        """The `mistral_common` object carrying `encode_speech_request`.

        `MistralCommonBackend` wraps a `MistralTokenizer`, which exposes the speech API as
        `instruct_tokenizer`; vLLM-Omni spells the same thing `instruct`, so accept either.
        """
        requires_backends(self, ["mistral-common"])
        mistral_tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        encoder = getattr(mistral_tokenizer, "instruct", None) or getattr(
            mistral_tokenizer, "instruct_tokenizer", None
        )
        if encoder is None or not hasattr(encoder, "encode_speech_request"):
            raise ValueError(
                "The tokenizer does not expose `encode_speech_request`. Voxtral TTS needs a "
                "`mistral_common` tokenizer new enough to support speech requests."
            )
        return encoder

    @auto_docstring
    def __call__(
        self,
        text=None,
        voice_preset=None,
        return_tensors="pt",
        **kwargs,
    ) -> BatchFeature:
        r"""
        text (`str`):
            The text to synthesize.
        voice_preset (`str`, *optional*):
            Name of a voice preset (e.g. `"neutral_female"`). The prompt layout depends on the
            voice, because the reference-audio placeholder run is as long as that voice's embedding,
            so the preset is resolved here rather than being an independent argument.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of tensors to return.

        Returns:
            [`BatchFeature`] with:
            - **input_ids** -- the full speech-request prompt, as produced by `mistral_common`:
              `[BOS, begin_audio, audio x T, ..., text, ..., begin_audio]`.
            - **voice_embeddings** -- *(optional)* `(T, hidden_size)` reference-voice embeddings,
              one row per `audio_token_id` position in `input_ids`. These are **embeddings, not
              codes**: the shipped presets are pre-computed activations, so they are scattered into
              the prompt rather than looked up in the audio codebook table.
        """
        if text is None:
            raise ValueError("You must provide `text` to the VoxtralTtsProcessor.")
        if isinstance(text, (list, tuple)):
            raise ValueError(
                "VoxtralTtsProcessor handles one prompt at a time; pass a single string. Batched "
                "synthesis is not supported yet."
            )

        encoder = self._speech_request_encoder()
        request = (
            SpeechRequest(input=text, voice=voice_preset) if voice_preset is not None else SpeechRequest(input=text)
        )
        token_ids = list(encoder.encode_speech_request(request).tokens)

        data = {"input_ids": [token_ids], "attention_mask": [[1] * len(token_ids)]}

        if voice_preset is not None:
            voice_embeddings = self._load_voice_preset(voice_preset, **kwargs)
            if voice_embeddings.dim() != 2:
                raise ValueError(
                    f"Expected voice preset embeddings of shape (num_frames, hidden_size), got "
                    f"{tuple(voice_embeddings.shape)}."
                )
            num_placeholders = sum(1 for token in token_ids if token == self.audio_token_id)
            if num_placeholders != voice_embeddings.shape[0]:
                raise ValueError(
                    f"Voice preset '{voice_preset}' has {voice_embeddings.shape[0]} frames but the "
                    f"prompt reserves {num_placeholders} audio placeholders. The preset file and "
                    "the tokenizer disagree about this voice."
                )
            data["voice_embeddings"] = voice_embeddings

        return BatchFeature(data=data, tensor_type=return_tensors)

    def save_audio(self, audio, saving_path, sampling_rate=24000):
        """
        Save an audio waveform tensor to a file.

        Args:
            audio (`torch.Tensor` or `np.ndarray`):
                Audio waveform to save. If a tensor, it will be converted to numpy.
            saving_path (`str` or `Path`):
                Output file path (e.g. `"output.wav"`).
            sampling_rate (`int`, *optional*, defaults to 24000):
                Sample rate of the audio in Hz.
        """
        if not is_soundfile_available():
            raise ImportError("Please install `soundfile` to save audio files: `pip install soundfile`.")

        if is_torch_available() and isinstance(audio, torch.Tensor):
            audio = audio.cpu().float().numpy()

        sf.write(str(saving_path), audio, sampling_rate)

    @property
    def model_input_names(self):
        return self.tokenizer.model_input_names


__all__ = ["VoxtralTtsProcessor"]
