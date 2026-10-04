# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Voxtral TTS model."""

import copy
import os
import shutil
import tempfile
import unittest

import pytest

from transformers import VoxtralTtsConfig, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import VoxtralTtsForTextToSpeech
    from transformers.models.voxtral_tts.convert_voxtral_tts_weights_to_hf import write_model


class VoxtralTtsModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=5,
        num_audio_frames=3,
        is_training=False,
        num_codebooks=37,
        hidden_size=64,
        config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "silu",
            "max_position_embeddings": 64,
            "audio_vocab_size": 384,
            "semantic_codebook_size": 64,
            "acoustic_codebook_size": 5,
            "n_acoustic_codebook": 36,
            "num_codebooks": 37,
            "tie_word_embeddings": True,
        },
        flow_matching_config={
            "input_dim": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "semantic_vocab_size": 128,
            "acoustic_dim": 36,
            "sigma": 1e-5,
            "sigma_max": 1.0,
        },
        codec_config={
            "semantic_codebook_size": 64,
            "semantic_dim": 32,
            "acoustic_codebook_size": 5,
            "acoustic_dim": 36,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "patch_size": 8,
            "patch_proj_kernel_size": 3,
            "decoder_transformer_lengths": [1, 1, 1, 1],
            "decoder_conv_kernels": [3, 4, 4, 4],
            "decoder_conv_strides": [1, 2, 2, 2],
            "rms_norm_eps": 0.01,
            "qk_norm_eps": 1e-6,
            "layer_scale_init": 0.01,
        },
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_audio_frames = num_audio_frames
        self.is_training = is_training
        self.num_codebooks = num_codebooks
        self.config_dict = config
        self.flow_matching_config = flow_matching_config
        self.codec_config = codec_config

        self.num_hidden_layers = config["num_hidden_layers"]
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.encoder_seq_length = self.num_audio_frames + self.seq_length

    def get_config(self):
        return VoxtralTtsConfig(
            flow_matching_config=self.flow_matching_config,
            codec_config=self.codec_config,
            **self.config_dict,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)

        audio_codes = torch.stack(
            [
                ids_tensor([self.batch_size, self.num_audio_frames], config.semantic_codebook_size),
            ]
            + [
                ids_tensor([self.batch_size, self.num_audio_frames], config.acoustic_codebook_size)
                for _ in range(config.n_acoustic_codebook)
            ],
            dim=-1,
        )

        attention_mask = torch.ones(
            self.batch_size, self.num_audio_frames + self.seq_length, dtype=torch.long, device=torch_device
        )

        return config, input_ids, audio_codes, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, audio_codes, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "audio_codes": audio_codes,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_forward(self, config, input_ids, audio_codes, attention_mask):
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids=input_ids, audio_codes=audio_codes, attention_mask=attention_mask)
        expected_seq_len = self.num_audio_frames + self.seq_length
        self.parent.assertEqual(result.logits.shape, (self.batch_size, expected_seq_len, config.vocab_size))

    def create_and_check_forward_text_only(self, config, input_ids, audio_codes, attention_mask):
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids=input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, config.vocab_size))

    def create_and_check_forward_audio_only(self, config, input_ids, audio_codes, attention_mask):
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        result = model(audio_codes=audio_codes)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_audio_frames, config.vocab_size))


@require_torch
class VoxtralTtsModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VoxtralTtsForTextToSpeech,) if is_torch_available() else ()
    all_generative_model_classes = ()
    test_resize_embeddings = False
    test_resize_embeddings_untied = False

    def setUp(self):
        self.model_tester = VoxtralTtsModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VoxtralTtsConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (
                    self.model_tester.batch_size,
                    self.model_tester.num_audio_frames + self.model_tester.seq_length,
                ),
                dtype=torch.long,
                device=torch_device,
            )
        return inputs_dict

    def test_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward(*config_and_inputs)

    def test_forward_text_only(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_text_only(*config_and_inputs)

    def test_forward_audio_only(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_audio_only(*config_and_inputs)

    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    @pytest.mark.skip(reason="Voxtral TTS has custom embedding approach (text and audio embeddings).")
    def test_model_get_set_embeddings(self):
        pass

    @pytest.mark.skip(reason="Voxtral TTS has no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass

    @pytest.mark.skip(reason="Voxtral TTS has special text+audio embeddings that are never tied in the standard way.")
    def test_tied_weights_keys(self):
        pass

    @pytest.mark.skip(
        reason="VoxtralTtsCodecConfig uses strict dataclass validation and layer_scale field type "
        "conflicts with the test's type coercion."
    )
    def test_can_load_ignoring_mismatched_shapes(self):
        pass

    @pytest.mark.skip(
        reason="inputs_embeds test passes only text-shaped embeds but prepare_config_and_inputs "
        "produces audio+text inputs. The two paths produce different sequence lengths."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @pytest.mark.skip(
        reason="TP plan paths use backbone_model.layers.* but the test resolves "
        "base_model_prefix='model' to backbone_model.backbone_model.layers.*. "
        "This is a known modular converter issue with composite models."
    )
    def test_tp_plan_matches_params(self):
        pass

    def test_state_survives_from_pretrained(self):
        """Everything the checkpoint supplies must survive the meta-device load path.

        Two distinct failure modes live here, and this model has shipped three instances between
        them:

        1. a buffer computed in `__init__` is absent from the checkpoint, so `from_pretrained` --
           which builds on the meta device and materialises from the checkpoint -- leaves it
           pointing at uninitialised memory (the flow-matching `inv_freq`, the codec's ALiBi slopes);
        2. `_init_weights` runs after materialisation and overwrites trained state with an
           initialiser (the semantic VQ codebook's centroids, and the codec's layer-scale vectors).

        Both are near-silent: the model loads, runs, and produces plausible-looking garbage. The
        second is why this test perturbs state before saving -- comparing a fresh model against its
        own reload cannot catch an initialiser that writes the *same* constant `__init__` does.
        """
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.eval()

        # Give every checkpointed tensor a distinctive value, so an initialiser that overwrites it
        # after loading is visible as a mismatch rather than a coincidence.
        checkpointed = set(model.state_dict())
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                if name in checkpointed and tensor.is_floating_point():
                    tensor.copy_(torch.rand(tensor.shape, generator=generator).to(tensor.dtype) + 0.5)

        expected = {k: v.clone() for k, v in model.state_dict().items()}
        fresh_buffers = {k: v.clone() for k, v in model.named_buffers()}

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            reloaded = VoxtralTtsForTextToSpeech.from_pretrained(tmpdir)

        actual = reloaded.state_dict()
        self.assertEqual(set(expected), set(actual), "checkpointed tensor set changed across save/load")
        for name, want in expected.items():
            torch.testing.assert_close(
                actual[name],
                want,
                msg=lambda m, n=name: (
                    f"'{n}' differs after from_pretrained: {m}\n"
                    "Something is overwriting checkpoint state -- most likely `_init_weights`, "
                    "which runs after materialisation and must never touch anything the "
                    "checkpoint supplies."
                ),
            )

        # Non-persistent buffers are absent from the checkpoint by design, so they must instead be
        # reconstructed identically from the config rather than left uninitialised.
        loaded_buffers = dict(reloaded.named_buffers())
        self.assertEqual(set(fresh_buffers), set(loaded_buffers), "buffer set changed across save/load")
        for name, want in fresh_buffers.items():
            if name in checkpointed:
                continue
            got = loaded_buffers[name]
            self.assertTrue(
                torch.isfinite(got).all(),
                f"non-persistent buffer '{name}' holds non-finite values after from_pretrained -- "
                "it was computed in __init__ and never restored by the load path",
            )
            torch.testing.assert_close(
                got, want, msg=lambda m, n=name: f"non-persistent buffer '{n}' differs: {m}"
            )

    def test_generate_with_tiny_model(self):
        """The custom TTS generate loop runs end to end and returns per-sequence outputs."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 5], config.vocab_size).to(torch_device)

        output = model.generate(input_ids=input_ids, max_new_tokens=2)
        self.assertEqual(len(output.audio), 1)
        self.assertEqual(len(output.audio_codes), 1)
        self.assertEqual(output.audio_codes[0].shape[-1], config.num_codebooks)
        self.assertLessEqual(output.audio_codes[0].shape[0], 2)

    def test_generate_emitted_codes_are_in_range(self):
        """Every emitted code must be addressable in its own codebook block.

        The audio embedding table lays codebooks out contiguously, so an out-of-range code silently
        reads a neighbouring codebook's rows instead of raising.
        """
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 5], config.vocab_size).to(torch_device)
        output = model.generate(input_ids=input_ids, max_new_tokens=3, output_audio=False)

        codes = output.audio_codes[0]
        offset = config.num_special_audio_tokens
        self.assertTrue(bool((codes[:, 0] < config.semantic_codebook_size + offset).all()))
        self.assertTrue(bool((codes[:, 1:] >= offset).all()))
        self.assertTrue(bool((codes[:, 1:] < config.acoustic_codebook_size + offset).all()))
        # The semantic codebook may never predict [EMPTY_AUDIO].
        self.assertTrue(bool((codes[:, 0] != config.empty_audio_token_id).all()))

    def test_generate_stops_at_end_audio(self):
        """Predicting [END_AUDIO] terminates generation well before the frame budget."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        # Force the semantic head to always choose [END_AUDIO].
        original = model.flow_matching_transformer.predict_semantic_logits

        def always_end(backbone_hidden):
            logits = original(backbone_hidden)
            logits[:, config.end_audio_token_id] = 1.0
            return logits

        model.flow_matching_transformer.predict_semantic_logits = always_end

        input_ids = ids_tensor([1, 5], config.vocab_size).to(torch_device)
        output = model.generate(input_ids=input_ids, max_new_tokens=64, output_audio=False)

        # The terminating frame is dropped, so nothing is emitted.
        self.assertEqual(output.audio_codes[0].shape[0], 0)

    def test_generate_without_audio_output(self):
        """`output_audio=False` returns codes but skips the codec."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 5], config.vocab_size).to(torch_device)

        output = model.generate(input_ids=input_ids, max_new_tokens=2, output_audio=False)
        self.assertIsNone(output.audio)
        self.assertIsNotNone(output.audio_codes)
        self.assertIsNotNone(output.semantic_tokens)

    def test_generate_with_voice_reference(self):
        """Voice reference frames are accepted and prepended to the text."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 5], config.vocab_size).to(torch_device)

        audio_codes = torch.stack(
            [ids_tensor([1, 3], config.semantic_codebook_size)]
            + [ids_tensor([1, 3], config.acoustic_codebook_size) for _ in range(config.n_acoustic_codebook)],
            dim=-1,
        ).to(torch_device)

        output = model.generate(input_ids=input_ids, audio_codes=audio_codes, max_new_tokens=2)
        self.assertEqual(len(output.audio), 1)

    def test_generate_scatters_voice_embeddings(self):
        """Voice presets are embeddings scattered onto placeholders, not codes looked up in a table."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        num_ref_frames = 4
        input_ids = ids_tensor([1, 8], config.vocab_size).to(torch_device)
        input_ids[0, 1 : 1 + num_ref_frames] = config.audio_token_id
        voice_embeddings = torch.randn(num_ref_frames, config.hidden_size, device=torch_device)

        output = model.generate(
            input_ids=input_ids, voice_embeddings=voice_embeddings, max_new_tokens=2, output_audio=False
        )
        self.assertIsNotNone(output.audio_codes)

    def test_voice_embeddings_length_must_match_placeholders(self):
        """A preset of the wrong length must raise rather than scatter silently."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 8], config.vocab_size).to(torch_device)
        input_ids[0, 1:4] = config.audio_token_id  # 3 placeholders
        voice_embeddings = torch.randn(5, config.hidden_size, device=torch_device)  # 5 frames

        with self.assertRaises(ValueError):
            model.generate(input_ids=input_ids, voice_embeddings=voice_embeddings, max_new_tokens=1)

    def test_generate_with_cfg(self):
        """Classifier-free guidance runs the batched conditional/unconditional pass."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 5], config.vocab_size).to(torch_device)

        output = model.generate(input_ids=input_ids, max_new_tokens=2, guidance_scale=1.2)
        self.assertEqual(len(output.audio), 1)

    def test_generate_rejects_padded_batches(self):
        """Padded batches are refused rather than silently generating from padding."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([2, 5], config.vocab_size).to(torch_device)
        attention_mask = torch.ones(2, 5, dtype=torch.long, device=torch_device)
        attention_mask[0, :2] = 0

        with self.assertRaises(NotImplementedError):
            model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=2)

    def test_semantic_head_reads_backbone_hidden_state(self):
        """The semantic codebook is predicted from the backbone hidden state, not the ODE output.

        Guards the wiring: `predict_semantic_logits` must be a function of the backbone hidden state
        alone, independent of the flow-matching sample.
        """
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        hidden = torch.randn(2, config.hidden_size, device=torch_device)
        with torch.no_grad():
            first = model.flow_matching_transformer.predict_semantic_logits(hidden)
            second = model.flow_matching_transformer.predict_semantic_logits(hidden)

        self.assertEqual(first.shape, (2, config.flow_matching_config.semantic_vocab_size))
        torch.testing.assert_close(first, second)


@require_torch_accelerator
class VoxtralTtsIntegrationTest(unittest.TestCase):
    model_checkpoint = "mistralai/Voxtral-4B-TTS-2603"
    converted_checkpoint_dir = None
    _cleanup_converted_checkpoint = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.converted_checkpoint_dir = os.environ.get("VOXTRAL_TTS_HF_MODEL_DIR")
        if cls.converted_checkpoint_dir is None:
            cls.converted_checkpoint_dir = tempfile.mkdtemp(prefix="voxtral-tts-hf-")
            cls._cleanup_converted_checkpoint = True
            write_model(cls.model_checkpoint, cls.converted_checkpoint_dir)

    @classmethod
    def tearDownClass(cls):
        if cls._cleanup_converted_checkpoint and cls.converted_checkpoint_dir is not None:
            shutil.rmtree(cls.converted_checkpoint_dir, ignore_errors=True)
        super().tearDownClass()

    def _get_model(self):
        model = VoxtralTtsForTextToSpeech.from_pretrained(
            self.converted_checkpoint_dir, device_map=torch_device, torch_dtype=torch.float16
        )
        model.eval()
        return model

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_forward_logits(self):
        """Test that forward pass matches a known logits slice from converted original weights."""
        model = self._get_model()

        input_ids = torch.tensor([[1, 306, 4658, 278, 6593, 310]], device=torch_device)
        with torch.no_grad():
            out = model(input_ids=input_ids)

        self.assertEqual(out.logits.shape[0], 1)
        self.assertEqual(out.logits.shape[1], 6)
        self.assertEqual(out.logits.shape[2], 131072)
        expected_mean = torch.tensor([[0.0539, 0.1095, 0.0874, 0.1063, 0.0758, 0.1047]])
        expected_last_logits = torch.tensor(
            [
                0.9287,
                -0.0992,
                -2.8809,
                1.0098,
                0.4404,
                0.9316,
                0.9287,
                0.9287,
                0.9287,
                0.4939,
                0.9287,
                0.9316,
                2.2695,
                0.1075,
                0.9287,
                0.9287,
                0.9287,
                0.9287,
                0.9321,
                0.9287,
            ]
        )
        torch.testing.assert_close(out.logits.float().cpu().mean(-1), expected_mean, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            out.logits[0, -1, :20].float().cpu(),
            expected_last_logits,
            rtol=1e-3,
            atol=1e-3,
        )

    @slow
    def test_generate_produces_audio(self):
        """Structural checks on a real generation.

        The exact sample values are deliberately not asserted here. The expectations this test
        previously carried were recorded from an implementation whose semantic head, acoustic
        quantisation and codebook offsets were all wrong -- they encoded the bug rather than
        detecting it. Numeric expectations must be regenerated against the reference implementation
        (vLLM-Omni) before being reinstated; see the parity harness described in the porting notes.
        """
        model = self._get_model()

        input_ids = torch.tensor([[1, 306, 4658, 278, 6593, 310]], device=torch_device)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        output = model.generate(input_ids=input_ids, max_new_tokens=5)

        self.assertEqual(len(output.audio), 1)
        self.assertEqual(len(output.audio_codes), 1)

        codes = output.audio_codes[0]
        config = model.config
        offset = config.num_special_audio_tokens
        self.assertEqual(codes.shape[-1], config.num_codebooks)
        self.assertLessEqual(codes.shape[0], 5)

        # Codes must address their own codebook block, and acoustic levels must span the FSQ range
        # rather than collapsing onto {0, 1} the way unscaled rounding does.
        self.assertTrue(bool((codes[:, 0] < config.semantic_codebook_size + offset).all()))
        self.assertTrue(bool((codes[:, 1:] >= offset).all()))
        self.assertTrue(bool((codes[:, 1:] < config.acoustic_codebook_size + offset).all()))

        if codes.shape[0] > 0:
            acoustic_levels = (codes[:, 1:] - offset).float()
            self.assertGreater(acoustic_levels.max().item(), 1.0, "acoustic levels collapsed -- quantisation is wrong")
            self.assertEqual(output.audio[0].numel(), codes.shape[0] * int(config.sampling_rate / config.frame_rate))
