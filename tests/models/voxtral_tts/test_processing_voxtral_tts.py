# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import unittest

from transformers import VoxtralTtsProcessor
from transformers.testing_utils import require_mistral_common, require_torch, slow
from transformers.utils import is_mistral_common_available


if is_mistral_common_available():
    from mistral_common.protocol.speech.request import SpeechRequest


@slow
@require_torch
@require_mistral_common
class VoxtralTtsProcessorTest(unittest.TestCase):
    """The prompt layout is owned by `mistral_common`, so these tests pin us to it exactly.

    Voxtral TTS is sensitive to prompt structure in a way that is invisible at the tensor level: a
    prompt that merely looks plausible still produces audio, just not the right audio. So the
    assertions here are equality against `mistral_common`'s own output rather than shape checks.
    """

    checkpoint = "mistralai/Voxtral-4B-TTS-2603"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.processor = VoxtralTtsProcessor.from_pretrained(cls.checkpoint)

    def _reference_tokens(self, text, voice):
        mistral_tokenizer = getattr(self.processor.tokenizer, "tokenizer", self.processor.tokenizer)
        encoder = getattr(mistral_tokenizer, "instruct", None) or mistral_tokenizer.instruct_tokenizer
        return list(encoder.encode_speech_request(SpeechRequest(input=text, voice=voice)).tokens)

    def test_prompt_matches_mistral_common_exactly(self):
        for text, voice in [
            ("Hello, my name is Voxtral.", "neutral_female"),
            ("Bonjour, comment allez-vous?", "fr_male"),
            ("A somewhat longer sentence, to be sure the text segment is not truncated.", "casual_female"),
        ]:
            with self.subTest(voice=voice):
                inputs = self.processor(text, voice_preset=voice, return_tensors=None)
                self.assertEqual(inputs["input_ids"][0], self._reference_tokens(text, voice))

    def test_voice_embeddings_match_placeholder_run(self):
        """One preset embedding row per audio placeholder -- the scatter depends on this."""
        inputs = self.processor("Hello, my name is Voxtral.", voice_preset="neutral_female", return_tensors=None)
        token_ids = inputs["input_ids"][0]
        num_placeholders = sum(1 for token in token_ids if token == self.processor.audio_token_id)

        voice_embeddings = inputs["voice_embeddings"]
        self.assertEqual(voice_embeddings.ndim, 2)
        self.assertEqual(voice_embeddings.shape[0], num_placeholders)
        # Presets are pre-computed embeddings, not codes; a float dtype is what distinguishes them.
        self.assertTrue(voice_embeddings.dtype.is_floating_point)

    def test_prompt_is_well_formed(self):
        """The layout the model was trained on: reference audio first, then text, then 'now generate'."""
        inputs = self.processor("Hello, my name is Voxtral.", voice_preset="neutral_female", return_tensors=None)
        token_ids = inputs["input_ids"][0]
        audio_token_id = self.processor.audio_token_id

        self.assertGreater(token_ids.count(audio_token_id), 0)
        first_audio = token_ids.index(audio_token_id)
        last_audio = len(token_ids) - 1 - token_ids[::-1].index(audio_token_id)
        # The placeholders form one contiguous run, ahead of the text.
        self.assertEqual(last_audio - first_audio + 1, token_ids.count(audio_token_id))
        self.assertLess(last_audio, len(token_ids) - 1)

    def test_rejects_batched_text(self):
        with self.assertRaises(ValueError):
            self.processor(["one", "two"], voice_preset="neutral_female")

    def test_requires_text(self):
        with self.assertRaises(ValueError):
            self.processor(None, voice_preset="neutral_female")
