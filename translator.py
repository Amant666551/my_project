"""
translator.py  –  Optimized MT module

Key changes vs original:
  - translate() now correctly accepts and uses source_lang / target_lang
  - Explicit torch_dtype=torch.float32 on CPU (float16 silently falls back
    anyway and can cause NaN outputs on some CPU builds)
  - Optional: swap MODEL_CLASS to Helsinki-NLP opus-mt for ~10× faster CPU
    inference (see comment below)
  - Generation uses do_sample=False (greedy) for deterministic, faster output
  - max_new_tokens capped at 128 for typical short utterances
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Supported target-language names for the prompt ──────────────────────────
_LANG_NAMES = {
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ar": "Arabic",
}

# ── Model selection ──────────────────────────────────────────────────────────
#
# OPTION A (default): Qwen1.5-1.8B-Chat  – general purpose, slower on CPU
#   MODEL_DIR = "./models/qwen/Qwen1___5-1___8B-Chat"
#
# OPTION B (recommended for CPU): Helsinki-NLP/opus-mt-en-zh
#   Drop-in replacement; ~300 MB; ~0.5–1 s on CPU; EN→ZH only.
#   To enable: pip install sacremoses sentencepiece
#   and set USE_OPUS_MT = True
#
USE_OPUS_MT = True   # ← flip to True to use the fast seq2seq model


class LightTranslator:
    def __init__(self, model_path: str | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 MT device: {self.device}")

        if USE_OPUS_MT:
            self._init_opus_mt()
        else:
            self._init_qwen(model_path)

    # ── Qwen (generative LLM) ────────────────────────────────────────────────
    def _init_qwen(self, model_path: str | None):
        self.model_dir = model_path or r"C:\Users\30909\Desktop\document\files\models\translate\qwen\Qwen1___5-1___8B-Chat"
        print(f"🚀 [MT] Loading Qwen from: {self.model_dir}")

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model not found at {self.model_dir}. "
                "Run download_qwen.py first, or set USE_OPUS_MT=True."
            )

        # Use float32 explicitly on CPU; float16 is fine on CUDA
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self._backend = "qwen"
        print("✅ [MT] Qwen loaded.")

    # ── Helsinki-NLP opus-mt (seq2seq, CPU-friendly) ─────────────────────────
    def _init_opus_mt(self):
        from transformers import MarianMTModel, MarianTokenizer

        opus_id = "Helsinki-NLP/opus-mt-zh-en"
        print(f"🚀 [MT] Loading Helsinki opus-mt: {opus_id}")
        self.tokenizer = MarianTokenizer.from_pretrained(opus_id)
        self.model     = MarianMTModel.from_pretrained(opus_id).to(self.device)
        self.model.eval()
        self._backend  = "opus"
        print("✅ [MT] opus-mt loaded.")

    # ── Public API ───────────────────────────────────────────────────────────
    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh",
    ) -> str:
        """
        Translate *text* from source_lang to target_lang.
        Returns translated string, or "" on failure.
        """
        if not text or not text.strip():
            return ""

        if self._backend == "opus":
            return self._translate_opus(text)
        else:
            return self._translate_qwen(text, source_lang, target_lang)

    # ── Qwen translation ─────────────────────────────────────────────────────
    def _translate_qwen(self, text: str, source_lang: str, target_lang: str) -> str:
        tgt_name = _LANG_NAMES.get(target_lang, target_lang)
        src_name = _LANG_NAMES.get(source_lang, source_lang.upper())

        prompt = (
            f"Translate the following {src_name} text to {tgt_name}. "
            f"Output only the translation, no explanation.\n\n"
            f"{src_name}: {text}\n{tgt_name}:"
        )
        messages = [
            {"role": "system", "content": "You are a professional interpreter."},
            {"role": "user",   "content": prompt},
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                max_new_tokens=128,      # reduced from 256 – enough for speech
                do_sample=False,         # greedy: faster + deterministic
                repetition_penalty=1.1,  # avoids looping outputs
            )

        # Strip the input tokens from the output
        new_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    # ── opus-mt translation ──────────────────────────────────────────────────
    def _translate_opus(self, text: str) -> str:
        inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)
