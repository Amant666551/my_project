from __future__ import annotations

import json
import logging
import os
import threading
import warnings
from dataclasses import dataclass
from itertools import count

import numpy as np
import torch

from app_logging import get_logger


warnings.filterwarnings(
    "ignore",
    message=r"The `force_filename` parameter is deprecated.*",
    category=FutureWarning,
    module=r"huggingface_hub\.file_download",
)
warnings.filterwarnings(
    "ignore",
    message=r"`local_dir_use_symlinks` parameter is deprecated and will be ignored.*",
    category=UserWarning,
    module=r"huggingface_hub\.file_download",
)
warnings.filterwarnings(
    "ignore",
    message=r".*local file already exists\. Defaulting to existing file\..*",
    category=UserWarning,
    module=r"huggingface_hub\.file_download",
)

logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class RegisteredSpeaker:
    label: str
    sample_alias: str
    voice_id: str
    sample_path: str
    target_model: str
    preferred_name: str
    active: bool
    embedding: np.ndarray
    prototypes: tuple[np.ndarray, ...]


@dataclass
class SessionSpeaker:
    speaker_id: str
    centroid: np.ndarray
    turns: int = 1
    matched_registry: RegisteredSpeaker | None = None


@dataclass
class SpeakerDecision:
    speaker_id: str
    voice_id: str | None = None
    registry_label: str | None = None
    registry_sample_alias: str | None = None
    session_score: float = 0.0
    registry_score: float = 0.0
    registry_margin: float = 0.0
    best_registry_label: str | None = None
    best_registry_score: float = 0.0
    second_registry_label: str | None = None
    second_registry_score: float = 0.0
    is_new_session: bool = False
    is_new_registry_match: bool = False


class SpeakerMatcher:
    @classmethod
    def from_env(cls, *, base_dir: str, sample_rate: int) -> "SpeakerMatcher":
        registry_path = os.getenv(
            "SPEAKER_REGISTRY_PATH",
            os.path.join(base_dir, "voice_samples", "voice_registry.json"),
        )
        target_model = os.getenv("QWEN_TTS_MODEL", "qwen3-tts-vc-realtime-2026-01-15").strip()
        return cls(
            base_dir=base_dir,
            registry_path=registry_path,
            target_model=target_model,
            sample_rate=sample_rate,
        )

    def __init__(
        self,
        *,
        base_dir: str,
        registry_path: str,
        target_model: str,
        sample_rate: int,
    ):
        self.base_dir = base_dir
        self.registry_path = registry_path
        self.target_model = target_model
        self.sample_rate = sample_rate
        self.enabled = _env_bool("SPEAKER_MATCHING_ENABLED", True)
        self.embedding_sample_rate = _env_int("SPEAKER_EMBEDDING_SAMPLE_RATE", 16000)
        self.cluster_threshold = self._load_threshold(
            "SPEAKER_CLUSTER_THRESHOLD",
            default=0.72,
            legacy_default=0.82,
        )
        self.registry_threshold = self._load_threshold(
            "SPEAKER_REGISTRY_THRESHOLD",
            default=0.70,
            legacy_default=0.80,
        )
        self.registry_margin_threshold = _env_float("SPEAKER_REGISTRY_MARGIN", 0.03)
        self.session_margin_threshold = _env_float("SPEAKER_CLUSTER_MARGIN", 0.02)
        self.min_duration_sec = _env_float("SPEAKER_MIN_DURATION_SEC", 0.5)
        self.active_model_only = _env_bool("SPEAKER_ACTIVE_MODEL_ONLY", True)
        self.prototype_window_sec = _env_float("SPEAKER_PROTOTYPE_WINDOW_SEC", 2.5)
        self.max_prototypes = max(1, _env_int("SPEAKER_MAX_PROTOTYPES", 4))
        self.session_update_alpha = min(0.95, max(0.05, _env_float("SPEAKER_SESSION_UPDATE_ALPHA", 0.35)))
        self.model_source = os.getenv("SPEAKER_SPEECHBRAIN_SOURCE", "speechbrain/spkrec-ecapa-voxceleb").strip()
        self.hf_home = os.getenv("HF_HOME", os.path.join(base_dir, "models", "huggingface"))
        self.hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
        self.model_cache_dir = os.getenv(
            "SPEAKER_SPEECHBRAIN_CACHE_DIR",
            os.path.join(base_dir, "models", "speechbrain", "spkrec-ecapa-voxceleb"),
        )
        self._model_source_resolved = self.model_source
        self.device = os.getenv(
            "SPEAKER_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        ).strip()
        self._registry: list[RegisteredSpeaker] = []
        self._sessions: list[SessionSpeaker] = []
        self._registry_sessions: dict[str, SessionSpeaker] = {}
        self._known_speaker_counter = count(1)
        self._guest_speaker_counter = count(1)
        self._registry_speaker_ids: dict[str, str] = {}
        self._lock = threading.Lock()
        self._torchaudio = None
        self._soundfile = None
        self._classifier = None
        self._min_samples = max(1, int(self.min_duration_sec * self.sample_rate))
        self._log = get_logger("SPEAKER")
        self._init_backend()
        self._load_registry()

    def _load_threshold(self, name: str, *, default: float, legacy_default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        if abs(value - legacy_default) < 1e-6:
            self._log = get_logger("SPEAKER")
            self._log.info(
                "speaker config | %s=%s detected from legacy matcher; using speechbrain default %.2f instead",
                name,
                raw,
                default,
            )
            return default
        return value

    def _init_backend(self) -> None:
        if not self.enabled:
            self._log.info("speaker matcher disabled by config")
            return

        os.makedirs(self.hf_home, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", self.hf_home)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
        if os.path.exists(os.path.join(self.model_cache_dir, "hyperparams.yaml")):
            self._model_source_resolved = self.model_cache_dir

        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for SpeechBrain speaker matching.") from exc
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required for SpeechBrain speaker matching.") from exc
        self._torchaudio = torchaudio
        self._soundfile = sf

        encoder_cls = None
        local_strategy = None
        fetch_config_cls = None
        import_error = None
        try:
            from speechbrain.inference.classifiers import EncoderClassifier as encoder_cls
            from speechbrain.utils.fetching import FetchConfig as fetch_config_cls
            from speechbrain.utils.fetching import LocalStrategy as local_strategy
        except ImportError as exc:
            import_error = exc
            try:
                from speechbrain.inference.speaker import EncoderClassifier as encoder_cls
                from speechbrain.utils.fetching import FetchConfig as fetch_config_cls
                from speechbrain.utils.fetching import LocalStrategy as local_strategy
            except ImportError:
                try:
                    from speechbrain.pretrained import EncoderClassifier as encoder_cls
                    from speechbrain.utils.fetching import FetchConfig as fetch_config_cls
                    from speechbrain.utils.fetching import LocalStrategy as local_strategy
                except ImportError:
                    pass
        if encoder_cls is None or local_strategy is None or fetch_config_cls is None:
            raise RuntimeError(
                "SpeechBrain is required for speaker matching. "
                "Install project requirements to enable the SpeechBrain backend."
            ) from import_error

        fetch_config = fetch_config_cls(
            allow_network=(self._model_source_resolved == self.model_source),
            allow_updates=False,
        )

        try:
            self._classifier = encoder_cls.from_hparams(
                source=self._model_source_resolved,
                savedir=self.model_cache_dir,
                fetch_config=fetch_config,
                local_strategy=local_strategy.COPY_SKIP_CACHE,
                run_opts={"device": self.device},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SpeechBrain speaker model '{self._model_source_resolved}': {exc}. "
                f"Cache dir: {self.model_cache_dir}. "
                "If Hugging Face is unreachable, set HF_ENDPOINT to an accessible mirror "
                "or pre-download the model into the cache directory."
            ) from exc

        self._log.info(
            "speaker backend | backend=speechbrain | source=%s | device=%s | cache=%s | hf_endpoint=%s",
            self._model_source_resolved,
            self.device,
            self.model_cache_dir,
            self.hf_endpoint or "default",
        )

    def _load_registry(self) -> None:
        if not self.enabled:
            return
        if not os.path.exists(self.registry_path):
            self._log.warning("speaker registry not found: %s", self.registry_path)
            self.enabled = False
            return

        try:
            with open(self.registry_path, "r", encoding="utf-8") as fp:
                entries = json.load(fp)
        except Exception as exc:
            self._log.warning("failed to load speaker registry: %s", exc)
            self.enabled = False
            return

        for index, entry in enumerate(entries, start=1):
            target_model = str(entry.get("target_model", "")).strip()
            if self.active_model_only and target_model != self.target_model:
                continue
            voice_id = str(entry.get("qwen_tts_voice", "")).strip()
            sample_path = str(entry.get("sample_path", "")).strip()
            if not voice_id or not sample_path:
                continue
            full_path = os.path.join(self.base_dir, sample_path.replace("/", os.sep))
            if not os.path.exists(full_path):
                self._log.warning("speaker sample missing, skipping: %s", full_path)
                continue
            try:
                audio = self._load_audio(full_path)
                embedding = self._extract_embedding(audio, self.sample_rate)
                prototypes = self._extract_prototypes(audio, self.sample_rate)
            except Exception as exc:
                self._log.warning("speaker sample failed for %s: %s", full_path, exc)
                continue
            if embedding is None:
                self._log.warning("speaker sample too short or invalid, skipping: %s", full_path)
                continue
            if not prototypes:
                prototypes = (embedding,)
            preferred_name = str(entry.get("preferred_name", "")).strip() or f"voice_{index}"
            sample_alias = os.path.splitext(os.path.basename(sample_path))[0]
            registry_label = sample_alias
            self._registry.append(
                RegisteredSpeaker(
                    label=registry_label,
                    sample_alias=sample_alias,
                    voice_id=voice_id,
                    sample_path=sample_path,
                    target_model=target_model,
                    preferred_name=preferred_name,
                    active=bool(entry.get("active")),
                    embedding=embedding,
                    prototypes=prototypes,
                )
            )

        if not self._registry:
            self._log.warning(
                "speaker matcher enabled but no usable registry voices were loaded for model=%s",
                self.target_model,
            )
            self.enabled = False
            return

        self._log.info(
            "speaker registry loaded | count=%s | model=%s | labels=%s",
            len(self._registry),
            self.target_model,
            ",".join(item.label for item in self._registry),
        )

    def _load_audio(self, path: str) -> np.ndarray:
        audio, sample_rate = self._soundfile.read(path, dtype="float32", always_2d=False)
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)
        if sample_rate != self.sample_rate:
            tensor = torch.from_numpy(samples).unsqueeze(0)
            tensor = self._torchaudio.functional.resample(tensor, sample_rate, self.sample_rate)
            samples = tensor.squeeze(0).cpu().numpy().astype(np.float32)
        return samples.reshape(-1).astype(np.float32)

    def _resample_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate == self.embedding_sample_rate:
            return samples
        tensor = torch.from_numpy(samples).unsqueeze(0)
        tensor = self._torchaudio.functional.resample(
            tensor,
            sample_rate,
            self.embedding_sample_rate,
        )
        return tensor.squeeze(0).cpu().numpy().astype(np.float32)

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None
        if samples.size < self._min_samples:
            samples = np.pad(samples, (0, self._min_samples - samples.size))
        samples = self._resample_audio(samples, sample_rate)
        peak = float(np.max(np.abs(samples)))
        if peak > 1e-5:
            samples = samples / peak
        return samples.astype(np.float32)

    def _extract_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        if self._classifier is None:
            return None
        prepared = self._prepare_audio(audio, sample_rate)
        if prepared is None:
            return None
        wav = torch.from_numpy(prepared).unsqueeze(0).to(self.device)
        wav_lens = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            embedding = self._classifier.encode_batch(wav, wav_lens)
        vector = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            return None
        return (vector / norm).astype(np.float32)

    def _extract_prototypes(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, ...]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ()
        window_samples = max(int(self.prototype_window_sec * sample_rate), self._min_samples)
        if samples.size <= window_samples:
            embedding = self._extract_embedding(samples, sample_rate)
            return (embedding,) if embedding is not None else ()

        max_offset = max(0, samples.size - window_samples)
        prototype_count = min(self.max_prototypes, max(2, samples.size // max(window_samples, 1)))
        starts = np.linspace(0, max_offset, num=prototype_count, dtype=int)
        seen: set[int] = set()
        prototypes: list[np.ndarray] = []
        for start in starts:
            start_int = int(start)
            if start_int in seen:
                continue
            seen.add(start_int)
            segment = samples[start_int : start_int + window_samples]
            embedding = self._extract_embedding(segment, sample_rate)
            if embedding is not None:
                prototypes.append(embedding)
        if not prototypes:
            embedding = self._extract_embedding(samples, sample_rate)
            return (embedding,) if embedding is not None else ()
        return tuple(prototypes)

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    def _registry_similarity(self, embedding: np.ndarray, speaker: RegisteredSpeaker) -> float:
        centroid_score = self._cosine_similarity(embedding, speaker.embedding)
        if not speaker.prototypes:
            return centroid_score
        prototype_scores = sorted(
            (self._cosine_similarity(embedding, prototype) for prototype in speaker.prototypes),
            reverse=True,
        )
        top_scores = prototype_scores[: min(2, len(prototype_scores))]
        prototype_score = float(sum(top_scores) / len(top_scores)) if top_scores else 0.0
        return max(centroid_score, (prototype_score * 0.75) + (centroid_score * 0.25))

    def _best_session(self, embedding: np.ndarray) -> tuple[SessionSpeaker | None, float, float]:
        if not self._sessions:
            return None, 0.0, 0.0
        ranked = sorted(
            (
                (speaker, self._cosine_similarity(embedding, speaker.centroid))
                for speaker in self._sessions
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        best_speaker, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        return best_speaker, max(best_score, 0.0), max(second_score, 0.0)

    def _best_registry(
        self, embedding: np.ndarray
    ) -> tuple[
        RegisteredSpeaker | None,
        float,
        RegisteredSpeaker | None,
        float,
    ]:
        if not self._registry:
            return None, 0.0, None, 0.0
        ranked = sorted(
            (
                (speaker, self._registry_similarity(embedding, speaker))
                for speaker in self._registry
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        best_speaker, best_score = ranked[0]
        second_speaker = ranked[1][0] if len(ranked) > 1 else None
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        return best_speaker, max(best_score, 0.0), second_speaker, max(second_score, 0.0)

    def _allocate_speaker_id(self, matched_registry: RegisteredSpeaker | None) -> str:
        if matched_registry is not None:
            existing = self._registry_speaker_ids.get(matched_registry.label)
            if existing is not None:
                return existing
            speaker_id = f"speaker_{next(self._known_speaker_counter)}"
            self._registry_speaker_ids[matched_registry.label] = speaker_id
            return speaker_id
        return f"guest_{next(self._guest_speaker_counter)}"

    def _create_session(self, embedding: np.ndarray, matched_registry: RegisteredSpeaker | None = None) -> SessionSpeaker:
        session = SessionSpeaker(
            speaker_id=self._allocate_speaker_id(matched_registry),
            centroid=embedding.astype(np.float32),
            matched_registry=matched_registry,
        )
        self._sessions.append(session)
        if matched_registry is not None:
            self._registry_sessions[matched_registry.label] = session
        return session

    def _update_session(self, session: SessionSpeaker, embedding: np.ndarray) -> float:
        score = self._cosine_similarity(embedding, session.centroid)
        updated = (
            (1.0 - self.session_update_alpha) * session.centroid
            + (self.session_update_alpha * embedding)
        )
        norm = float(np.linalg.norm(updated))
        if norm > 0.0:
            session.centroid = (updated / norm).astype(np.float32)
        session.turns += 1
        return max(score, 0.0)

    def _bind_registry_session(
        self,
        session: SessionSpeaker,
        registry: RegisteredSpeaker | None,
    ) -> bool:
        if registry is None:
            return False
        previous = self._registry_sessions.get(registry.label)
        session.speaker_id = self._allocate_speaker_id(registry)
        session.matched_registry = registry
        self._registry_sessions[registry.label] = session
        return previous is None or previous.speaker_id != session.speaker_id

    def match_utterance(self, audio: np.ndarray | None) -> SpeakerDecision | None:
        if not self.enabled or audio is None:
            return None
        try:
            embedding = self._extract_embedding(audio, self.sample_rate)
        except Exception as exc:
            self._log.warning("speaker embedding extraction failed: %s", exc)
            return None
        if embedding is None:
            return None

        with self._lock:
            (
                registry_speaker,
                registry_score,
                second_registry_speaker,
                second_registry_score,
            ) = self._best_registry(embedding)
            registry_margin = registry_score - second_registry_score
            registry_candidate = None
            if (
                registry_speaker is not None
                and registry_score >= self.registry_threshold
                and registry_margin >= self.registry_margin_threshold
            ):
                registry_candidate = registry_speaker

            bound_registry_session = None
            if registry_candidate is not None:
                bound_registry_session = self._registry_sessions.get(registry_candidate.label)

            session_speaker, session_score, second_session_score = self._best_session(embedding)
            force_new_session = False
            if (
                session_speaker is not None
                and registry_candidate is not None
                and bound_registry_session is None
            ):
                bound_registry = session_speaker.matched_registry
                if bound_registry is not None and bound_registry.label != registry_candidate.label:
                    bound_score = self._registry_similarity(embedding, bound_registry)
                    if registry_score >= bound_score + self.registry_margin_threshold:
                        force_new_session = True

            ambiguous_session = (
                session_speaker is not None
                and second_session_score > 0.0
                and (session_score - second_session_score) < self.session_margin_threshold
            )
            if bound_registry_session is not None:
                session_speaker = bound_registry_session
                session_score = self._update_session(session_speaker, embedding)
                is_new_session = False
                is_new_registry_match = self._bind_registry_session(session_speaker, registry_candidate)
            elif (
                session_speaker is None
                or session_score < self.cluster_threshold
                or force_new_session
                or ambiguous_session
            ):
                session_speaker = self._create_session(embedding, matched_registry=registry_candidate)
                session_score = 1.0
                is_new_session = True
                is_new_registry_match = registry_candidate is not None
            else:
                session_score = self._update_session(session_speaker, embedding)
                is_new_session = False
                is_new_registry_match = False
                if registry_candidate is not None:
                    is_new_registry_match = self._bind_registry_session(
                        session_speaker,
                        registry_candidate,
                    )

            chosen_registry = session_speaker.matched_registry
            chosen_registry_score = 0.0
            if chosen_registry is not None:
                chosen_registry_score = self._registry_similarity(embedding, chosen_registry)
                if (
                    chosen_registry.label != (registry_candidate.label if registry_candidate is not None else None)
                    and chosen_registry_score < max(0.0, self.registry_threshold - self.registry_margin_threshold)
                ):
                    chosen_registry = None
                    chosen_registry_score = 0.0

            decision = SpeakerDecision(
                speaker_id=session_speaker.speaker_id,
                voice_id=chosen_registry.voice_id if chosen_registry is not None else None,
                registry_label=chosen_registry.label if chosen_registry is not None else None,
                registry_sample_alias=chosen_registry.sample_alias if chosen_registry is not None else None,
                session_score=session_score,
                registry_score=chosen_registry_score,
                registry_margin=registry_margin,
                best_registry_label=registry_speaker.label if registry_speaker is not None else None,
                best_registry_score=registry_score,
                second_registry_label=(
                    second_registry_speaker.label if second_registry_speaker is not None else None
                ),
                second_registry_score=second_registry_score,
                is_new_session=is_new_session,
                is_new_registry_match=is_new_registry_match,
            )

        if decision.is_new_session:
            self._log.info(
                "speaker_map | session=%s | action=new_session | registry=%s | voice=%s",
                decision.speaker_id,
                decision.registry_label or "unmatched",
                decision.voice_id or "default",
            )
        if decision.is_new_registry_match:
            self._log.info(
                "speaker_map | session=%s | action=matched_registry | registry=%s | voice=%s | registry_score=%.3f | margin=%.3f",
                decision.speaker_id,
                decision.registry_label,
                decision.voice_id or "default",
                decision.registry_score,
                decision.registry_margin,
            )
        self._log.info(
            "speaker_turn | session=%s | registry=%s | voice=%s | session_score=%.3f | registry_score=%.3f | margin=%.3f | top1=%s:%.3f | top2=%s:%.3f",
            decision.speaker_id,
            decision.registry_label or "none",
            decision.voice_id or "default",
            decision.session_score,
            decision.registry_score,
            decision.registry_margin,
            decision.best_registry_label or "none",
            decision.best_registry_score,
            decision.second_registry_label or "none",
            decision.second_registry_score,
        )
        return decision
