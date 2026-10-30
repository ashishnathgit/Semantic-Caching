"""Settings, read from config.toml at the project root.

Every value has a default here, so a missing key in config.toml is fine.
Set the environment variable SEMCACHE_CONFIG to use a different file, or
SEMCACHE_ARTIFACTS to keep the built model and cache somewhere else.

Profiles. ``[profiles.<name>.<section>]`` tables in config.toml override the base
settings, e.g. ``[profiles.quora.cache] lsa_weight = 0.5``. The profile is chosen by
``--profile`` on the command line, else SEMCACHE_PROFILE, else ``[profiles] default``.
Each profile keeps its own model, cache, evaluation and metrics in
artifacts/<profile>/; users and their conversations are shared in artifacts/users/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .embedder import EmbedderConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathsConfig:
    artifacts_dir: str = "artifacts"
    faq_csv: str = "data/faq.csv"
    quora_train: str = "data/quora/train.csv"
    quora_eval: str = "data/quora/validation.csv"


@dataclass
class CorpusConfig:
    use_faq: bool = True
    use_quora: bool = True           # used only if the Quora file exists
    quora_max_questions: int = 100000
    require_quora: bool = False      # fail the build instead of falling back to the FAQ only
    seed: int = 42


@dataclass
class CacheConfig:
    threshold: float = 0.70
    min_known_ratio: float = 0.5
    lsa_weight: float = 1.0
    min_latent_coverage: float = 0.30   # pairs with a smaller rho are scored by word overlap only
    max_entries: int = 5000
    top_k: int = 5
    use_lsh: bool = False
    lsh_bits: int = 12
    lsh_tables: int = 8
    lsh_min_entries: int = 1000
    autosave_every: int = 10
    warm_with_faq: bool = True


@dataclass
class LLMConfig:
    backend: str = "ollama"          # "ollama" or "stub"
    fallback_to_stub: bool = True
    ollama_url: str = "http://localhost:11434"
    model: str = "auto"
    temperature: float = 0.2
    timeout_s: float = 120.0
    system_prompt: str = "Answer clearly and concisely in 2-4 sentences."
    cache_stub_answers: bool = True     # only when backend = "stub"; fallback answers are never cached
    stub_delay_ms: float = 0.0


@dataclass
class UsersConfig:
    detect_follow_ups: bool = True     # follow-ups skip the shared cache and use the thread history
    history_messages: int = 8          # earlier messages sent to the LLM with a follow-up
    history_chars: int = 6000          # ... and at most this many characters of them
    max_threads_per_user: int = 100    # the least recently used thread is dropped beyond this
    max_messages_per_thread: int = 200


@dataclass
class APIConfig:
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173",
                                                             "http://127.0.0.1:5173"])


@dataclass
class MetricsConfig:
    enabled: bool = True                       # record events in artifacts/<profile>/metrics/
    bucket_seconds: int = 10                   # default time-series bucket
    # What-if prices for "Est. LLM cost" / "Est. savings" (Ollama is free locally). None = unset.
    price_per_1k_input_tokens: float | None = None
    price_per_1k_output_tokens: float | None = None
    cost_per_llm_call_usd: float | None = None
    price_note: str = ""                       # shown next to the estimate, e.g. "assumed: <model> list price"


@dataclass
class EvalConfig:
    pairs: str = "auto"              # "faq", "quora", or "auto" (Quora when the file exists)
    max_pairs: int = 20000
    ranks: list[int] = field(default_factory=lambda: [10, 25, 50, 100, 150, 200])
    cache_sim_pairs: int = 5000
    latency_sizes: list[int] = field(default_factory=lambda: [1000, 5000, 20000, 50000])
    seed: int = 7


ADMIN_KEY_ENV = "SEMCACHE_ADMIN_KEY"
PROFILE_ENV = "SEMCACHE_PROFILE"
BACKEND_ENV = "SEMCACHE_LLM_BACKEND"     # overrides [llm] backend, e.g. "stub" if Ollama fails on stage
DEFAULT_PROFILE = "faq"


@dataclass
class Settings:
    root: Path = PROJECT_ROOT
    profile: str = DEFAULT_PROFILE
    profiles: list[str] = field(default_factory=list)   # the profiles config.toml defines
    paths: PathsConfig = field(default_factory=PathsConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    users: UsersConfig = field(default_factory=UsersConfig)
    api: APIConfig = field(default_factory=APIConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # resolved paths ---------------------------------------------------------
    def path(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.root / p

    @property
    def artifacts_root(self) -> Path:
        """Shared by all profiles (holds artifacts/users)."""
        override = os.environ.get("SEMCACHE_ARTIFACTS")
        return Path(override).expanduser() if override else self.path(self.paths.artifacts_dir)

    @property
    def artifacts_dir(self) -> Path:
        """This profile's model, cache, evaluation and metrics."""
        return self.artifacts_root / self.profile

    @property
    def model_dir(self) -> Path:
        return self.artifacts_dir / "model"

    @property
    def cache_dir(self) -> Path:
        return self.artifacts_dir / "cache"

    @property
    def users_dir(self) -> Path:
        return self.artifacts_root / "users"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def eval_dir(self) -> Path:
        return self.artifacts_dir / "eval"

    # loading ----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, profile: str | None = None) -> "Settings":
        path = Path(path or os.environ.get("SEMCACHE_CONFIG") or PROJECT_ROOT / "config.toml")
        settings = cls(root=path.resolve().parent if path.exists() else PROJECT_ROOT)
        raw: dict = {}
        if path.exists():
            with open(path, "rb") as fh:
                raw = tomllib.load(fh)
        profiles = dict(raw.get("profiles", {}))
        default = profiles.pop("default", DEFAULT_PROFILE)
        settings.profiles = sorted(name for name, table in profiles.items() if isinstance(table, dict))
        name = (profile or os.environ.get(PROFILE_ENV) or default).strip().lower()
        if settings.profiles and name not in settings.profiles:
            raise ValueError(f"unknown profile {name!r}; config.toml defines {', '.join(settings.profiles)}")
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"profile names may only use letters, digits, '-' and '_': {name!r}")
        settings.profile = name
        settings._apply(raw, "config.toml")
        settings._apply(profiles.get(name, {}), f"config.toml [profiles.{name}]")
        backend = os.environ.get(BACKEND_ENV, "").strip().lower()
        if backend:
            if backend not in ("ollama", "stub"):
                raise ValueError(f"{BACKEND_ENV} must be ollama or stub, not {backend!r}")
            settings.llm.backend = backend
        return settings

    def _apply(self, raw: dict, where: str) -> None:
        for section in fields(self):
            if section.name in ("root", "profile", "profiles") or section.name not in raw:
                continue
            target = getattr(self, section.name)
            allowed = {f.name for f in fields(target)}
            unknown = set(raw[section.name]) - allowed
            if unknown:
                raise ValueError(f"{where} [{section.name}]: unknown keys {sorted(unknown)}")
            for key, value in raw[section.name].items():
                setattr(target, key, value)
        extra = set(raw) - {f.name for f in fields(self)} - {"profiles"}
        if extra:
            raise ValueError(f"{where}: unknown sections {sorted(extra)}")
