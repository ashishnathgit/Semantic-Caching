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
    include_cached: bool = False     # also fit on every shared cached question joined with its answer


@dataclass
class CacheConfig:
    threshold: float = 0.70
    min_known_ratio: float = 0.5
    # Topic-word rule (embedder.KnownWordPolicy): a question failing min_known_ratio still gets a
    # semantic match, on its known topic words alone, when unknown everyday words are not counted
    # and it has >= min_topic_words known topic words making up >= min_topic_ratio of them.
    relaxed_known_words: bool = True
    min_topic_words: int = 2
    min_topic_ratio: float = 0.4
    lsa_weight: float = 1.0
    min_latent_coverage: float = 0.30   # pairs with a smaller rho are scored by word overlap only
    max_entries: int = 5000
    top_k: int = 5
    use_lsh: bool = False
    lsh_bits: int = 10               # b and L from `measure` (LSH sweep): recall >= 0.95 at 10k and 50k
    lsh_tables: int = 24
    lsh_min_entries: int = 50000     # below this, one brute-force product C e is as fast as LSH
    autosave_every: int = 10
    warm_with_faq: bool = True
    ttl_seconds: float = 0.0           # default lifetime of a new entry; 0 = never expires (off)
    per_entry_thresholds: bool = False # experimental: each entry learns its own tau from feedback
    entry_precision_target: float = 0.95
    rebuild_every: int = 0             # with corpus.include_cached: refit after this many new entries (0 = never)
    negation_guard: bool = False       # refuse a semantic hit whose negation / antonym / number / name differs
    antonyms: list[str] = field(default_factory=lambda: [
        "on/off", "enable/disable", "increase/decrease", "allow/block", "include/exclude",
        "before/after", "true/false"])


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
    # Online model (backend or chain entry "openai_compatible"). The base URL, model and key come from
    # .env (SEMCACHE_OPENAI_BASE_URL, SEMCACHE_OPENAI_MODEL, SEMCACHE_OPENAI_API_KEY); the two below are
    # defaults when .env doesn't set them. The key is only read from the environment, never from here.
    openai_base_url: str = ""
    openai_model: str = ""
    openai_api_key_env: str = "SEMCACHE_OPENAI_API_KEY"
    openai_max_tokens: int | None = None
    chain: list[str] = field(default_factory=list)   # e.g. ["openai_compatible", "ollama", "stub"]; empty =
                                                     # backend (+ the stub when fallback_to_stub)
    model_params_billion: float | None = None   # for the Matrix Lab's FLOP estimate; None = read the
                                                # size from the model tag ("llama3.2:3b" -> 3)


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
    # `measure` traffic-mix simulation (measurement only; nothing here changes behaviour)
    traffic_mix: list[float] = field(default_factory=lambda: [0.18, 0.47, 0.35])  # exact, reworded, new
    traffic_queries: int = 10000
    traffic_cached_share: float = 0.65       # share of FAQ items in the simulated cache
    llm_usd_per_call: float = 0.001          # assumed price of one LLM call (an estimate)
    cpu_usd_per_hour: float = 0.05           # assumed price of one CPU core-hour, prices a lookup


ADMIN_KEY_ENV = "SEMCACHE_ADMIN_KEY"
PROFILE_ENV = "SEMCACHE_PROFILE"
BACKEND_ENV = "SEMCACHE_LLM_BACKEND"     # overrides [llm] backend, e.g. "stub" if Ollama fails on stage
DEFAULT_PROFILE = "faq"


def load_env_file(path: Path) -> list[str]:
    """Read KEY=value lines from a .env file into os.environ (variables already set win).
    Returns the names it set. Values are never printed or logged."""
    if not path.is_file():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            names.append(key)
    return names


IMPORTED_PROFILE_FILE = "profile.toml"


def valid_profile_name(name: str) -> bool:
    return bool(name) and len(name) <= 40 and name.replace("_", "").replace("-", "").isalnum()


def imported_profiles(artifacts_root: Path, exclude: set[str] = frozenset()) -> dict[str, dict]:
    """{name: override table} for every artifacts/<name>/profile.toml (made by import_faq_csv)."""
    out: dict[str, dict] = {}
    if not artifacts_root.is_dir():
        return out
    for path in sorted(artifacts_root.glob(f"*/{IMPORTED_PROFILE_FILE}")):
        name = path.parent.name
        if name in exclude or not valid_profile_name(name):
            continue
        try:
            with open(path, "rb") as fh:
                out[name] = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            continue
    return out


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
        load_env_file(settings.root / ".env")
        raw: dict = {}
        if path.exists():
            with open(path, "rb") as fh:
                raw = tomllib.load(fh)
        profiles = dict(raw.get("profiles", {}))
        default = profiles.pop("default", DEFAULT_PROFILE)
        configured = sorted(name for name, table in profiles.items() if isinstance(table, dict))
        # Profiles created by "Import FAQ CSV" live in artifacts/<name>/profile.toml, never in config.toml.
        settings.paths.artifacts_dir = raw.get("paths", {}).get("artifacts_dir", settings.paths.artifacts_dir)
        imported = imported_profiles(settings.artifacts_root, exclude=set(configured))
        settings.profiles = sorted(set(configured) | set(imported))
        name = (profile or os.environ.get(PROFILE_ENV) or default).strip().lower()
        if settings.profiles and name not in settings.profiles:
            raise ValueError(f"unknown profile {name!r}; config.toml defines {', '.join(configured)}"
                             + (f" and these were imported: {', '.join(sorted(imported))}" if imported else ""))
        if not valid_profile_name(name):
            raise ValueError(f"profile names may only use letters, digits, '-' and '_': {name!r}")
        if name in imported:
            profiles[name] = imported[name]
        settings.profile = name
        settings._apply(raw, "config.toml")
        settings._apply(profiles.get(name, {}), f"config.toml [profiles.{name}]")
        backend = os.environ.get(BACKEND_ENV, "").strip().lower()
        if backend:
            if backend not in ("ollama", "stub", "openai_compatible"):
                raise ValueError(f"{BACKEND_ENV} must be ollama, openai_compatible or stub, not {backend!r}")
            settings.llm.backend = backend
            settings.llm.chain = []                # an explicit override beats the chain
        for name in settings.llm.chain:
            if name not in ("ollama", "stub", "openai_compatible"):
                raise ValueError(f"[llm] chain: unknown backend {name!r}")
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
