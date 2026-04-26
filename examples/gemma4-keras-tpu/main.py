# ruff: noqa: E402
# %% [markdown]
# Kaggle's notebook VM has limited host memory, so we read the attached
# `gemma-4-31b-it` safetensors index tensor by tensor, place each tensor onto a
# 1-D JAX `model` mesh with `NamedSharding`, and keep all 60 layers resident in
# TPU HBM before generation starts.
#
# Sharding layout:
#
# ```
# q/gate/up       : P("model", None)
# k/v sliding     : P("model", None)
# k/v full        : replicated (4 KV heads cannot split over 8 chips)
# o/down          : P(None, "model")
# embedding       : P("model", None)
# norm/scalars    : replicated
# ```
#
# Default is one greedy token (`MAX_NEW_TOKENS=1`) so the run validates the
# full 31B route quickly. Resident weights make extra tokens cheap.
#
# Notebook-runtime env overrides (set inside the Kaggle notebook, not the local
# push command — local env vars do not propagate to the kernel):
#
# ```
# VALIDATE_ONLY=1     # validate config/index and skip generation
# LAYER_LIMIT=2       # debug the first N layers; output is not meaningful
# MAX_NEW_TOKENS=2    # generate more greedy tokens
# PROMPT="..."        # replace the default text prompt
# SHARD_WEIGHTS=0     # disable JAX NamedSharding for comparison/debugging
# SHARD_EMBEDDING=0   # keep tied embedding replicated while sharding layers
# PRINT_SHARDING=0    # silence the first-layer sharding summary
# ```

# %%
import gc
import os
import json
import math
import time
import typing
import pathlib
import dataclasses
import importlib.metadata

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import jax.numpy as jnp
import tokenizers
import safetensors

BF16 = jnp.bfloat16
MODEL_AXIS = "model"
LAYER_PREFIX = "model.language_model.layers"
EMBEDDING = "model.language_model.embed_tokens.weight"
FINAL_NORM = "model.language_model.norm.weight"


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclasses.dataclass(frozen=True)
class Gemma4TextConfig:
    attention_k_eq_v: bool
    final_logit_softcapping: float | None
    global_head_dim: int
    head_dim: int
    hidden_activation: str
    hidden_size: int
    intermediate_size: int
    layer_types: list[str]
    num_attention_heads: int
    num_global_key_value_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_parameters: dict[str, dict[str, typing.Any]]
    sliding_window: int
    vocab_size: int


@dataclasses.dataclass(frozen=True)
class LayerWeights:
    input_layernorm: jax.Array
    k_norm: jax.Array
    k_proj: jax.Array
    layer_scalar: jax.Array
    o_proj: jax.Array
    post_attention_layernorm: jax.Array
    post_feedforward_layernorm: jax.Array
    pre_feedforward_layernorm: jax.Array
    q_norm: jax.Array
    q_proj: jax.Array
    up_proj: jax.Array
    gate_proj: jax.Array
    down_proj: jax.Array
    v_proj: jax.Array | None


@dataclasses.dataclass(frozen=True)
class ResidentWeights:
    embedding: jax.Array
    layers: list[LayerWeights]
    norm: jax.Array


@dataclasses.dataclass(frozen=True)
class ShardingPlan:
    enabled: bool
    mesh: typing.Any | None
    replicated: typing.Any | None
    row_sharded: typing.Any | None
    column_sharded: typing.Any | None
    vocab_sharded: typing.Any | None

    @property
    def shard_count(self) -> int:
        if self.mesh is None:
            return 1
        return int(self.mesh.devices.size)


class TensorStore:
    def __init__(self, root: pathlib.Path):
        self.root = root
        index_path = root / "model.safetensors.index.json"
        self.index = json.loads(index_path.read_text())
        self.weight_map = self.index["weight_map"]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def tensor(self, name: str, dtype: typing.Any = BF16, sharding: typing.Any | None = None) -> jax.Array:
        shard = self.root / self.weight_map[name]
        with safetensors.safe_open(str(shard), framework="np") as handle:
            array = handle.get_tensor(name)
        if sharding is None:
            value = jnp.asarray(array)
        else:
            value = jax.device_put(array, sharding)
        if dtype is not None:
            value = value.astype(dtype)
        value.block_until_ready()
        del array
        return value

    def summary(self) -> str:
        files = sorted(set(self.weight_map.values()))
        total_parameters = self.index.get("metadata", {}).get("total_parameters")
        total_size = self.index.get("metadata", {}).get("total_size")
        return (
            f"{len(self.weight_map)} tensors across {len(files)} safetensors "
            f"shards; parameters={total_parameters}; bytes={total_size}"
        )


def find_model_dir() -> pathlib.Path:
    configured = os.environ.get("GEMMA4_MODEL_DIR")
    candidates = []
    if configured:
        candidates.append(pathlib.Path(configured))

    candidates.extend(
        [
            pathlib.Path("/kaggle/input/gemma-4/transformers/gemma-4-31b-it/1"),
            pathlib.Path("/kaggle/input/gemma-4-31b-it/transformers/default/1"),
        ]
    )

    for candidate in candidates:
        if (candidate / "model.safetensors.index.json").exists():
            return candidate

    input_root = pathlib.Path("/kaggle/input")
    matches = sorted(input_root.glob("**/model.safetensors.index.json"))
    if matches:
        return matches[0].parent

    raise FileNotFoundError(
        "Could not find model.safetensors.index.json. "
        "Attach google/gemma-4/transformers/gemma-4-31b-it/1 or set GEMMA4_MODEL_DIR."
    )


def load_config(root: pathlib.Path) -> tuple[dict[str, typing.Any], Gemma4TextConfig]:
    raw = json.loads((root / "config.json").read_text())
    text = raw["text_config"]
    return raw, Gemma4TextConfig(
        attention_k_eq_v=bool(text["attention_k_eq_v"]),
        final_logit_softcapping=text.get("final_logit_softcapping"),
        global_head_dim=int(text["global_head_dim"]),
        head_dim=int(text["head_dim"]),
        hidden_activation=str(text["hidden_activation"]),
        hidden_size=int(text["hidden_size"]),
        intermediate_size=int(text["intermediate_size"]),
        layer_types=list(text["layer_types"]),
        num_attention_heads=int(text["num_attention_heads"]),
        num_global_key_value_heads=int(text["num_global_key_value_heads"]),
        num_hidden_layers=int(text["num_hidden_layers"]),
        num_key_value_heads=int(text["num_key_value_heads"]),
        rms_norm_eps=float(text["rms_norm_eps"]),
        rope_parameters=dict(text["rope_parameters"]),
        sliding_window=int(text["sliding_window"]),
        vocab_size=int(text["vocab_size"]),
    )


def load_eos_ids(root: pathlib.Path, config: dict[str, typing.Any]) -> set[int]:
    source = config.get("eos_token_id", config["text_config"].get("eos_token_id", []))
    generation_config = root / "generation_config.json"
    if generation_config.exists():
        source = json.loads(generation_config.read_text()).get("eos_token_id", source)

    if isinstance(source, int):
        return {source}
    return {int(value) for value in source}


def build_sharding_plan() -> ShardingPlan:
    enabled = env_flag("SHARD_WEIGHTS", True)
    devices = np.array(jax.devices())
    if not enabled or devices.size < 2:
        return ShardingPlan(
            enabled=False,
            mesh=None,
            replicated=None,
            row_sharded=None,
            column_sharded=None,
            vocab_sharded=None,
        )

    mesh = jax.sharding.Mesh(devices.reshape((devices.size,)), (MODEL_AXIS,))
    partition = jax.sharding.PartitionSpec
    return ShardingPlan(
        enabled=True,
        mesh=mesh,
        replicated=jax.sharding.NamedSharding(mesh, partition()),
        row_sharded=jax.sharding.NamedSharding(mesh, partition(MODEL_AXIS, None)),
        column_sharded=jax.sharding.NamedSharding(mesh, partition(None, MODEL_AXIS)),
        vocab_sharded=jax.sharding.NamedSharding(mesh, partition(MODEL_AXIS, None)),
    )


def place(value: jax.Array, sharding: typing.Any | None) -> jax.Array:
    if sharding is None:
        return value
    return jax.device_put(value, sharding)


def sharding_spec(value: jax.Array) -> str:
    sharding = getattr(value, "sharding", None)
    spec = getattr(sharding, "spec", None)
    return str(spec)


def configured_layer_limit(config: Gemma4TextConfig) -> int:
    return int(os.environ.get("LAYER_LIMIT", str(config.num_hidden_layers)))


def required_layer_tensors(config: Gemma4TextConfig, layer_idx: int) -> list[str]:
    prefix = f"{LAYER_PREFIX}.{layer_idx}"
    names = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.layer_scalar",
        f"{prefix}.mlp.down_proj.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.post_feedforward_layernorm.weight",
        f"{prefix}.pre_feedforward_layernorm.weight",
        f"{prefix}.self_attn.k_norm.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.self_attn.q_norm.weight",
        f"{prefix}.self_attn.q_proj.weight",
    ]
    if uses_v_proj(config, layer_idx):
        names.append(f"{prefix}.self_attn.v_proj.weight")
    return names


def uses_v_proj(config: Gemma4TextConfig, layer_idx: int) -> bool:
    is_sliding = config.layer_types[layer_idx] == "sliding_attention"
    return not (config.attention_k_eq_v and not is_sliding)


def kv_projection_sharding(config: Gemma4TextConfig, sharding: ShardingPlan, layer_idx: int) -> typing.Any | None:
    is_sliding = config.layer_types[layer_idx] == "sliding_attention"
    kv_heads = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads
    if kv_heads % sharding.shard_count == 0:
        return sharding.row_sharded
    return sharding.replicated


def validate_index(store: TensorStore, config: Gemma4TextConfig) -> None:
    missing = []
    for name in [EMBEDDING, FINAL_NORM]:
        if not store.has(name):
            missing.append(name)
    for layer_idx in range(config.num_hidden_layers):
        missing.extend(name for name in required_layer_tensors(config, layer_idx) if not store.has(name))
    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"Missing {len(missing)} required tensors:\n{preview}")

    full_layers = sum(layer_type == "full_attention" for layer_type in config.layer_types)
    sliding_layers = config.num_hidden_layers - full_layers
    print(f"validated text tensors: {sliding_layers} sliding layers, {full_layers} full layers")


def load_layer(store: TensorStore, config: Gemma4TextConfig, sharding: ShardingPlan, layer_idx: int) -> LayerWeights:
    prefix = f"{LAYER_PREFIX}.{layer_idx}"
    attn = f"{prefix}.self_attn"
    mlp = f"{prefix}.mlp"
    kv_sharding = kv_projection_sharding(config, sharding, layer_idx)
    return LayerWeights(
        input_layernorm=store.tensor(f"{prefix}.input_layernorm.weight", sharding=sharding.replicated),
        k_norm=store.tensor(f"{attn}.k_norm.weight", sharding=sharding.replicated),
        k_proj=store.tensor(f"{attn}.k_proj.weight", sharding=kv_sharding),
        layer_scalar=store.tensor(f"{prefix}.layer_scalar", sharding=sharding.replicated),
        o_proj=store.tensor(f"{attn}.o_proj.weight", sharding=sharding.column_sharded),
        post_attention_layernorm=store.tensor(
            f"{prefix}.post_attention_layernorm.weight",
            sharding=sharding.replicated,
        ),
        post_feedforward_layernorm=store.tensor(
            f"{prefix}.post_feedforward_layernorm.weight",
            sharding=sharding.replicated,
        ),
        pre_feedforward_layernorm=store.tensor(f"{prefix}.pre_feedforward_layernorm.weight", sharding=sharding.replicated),
        q_norm=store.tensor(f"{attn}.q_norm.weight", sharding=sharding.replicated),
        q_proj=store.tensor(f"{attn}.q_proj.weight", sharding=sharding.row_sharded),
        up_proj=store.tensor(f"{mlp}.up_proj.weight", sharding=sharding.row_sharded),
        gate_proj=store.tensor(f"{mlp}.gate_proj.weight", sharding=sharding.row_sharded),
        down_proj=store.tensor(f"{mlp}.down_proj.weight", sharding=sharding.column_sharded),
        v_proj=store.tensor(f"{attn}.v_proj.weight", sharding=kv_sharding)
        if uses_v_proj(config, layer_idx)
        else None,
    )


def load_embedding(store: TensorStore, sharding: ShardingPlan) -> jax.Array:
    embedding_sharding = sharding.vocab_sharded if env_flag("SHARD_EMBEDDING", True) else sharding.replicated
    return store.tensor(EMBEDDING, sharding=embedding_sharding)


def load_resident_weights(
    store: TensorStore,
    config: Gemma4TextConfig,
    sharding: ShardingPlan,
    layer_limit: int,
) -> ResidentWeights:
    start = time.monotonic()
    embedding = load_embedding(store, sharding)
    norm = store.tensor(FINAL_NORM, sharding=sharding.replicated)
    print("resident weights: embedding and final norm loaded")

    layers = []
    for layer_idx in range(layer_limit):
        layer_start = time.monotonic()
        layers.append(load_layer(store, config, sharding, layer_idx))
        gc.collect()
        if (layer_idx + 1) % 5 == 0 or layer_idx == layer_limit - 1:
            elapsed = time.monotonic() - layer_start
            print(f"resident layer {layer_idx + 1:02d}/{layer_limit}: loaded in {elapsed:.1f}s")

    total = time.monotonic() - start
    print(f"resident weights: {layer_limit} layers loaded in {total:.1f}s")
    return ResidentWeights(embedding=embedding, layers=layers, norm=norm)


def rms_norm(hidden_states: jax.Array, weight: jax.Array | None, eps: float) -> jax.Array:
    states = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(states), axis=-1, keepdims=True)
    states = states * jax.lax.rsqrt(variance + eps)
    if weight is not None:
        states = states * weight.astype(jnp.float32)
    return states.astype(hidden_states.dtype)


def linear(hidden_states: jax.Array, weight: jax.Array) -> jax.Array:
    return jnp.matmul(hidden_states, weight.T).astype(BF16)


def gelu_pytorch_tanh(x: jax.Array) -> jax.Array:
    x32 = x.astype(jnp.float32)
    coeff = math.sqrt(2.0 / math.pi)
    return (0.5 * x32 * (1.0 + jnp.tanh(coeff * (x32 + 0.044715 * jnp.power(x32, 3))))).astype(BF16)


def rotate_half(x: jax.Array) -> jax.Array:
    left, right = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-right, left), axis=-1)


def rotary_embedding(config: Gemma4TextConfig, seq_len: int, layer_type: str, head_dim: int) -> tuple[jax.Array, jax.Array]:
    params = config.rope_parameters[layer_type]
    base = float(params["rope_theta"])
    rope_type = params.get("rope_type", "default")

    if rope_type == "proportional":
        proportion = float(params.get("partial_rotary_factor", 1.0))
        rope_angles = int(proportion * head_dim // 2)
        rotated = jnp.arange(0, 2 * rope_angles, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (base ** (rotated / head_dim))
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = jnp.concatenate((inv_freq, jnp.zeros((nope_angles,), dtype=jnp.float32)))
    else:
        positions = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (base ** (positions / head_dim))

    token_positions = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("s,d->sd", token_positions, inv_freq)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb).astype(BF16), jnp.sin(emb).astype(BF16)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    return (x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]).astype(BF16)


def repeat_kv(hidden_states: jax.Array, repeats: int) -> jax.Array:
    if repeats == 1:
        return hidden_states
    return jnp.repeat(hidden_states, repeats, axis=1)


def attention_mask(config: Gemma4TextConfig, seq_len: int, layer_type: str) -> jax.Array:
    query = jnp.arange(seq_len)[:, None]
    key = jnp.arange(seq_len)[None, :]
    mask = key <= query
    if layer_type == "sliding_attention":
        mask = mask & ((query - key) < config.sliding_window)
    return mask


def self_attention(
    hidden_states: jax.Array,
    weights: LayerWeights,
    config: Gemma4TextConfig,
    layer_idx: int,
    rope_cache: dict[str, tuple[jax.Array, jax.Array]],
) -> jax.Array:
    seq_len = hidden_states.shape[0]
    layer_type = config.layer_types[layer_idx]
    is_sliding = layer_type == "sliding_attention"
    head_dim = config.head_dim if is_sliding else config.global_head_dim
    kv_heads = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads
    kv_groups = config.num_attention_heads // kv_heads
    cos, sin = rope_cache[layer_type]

    query = linear(hidden_states, weights.q_proj).reshape(seq_len, config.num_attention_heads, head_dim)
    query = rms_norm(query, weights.q_norm, config.rms_norm_eps)
    query = apply_rope(query, cos, sin)

    key = linear(hidden_states, weights.k_proj).reshape(seq_len, kv_heads, head_dim)
    value = key if weights.v_proj is None else linear(hidden_states, weights.v_proj).reshape(seq_len, kv_heads, head_dim)
    key = rms_norm(key, weights.k_norm, config.rms_norm_eps)
    key = apply_rope(key, cos, sin)
    value = rms_norm(value, None, config.rms_norm_eps)

    key = repeat_kv(key, kv_groups)
    value = repeat_kv(value, kv_groups)

    scores = jnp.einsum("qhd,khd->hqk", query.astype(jnp.float32), key.astype(jnp.float32))
    mask = attention_mask(config, seq_len, layer_type)
    scores = jnp.where(mask[None, :, :], scores, jnp.array(-1.0e30, dtype=jnp.float32))
    probs = jax.nn.softmax(scores, axis=-1).astype(BF16)
    output = jnp.einsum("hqk,khd->qhd", probs, value).reshape(seq_len, config.num_attention_heads * head_dim)
    return linear(output, weights.o_proj)


def mlp(hidden_states: jax.Array, weights: LayerWeights) -> jax.Array:
    gate = linear(hidden_states, weights.gate_proj)
    up = linear(hidden_states, weights.up_proj)
    return linear(gelu_pytorch_tanh(gate) * up, weights.down_proj)


def decoder_layer(
    hidden_states: jax.Array,
    weights: LayerWeights,
    config: Gemma4TextConfig,
    layer_idx: int,
    rope_cache: dict[str, tuple[jax.Array, jax.Array]],
) -> jax.Array:
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.input_layernorm, config.rms_norm_eps)
    hidden_states = self_attention(hidden_states, weights, config, layer_idx, rope_cache)
    hidden_states = rms_norm(hidden_states, weights.post_attention_layernorm, config.rms_norm_eps)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights.pre_feedforward_layernorm, config.rms_norm_eps)
    hidden_states = mlp(hidden_states, weights)
    hidden_states = rms_norm(hidden_states, weights.post_feedforward_layernorm, config.rms_norm_eps)
    hidden_states = residual + hidden_states

    return (hidden_states * weights.layer_scalar.reshape(())).astype(BF16)


def forward_last_logits(
    config: Gemma4TextConfig,
    sharding: ShardingPlan,
    resident: ResidentWeights,
    input_ids: list[int],
) -> jax.Array:
    layer_limit = configured_layer_limit(config)
    token_ids = jnp.asarray(np.array(input_ids, dtype=np.int32))
    embedding = resident.embedding
    hidden_states = jnp.take(embedding, token_ids, axis=0)
    hidden_states = (hidden_states * jnp.asarray(math.sqrt(config.hidden_size), dtype=BF16)).astype(BF16)
    hidden_states = place(hidden_states, sharding.replicated)

    seq_len = len(input_ids)
    rope_cache = {}
    for layer_type in set(config.layer_types[:layer_limit]):
        head_dim = config.head_dim if layer_type == "sliding_attention" else config.global_head_dim
        cos, sin = rotary_embedding(config, seq_len, layer_type, head_dim)
        rope_cache[layer_type] = place(cos, sharding.replicated), place(sin, sharding.replicated)

    for layer_idx in range(layer_limit):
        start = time.monotonic()
        weights = resident.layers[layer_idx]
        if layer_idx == 0 and env_flag("PRINT_SHARDING", True):
            print(
                "layer 0 sharding:",
                {
                    "q_proj": sharding_spec(weights.q_proj),
                    "k_proj": sharding_spec(weights.k_proj),
                    "o_proj": sharding_spec(weights.o_proj),
                    "gate_proj": sharding_spec(weights.gate_proj),
                    "down_proj": sharding_spec(weights.down_proj),
                },
        )
        hidden_states = decoder_layer(hidden_states, weights, config, layer_idx, rope_cache)
        hidden_states.block_until_ready()
        if (layer_idx + 1) % 5 == 0 or layer_idx == layer_limit - 1:
            elapsed = time.monotonic() - start
            print(f"layer {layer_idx + 1:02d}/{layer_limit}: evaluated in {elapsed:.1f}s")

    norm = resident.norm
    hidden_states = rms_norm(hidden_states, norm, config.rms_norm_eps)
    logits = jnp.matmul(hidden_states[-1].astype(jnp.float32), embedding.T.astype(jnp.float32))
    if config.final_logit_softcapping is not None:
        softcap = config.final_logit_softcapping
        logits = jnp.tanh(logits / softcap) * softcap
    logits.block_until_ready()
    return logits


def chat_prompt(user_prompt: str) -> str:
    return f"<bos><|turn>user\n{user_prompt.strip()}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def generate(
    store: TensorStore,
    config: Gemma4TextConfig,
    sharding: ShardingPlan,
    tokenizer: tokenizers.Tokenizer,
    prompt: str,
    eos_ids: set[int],
    max_new_tokens: int,
) -> tuple[list[int], list[int]]:
    input_ids = tokenizer.encode(prompt).ids
    prompt_len = len(input_ids)
    print(f"prompt tokens: {prompt_len}; max_new_tokens: {max_new_tokens}")
    layer_limit = configured_layer_limit(config)
    print("load strategy: resident")
    resident = load_resident_weights(store, config, sharding, layer_limit)

    for step in range(max_new_tokens):
        start = time.monotonic()
        logits = forward_last_logits(config, sharding, resident, input_ids)
        next_id = int(np.asarray(jnp.argmax(logits)))
        del logits
        gc.collect()
        input_ids.append(next_id)
        elapsed = time.monotonic() - start
        print(f"token {step + 1}: id={next_id}; elapsed={elapsed:.1f}s")
        if next_id in eos_ids:
            break

    return input_ids[:prompt_len], input_ids[prompt_len:]


def print_runtime(root: pathlib.Path, store: TensorStore, config: Gemma4TextConfig, sharding: ShardingPlan) -> None:
    print("model_dir:", root)
    print("safetensors:", importlib.metadata.version("safetensors"))
    print("tokenizers:", importlib.metadata.version("tokenizers"))
    print("jax:", importlib.metadata.version("jax"))
    print("jax backend:", jax.default_backend())
    print("jax devices:", jax.devices())
    print("shards:", store.summary())
    print(
        "jax sharding:",
        {
            "enabled": sharding.enabled,
            "axis": MODEL_AXIS if sharding.enabled else None,
            "shards": sharding.shard_count,
            "row": str(getattr(sharding.row_sharded, "spec", None)),
            "column": str(getattr(sharding.column_sharded, "spec", None)),
            "vocab": str(getattr(sharding.vocab_sharded, "spec", None)),
        },
    )
    print(
        "text config:",
        {
            "layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "heads": config.num_attention_heads,
            "sliding_window": config.sliding_window,
            "vocab_size": config.vocab_size,
        },
    )


# %%
model_dir = find_model_dir()
raw_config, text_config = load_config(model_dir)
tensor_store = TensorStore(model_dir)
sharding_plan = build_sharding_plan()
print_runtime(model_dir, tensor_store, text_config, sharding_plan)
validate_index(tensor_store, text_config)

# %%
if env_flag("VALIDATE_ONLY", False):
    print("VALIDATE_ONLY=1: index/config validation complete; generation skipped")
else:
    tokenizer = tokenizers.Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    user_prompt = os.environ.get("PROMPT", "Explain quantum entanglement in one sentence.")
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1"))
    eos_token_ids = load_eos_ids(model_dir, raw_config)
    full_prompt = chat_prompt(user_prompt)
    if sharding_plan.mesh is None:
        prompt_ids, generated_ids = generate(
            tensor_store,
            text_config,
            sharding_plan,
            tokenizer,
            full_prompt,
            eos_token_ids,
            max_new_tokens,
        )
    else:
        with sharding_plan.mesh:
            prompt_ids, generated_ids = generate(
                tensor_store,
                text_config,
                sharding_plan,
                tokenizer,
                full_prompt,
                eos_token_ids,
                max_new_tokens,
            )
    print("generated token ids:", generated_ids)
    print("generated text:")
    print(tokenizer.decode(generated_ids, skip_special_tokens=False))
