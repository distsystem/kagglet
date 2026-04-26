# gemma4-keras-tpu

Gemma 4 31B text inference on Kaggle TPU v5e-8 using resident JAX tensor
sharding.

This example intentionally does not use the smaller A2B/A4B presets and does
not call `keras_hub.models.Gemma4CausalLM.from_preset()`. Kaggle's notebook VM
has limited host memory, so the notebook attaches the Transformers
`google/gemma-4/transformers/gemma-4-31b-it/1` model source and reads
`model.safetensors.index.json` tensor by tensor. Each tensor is immediately
placed on a one-dimensional JAX `model` mesh with `NamedSharding`; the full
sharded model stays resident in TPU HBM before generation starts.

Runtime shape:

```text
┌────────────────────────────────────────────┐
│ startup                                    │
│   read safetensors tensor by tensor        │
│   place each tensor with NamedSharding     │
│   keep all 60 layers resident on TPU       │
└─────┬──────────────────────────────────────┘
      ▼
┌────────────────────────────────────────────┐
│ generation                                 │
│   token ids -> sharded tied embedding      │
│   run resident layer 0                     │
│   run resident layer 1                     │
│   ...                                      │
│   run resident layer 59                    │
│   final norm + tied embedding logits       │
└────────────────────────────────────────────┘
```

The sharding layout is intentionally simple:

```text
q/gate/up       : P("model", None)
k/v sliding     : P("model", None)
k/v full        : replicated, because 4 global KV heads cannot split over 8 chips
o/down          : P(None, "model")
embedding       : P("model", None)
norm/scalars    : replicated
```

The default generation is one greedy token (`MAX_NEW_TOKENS=1`) so the example
validates the full 31B route. Increase `MAX_NEW_TOKENS` after the first run
finishes; resident weights make extra tokens avoid rereading all 60 layers.

## Run

```bash
pixi run kagglet push examples/gemma4-keras-tpu --poll
```

Useful notebook environment overrides:

```bash
VALIDATE_ONLY=1     # validate config/index and skip generation
LAYER_LIMIT=2       # debug the first N layers; output is not meaningful
MAX_NEW_TOKENS=2    # generate more greedy tokens after the full route works
PROMPT="..."        # replace the default text prompt
SHARD_WEIGHTS=0     # disable JAX NamedSharding for comparison/debugging
SHARD_EMBEDDING=0   # keep tied embedding replicated while sharding layers
PRINT_SHARDING=0    # silence the first-layer sharding summary
```

## Inspect

```bash
pixi run kagglet show examples/gemma4-keras-tpu
```
