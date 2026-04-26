# hello

Smallest possible kagglet usage.

```
hello/
├── notebook.yaml   # kernel name, optional owner/title, ...
├── bootstrap.py    # cell 1 (markdown header + env probe)
└── main.py         # cell 2-3 (hello + tiny computation)
```

`notebook.yaml` is loaded by `pydantic-settings`; `kernel` describes the Kaggle
kernel, while `*.py` files are auto-discovered in alphabetical order (override
with `sources: [...]` in YAML).

## Run

```bash
pixi run kagglet push examples/hello --poll
```

`--poll` blocks until the kernel finishes and prints the downloaded `.log` files.

The `kernel.name` in `notebook.yaml` is `"kagglet-hello"`; the CLI fills
`kernel.owner` from your active Kaggle account. Set `kernel.owner` only when
pushing to a different owner.

## Inspect without pushing

```bash
pixi run kagglet show examples/hello
```

Prints the derived `kernel-metadata.json` so you can sanity-check before uploading.
