# hello

Smallest possible kagglet usage.

```
hello/
├── notebook.toml   # slug, title, ...
├── bootstrap.py    # cell 1 (markdown header + env probe)
└── main.py         # cell 2-3 (hello + tiny computation)
```

`notebook.toml` is loaded by `pydantic-settings`; `*.py` files are auto-discovered
in alphabetical order (override with `sources = [...]` in the TOML).

## Run

```bash
pixi run kagglet push examples/hello --poll
```

`--poll` blocks until the kernel finishes and prints the downloaded `.log` files.

The `slug` in `notebook.toml` is just `"kagglet-hello"` — the CLI prepends your
kaggle username (read from `$KAGGLE_USERNAME` or `~/.kaggle/kaggle.json`).
Use a fully qualified slug like `"alice/kagglet-hello"` only when sharing.

## Inspect without pushing

```bash
pixi run kagglet show examples/hello
```

Prints the derived `kernel-metadata.json` so you can sanity-check before uploading.
