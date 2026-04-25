"""kagglet CLI: push a directory described by `notebook.toml`.

Each example/project directory contains:
  * `notebook.toml` — slug, title, optional fields (sources, internet, ...)
  * one or more percent-format `.py` files (auto-discovered if `sources` is omitted)

Subcommands:
  * `push <dir> [--poll]` — build + push the notebook; optionally poll until done
  * `show <dir>` — print the derived `kernel-metadata.json` without pushing
"""

import argparse
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

from kagglet.notebook import KaggleNotebook
from kagglet.api.client import kaggle_api


class NotebookSettings(BaseSettings):
    """Schema for `<dir>/notebook.toml`.

    Required: `slug`, `title`. Other fields default to `KaggleNotebook` defaults;
    `sources` defaults to all `*.py` files in the directory (sorted) when omitted.
    """

    model_config = SettingsConfigDict(extra="forbid")

    slug: str
    title: str
    sources: list[str] = Field(default_factory=list)
    internet: bool = True
    competition: str = ""
    accelerator: str = ""

    @classmethod
    def load(cls, dir: Path) -> "NotebookSettings":
        toml_path = dir / "notebook.toml"
        if not toml_path.exists():
            raise FileNotFoundError(f"missing {toml_path}")
        source = TomlConfigSettingsSource(cls, toml_file=toml_path)
        return cls.model_validate(source())

    def to_notebook(self, dir: Path) -> KaggleNotebook:
        sources = self.sources or sorted(p.name for p in dir.glob("*.py"))
        if not sources:
            raise ValueError(f"no .py sources found in {dir}")
        slug = self.slug if "/" in self.slug else f"{kaggle_api().config_values['username']}/{self.slug}"
        return KaggleNotebook(
            slug=slug,
            title=self.title,
            sources=sources,
            sources_dir=dir,
            internet=self.internet,
            competition=self.competition,
            accelerator=self.accelerator,
        )


def _load(dir_arg: str) -> KaggleNotebook:
    dir = Path(dir_arg).resolve()
    return NotebookSettings.load(dir).to_notebook(dir)


def push_command(args):
    nb = _load(args.dir)
    nb.push()
    if args.poll:
        nb.poll()


def show_command(args):
    nb = _load(args.dir)
    print(nb.metadata.to_json())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kagglet")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="push notebook from <dir>/notebook.toml")
    p_push.add_argument("dir")
    p_push.add_argument("--poll", action="store_true", help="block until kernel finishes + print logs")
    p_push.set_defaults(func=push_command)

    p_show = sub.add_parser("show", help="print derived kernel-metadata.json without pushing")
    p_show.add_argument("dir")
    p_show.set_defaults(func=show_command)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
