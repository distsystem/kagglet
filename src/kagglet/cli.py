"""kagglet CLI: push a directory described by `notebook.yaml`.

Each example/project directory contains:
  * `notebook.yaml` — kernel metadata, plus `sources` (explicit list or globs
    like `["*.py"]`) and optional project fields
  * one or more percent-format `.py` files referenced by `sources`

Subcommands:
  * `push <dir> [--poll]` - build + push the notebook; optionally poll until done
  * `show <dir>` - print the derived `kernel-metadata.json` without pushing
  * `whoami` - print the authenticated Kaggle account
"""

from typing import ClassVar
from pathlib import Path

import pydantic
from pydantic_settings import (
    CliApp,
    BaseSettings,
    CliSubCommand,
    CliPositionalArg,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from kagglet.api import kaggle
from kagglet.notebook import NotebookProject


class NotebookProjectSettings(NotebookProject, BaseSettings):
    """Schema for `<dir>/notebook.yaml`.

    Required: `kernel.name` and `sources` (list of percent-format `.py` paths
    or globs like `["*.py"]`). `kernel.owner` defaults to the active Kaggle
    account, and `kernel.title` defaults from `kernel.name`.
    """

    model_config = SettingsConfigDict(extra="forbid")

    _sources_dir: ClassVar[Path]
    _yaml_path: ClassVar[Path]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=cls._yaml_path)
        return init_settings, env_settings, dotenv_settings, yaml_settings, file_secret_settings

    @classmethod
    def load(cls, dir: Path) -> "NotebookProjectSettings":
        yaml_path = dir / "notebook.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"missing {yaml_path}")

        class BoundNotebookProjectSettings(cls):
            _sources_dir = dir
            _yaml_path = yaml_path

        return BoundNotebookProjectSettings()

    @pydantic.model_validator(mode="after")
    def apply_cli_defaults(self):
        if not self.kernel.owner:
            self.kernel.owner = kaggle().username
        if self.sources_dir is None:
            self.sources_dir = self._sources_dir
        if not self.sources:
            raise ValueError(
                f"{self._yaml_path}: 'sources' is required; "
                "use ['*.py'] to include every .py file in the directory"
            )
        return self


def _load(dir_arg: str) -> NotebookProject:
    dir = Path(dir_arg).resolve()
    return NotebookProjectSettings.load(dir)


def push_command(args):
    project = _load(args.dir)
    project.push()
    if args.poll:
        project.poll()


def show_command(args):
    import json

    project = _load(args.dir)
    request = project.save_request()
    print(json.dumps(request.to_field_map(), indent=2, default=str))


def whoami_command(_args):
    api = kaggle()
    print(f"username: {api.username}")
    print(f"auth_method: {api.auth_method}")


class PushCommand(pydantic.BaseModel):
    dir: CliPositionalArg[str]
    poll: bool = False

    def cli_cmd(self) -> None:
        push_command(self)


class ShowCommand(pydantic.BaseModel):
    dir: CliPositionalArg[str]

    def cli_cmd(self) -> None:
        show_command(self)


class WhoamiCommand(pydantic.BaseModel):
    def cli_cmd(self) -> None:
        whoami_command(self)


class KaggletCli(BaseSettings):
    model_config = SettingsConfigDict(
        cli_prog_name="kagglet",
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_enforce_required=True,
    )

    push: CliSubCommand[PushCommand]
    show: CliSubCommand[ShowCommand]
    whoami: CliSubCommand[WhoamiCommand]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main(argv: list[str] | None = None) -> None:
    CliApp.run(KaggletCli, cli_args=argv)


if __name__ == "__main__":
    main()
