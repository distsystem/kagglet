import pytest

import kagglet
import kagglet.cli
import kagglet.assets


def test_split_slug_handles_owner_and_local_name():
    assert kagglet.assets.split_slug("alice/notebook") == ("alice", "notebook")
    assert kagglet.assets.split_slug("notebook") == ("", "notebook")


def test_accelerator_maps_to_save_request_fields():
    kernel = kagglet.KaggleKernel(
        owner="alice",
        name="nb",
        accelerator=kagglet.Accelerator.TPU_V5E8,
    )

    request = kernel.save_request("nb-source")

    assert request.enable_gpu is False
    assert request.enable_tpu is True
    assert request.machine_shape == "TpuV5E8"
    assert request.new_title == "nb"
    assert request.text == "nb-source"
    assert request.slug == "alice/nb"


def test_kernel_title_defaults_from_name():
    kernel = kagglet.KaggleKernel(owner="alice", name="my_kernel-name")

    assert kernel.save_request("").new_title == "my kernel name"


def test_kernel_accepts_slug_constructor():
    kernel = kagglet.KaggleKernel(slug="alice/nb")

    assert kernel.owner == "alice"
    assert kernel.name == "nb"
    assert kernel.slug == "alice/nb"


def test_notebook_project_collects_dataset_deps_in_save_request():
    dataset = kagglet.KaggleDataset(owner="alice", name="images")
    project = kagglet.NotebookProject(
        kernel=kagglet.KaggleKernel(owner="alice", name="nb"),
        deps=[dataset],
    )

    assert project.save_request().dataset_data_sources == ["alice/images"]


def test_model_inherits_asset_identity_and_keeps_model_slug_shape():
    model = kagglet.KaggleModel(owner="alice", name="gemma", framework="jax", variation="default", version=2)

    assert model.ref == "alice/gemma"
    assert model.slug == "alice/gemma/jax/default"
    assert model.versioned_slug == "alice/gemma/jax/default/2"


def test_unknown_accelerator_is_rejected():
    with pytest.raises(ValueError):
        kagglet.KaggleKernel(owner="alice", name="nb", accelerator="TPU-v9")


def test_kaggle_notebook_compat_name_is_removed():
    assert not hasattr(kagglet, "KaggleNotebook")


def test_notebook_settings_loads_yaml_into_notebook(tmp_path):
    (tmp_path / "notebook.yaml").write_text(
        "kernel:\n"
        "  owner: alice\n"
        "  name: nb\n"
        "  accelerator: TpuV5E8\n"
        "sources: ['*.py']\n"
    )
    (tmp_path / "bootstrap.py").write_text("# %%\nprint('boot')\n")
    (tmp_path / "main.py").write_text("# %%\nprint('ok')\n")

    settings = kagglet.cli.NotebookProjectSettings.load(tmp_path)

    assert isinstance(settings, kagglet.NotebookProject)
    assert settings.kernel.owner == "alice"
    assert settings.kernel.name == "nb"
    assert settings.sources == ["*.py"]
    assert [p.name for p in settings._expand_sources()] == ["bootstrap.py", "main.py"]
    assert settings.sources_dir == tmp_path
    assert settings.kernel.accelerator is kagglet.Accelerator.TPU_V5E8


def test_notebook_settings_requires_explicit_sources(tmp_path):
    (tmp_path / "notebook.yaml").write_text("kernel:\n  owner: alice\n  name: nb\n")
    (tmp_path / "main.py").write_text("# %%\nprint('ok')\n")

    with pytest.raises(ValueError, match="'sources' is required"):
        kagglet.cli.NotebookProjectSettings.load(tmp_path)


def test_notebook_settings_glob_must_match(tmp_path):
    (tmp_path / "notebook.yaml").write_text(
        "kernel:\n  owner: alice\n  name: nb\n"
        "sources: ['*.py']\n"
    )

    settings = kagglet.cli.NotebookProjectSettings.load(tmp_path)
    with pytest.raises(ValueError, match="matched no files"):
        settings._expand_sources()


def test_notebook_settings_defaults_owner_from_kaggle_account(tmp_path, monkeypatch):
    class FakeKaggle:
        username = "alice"

    monkeypatch.setattr(kagglet.cli, "kaggle", lambda: FakeKaggle())
    (tmp_path / "notebook.yaml").write_text(
        "kernel:\n  name: nb\nsources: ['*.py']\n"
    )
    (tmp_path / "main.py").write_text("# %%\nprint('ok')\n")

    settings = kagglet.cli.NotebookProjectSettings.load(tmp_path)

    assert settings.kernel.owner == "alice"
    assert settings.kernel.name == "nb"
