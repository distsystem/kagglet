import pytest

import kagglet
import kagglet.cli
import kagglet.asset


def test_split_slug_handles_owner_and_local_name():
    assert kagglet.asset.split_slug("alice/notebook") == ("alice", "notebook")
    assert kagglet.asset.split_slug("notebook") == ("", "notebook")


def test_accelerator_maps_to_kernel_metadata_fields():
    kernel = kagglet.KaggleKernel(
        owner="alice",
        name="nb",
        accelerator=kagglet.Accelerator.TPU_V5E8,
    )

    metadata = kernel.metadata

    assert metadata.enable_gpu == "false"
    assert metadata.enable_tpu == "true"
    assert metadata.machine_shape == "TpuV5E8"
    assert metadata.title == "nb"


def test_kernel_title_defaults_from_name():
    kernel = kagglet.KaggleKernel(owner="alice", name="my_kernel-name")

    assert kernel.metadata.title == "my kernel name"


def test_kernel_accepts_slug_constructor():
    kernel = kagglet.KaggleKernel(slug="alice/nb")

    assert kernel.owner == "alice"
    assert kernel.name == "nb"
    assert kernel.slug == "alice/nb"


def test_notebook_project_collects_dataset_deps_in_metadata():
    dataset = kagglet.KaggleDataset(owner="alice", name="images")
    project = kagglet.NotebookProject(
        kernel=kagglet.KaggleKernel(owner="alice", name="nb"),
        deps=[dataset],
    )

    assert project.metadata.dataset_sources == ["alice/images"]


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
    )
    (tmp_path / "main.py").write_text("# %%\nprint('ok')\n")

    settings = kagglet.cli.NotebookProjectSettings.load(tmp_path)

    assert isinstance(settings, kagglet.NotebookProject)
    assert settings.kernel.owner == "alice"
    assert settings.kernel.name == "nb"
    assert settings.sources == ["main.py"]
    assert settings.sources_dir == tmp_path
    assert settings.kernel.accelerator is kagglet.Accelerator.TPU_V5E8


def test_notebook_settings_defaults_owner_from_kaggle_account(tmp_path, monkeypatch):
    class FakeApi:
        config_values = {"username": "alice"}

    monkeypatch.setattr(kagglet.cli, "kaggle_api", lambda: FakeApi())
    (tmp_path / "notebook.yaml").write_text("kernel:\n  name: nb\n")
    (tmp_path / "main.py").write_text("# %%\nprint('ok')\n")

    settings = kagglet.cli.NotebookProjectSettings.load(tmp_path)

    assert settings.kernel.owner == "alice"
    assert settings.kernel.name == "nb"
