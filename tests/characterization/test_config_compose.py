# SPDX-License-Identifier: Apache-2.0
"""Smoke test: every model-config family must compose to a top-level ``model``.

Guards the regression where a UTF-8 BOM on a config's first line defeats Hydra's
``# @package _global_`` directive, nesting the config under ``cfg.configs`` instead
of ``cfg.model`` (which silently breaks every builder for that family).
"""
import pytest
import sam  # noqa: F401 -- importing sam initializes the hydra config module
from hydra import compose


@pytest.mark.parametrize(
    "cfg_name",
    [
        "configs/efficienttam/efficienttam_ti.yaml",
        "configs/sam2/sam2_hiera_t.yaml",
        "configs/sam2.1/sam2.1_hiera_t.yaml",
    ],
)
def test_config_composes_to_model(cfg_name):
    cfg = compose(config_name=cfg_name)
    assert "model" in cfg, (
        f"{cfg_name} did not compose to a top-level 'model' (BOM/@package regression?)"
    )
    assert cfg.model._target_
