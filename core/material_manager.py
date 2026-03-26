"""
Material store / manager (Milestone 2).

Keeps imported materials isolated and provides access for UI panels.
"""

from __future__ import annotations

from typing import Any

from core.importer import build_material_dataset, load_workbook
from core.schemas import MaterialDataset, TargetInstance


class MaterialManager:
    def __init__(self) -> None:
        self._materials: dict[str, MaterialDataset] = {}

    def clear(self) -> None:
        self._materials.clear()

    def load_workbook(
        self,
        path: str,
        *,
        column_map: dict[str, Any],
        config: dict[str, Any],
    ) -> list[str]:
        """Load all sheets as separate materials into the store."""
        workbook = load_workbook(path)

        for sheet_name, df in workbook.items():
            material_dataset = build_material_dataset(
                sheet_name,
                df,
                column_map=column_map,
                config=config,
            )
            self._materials[sheet_name] = material_dataset

        return self.list_materials()

    def list_materials(self) -> list[str]:
        return sorted(self._materials.keys())

    def get_material(self, material_name: str) -> MaterialDataset:
        return self._materials[material_name]

    def replace_material(self, material_name: str, dataset: MaterialDataset) -> None:
        self._materials[material_name] = dataset

    def get_target_instances(self, material_name: str) -> dict[str, TargetInstance]:
        return self.get_material(material_name).target_instances

    def get_target_instance_ids(self, material_name: str) -> list[str]:
        return sorted(self.get_target_instances(material_name).keys())

