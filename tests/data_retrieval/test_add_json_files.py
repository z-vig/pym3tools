from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import TestCase
from unittest.mock import Mock, call, patch

from pym3tools.data_retrieval import add_json_files
from pym3tools.data_retrieval.data_directory import M3DataPaths
from pym3tools.types import DataLevel


class TestAddJsonFiles(TestCase):
    def test_add_json_to_level_writes_configured_images(self):
        configurations: list[tuple[DataLevel, list[tuple[str, str]]]] = [
            ("L0", [("raw", "RawData")]),
            (
                "L1B",
                [
                    ("rdn", "Radiance"),
                    ("obs", "Geometry"),
                    ("loc", "Location"),
                ],
            ),
            ("L2", [("rfl", "Reflectance"), ("sup", "Supplemental")]),
        ]

        for level, expected in configurations:
            with self.subTest(level=level):
                catalog = cast(
                    M3DataPaths,
                    SimpleNamespace(
                        **{
                            attr_name: SimpleNamespace(img=Path(f"{attr_name}.img"))
                            for attr_name, _data_name in expected
                        }
                    ),
                )
                with patch.object(add_json_files, "_write_json") as write_json:
                    add_json_files.add_json_to_level(catalog, level)

                self.assertEqual(
                    write_json.call_args_list,
                    [
                        call(Path(f"{attr_name}.img"), data_name)
                        for attr_name, data_name in expected
                    ],
                )

    def test_add_json_to_level_rejects_unknown_level(self):
        with self.assertRaisesRegex(ValueError, "Unknown data level: L3"):
            add_json_files.add_json_to_level(
                cast(M3DataPaths, SimpleNamespace()), cast(DataLevel, "L3")
            )

    def test_compatibility_wrappers_delegate_to_add_json_to_level(self):
        wrappers = [
            (add_json_files.add_json_to_l0, "L0"),
            (add_json_files.add_json_to_l1, "L1B"),
            (add_json_files.add_json_to_l2, "L2"),
        ]
        catalog = cast(M3DataPaths, object())

        for wrapper, level in wrappers:
            with self.subTest(level=level):
                with patch.object(
                    add_json_files, "add_json_to_level"
                ) as add_json_to_level:
                    wrapper(catalog)

                add_json_to_level.assert_called_once_with(catalog, level)

    def test_write_json_uses_cube_context_to_write_header(self):
        image_path = Path("image.img")
        raw_context = object()
        raw_cube = object()
        header_writer = Mock()

        with (
            patch.object(
                add_json_files,
                "cube_from_envi",
                return_value=(raw_context, raw_cube),
            ) as cube_from_envi,
            patch.object(
                add_json_files,
                "EnviHeaderWriter",
                return_value=header_writer,
            ) as header_writer_factory,
        ):
            add_json_files._write_json(image_path, "Radiance")

        cube_from_envi.assert_called_once_with(image_path, "Radiance")
        header_writer_factory.assert_called_once_with(raw_context)
        header_writer.to_file.assert_called_once_with(image_path)
