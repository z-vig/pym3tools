from pathlib import Path

from pydantic import BaseModel

from pym3tools.m3catalog import DataIDString


class RegionalPipelineConfig(BaseModel):
    name: str
    gcps: Path
    slope: Path
    aspect: Path
    catalog: Path
    cache: Path


class RegionalPipelineBatch(BaseModel):
    batch_name: str
    config_list: list[RegionalPipelineConfig]


sample_ids: list[DataIDString] = [
    "M3T20090418T020644",
    "M3T20090418T020848",
    "M3G20090208T160125",
    "M3G20090208T175211",
    "M3G20090208T194335",
]


def create_batch_file(
    gcps_dir: Path | str,
    region_dir: Path | str,
    slope_path: Path | str,
    aspect_path: Path | str,
):
    gcps_dir = Path(gcps_dir)
    gcps_dict = {i.stem: i for i in gcps_dir.iterdir() if i.suffix == ".gcps"}

    region_dir = Path(region_dir)
    catalog_dict = {i: region_dir / i for i in sample_ids}
    cache_dict = {
        k: v / "pipeline_cache.hdf5" for k, v in catalog_dict.items()
    }

    batch = RegionalPipelineBatch(
        batch_name="Gruithuisen_Region",
        config_list=[
            RegionalPipelineConfig(
                name=i,
                gcps=gcps_dict[i],
                slope=Path(slope_path),
                aspect=Path(aspect_path),
                cache=cache_dict[i],
                catalog=catalog_dict[i],
            )
            for i in sample_ids
        ],
    )

    json = batch.model_dump_json(indent=2)
    with open(Path(__file__).parent / "batch_config.json", "w") as f:
        f.write(json)


if __name__ == "__main__":
    create_batch_file(
        "D:/moon_data/m3/Gruithuisen_Region_OLD/gcps_files/",
        "D:/moon_data/m3/Gruithuisen_Region/",
        "D:/moon_data/derived_products/gruithuisen_region/Gruit_Slope.json",
        "D:/moon_data/derived_products/gruithuisen_region/Gruit_Aspect.json",
    )
