from resolv_pipelines.pipelines.datasets import ImportArchiveDatasetPipeline

from scripts.utilities.constants import Paths


if __name__ == '__main__':
    ImportArchiveDatasetPipeline(
        output_path=Paths.DATASETS_DIR,
        source_dataset_names=["lakh-midi-v1"],
        source_dataset_modes=["clean"],
        force_overwrite=True,
        allow_invalid_checksum=False,
        logging_level="INFO"
    ).run_pipeline()
