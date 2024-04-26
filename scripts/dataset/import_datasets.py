from resolv_pipelines.pipelines.datasets import ImportArchiveDatasetPipeline, CanonicalizeDatasetPipeline

from scripts.dataset.utilities import Paths


if __name__ == '__main__':
    ImportArchiveDatasetPipeline(
        output_path=Paths.DATASETS_DIR,
        source_dataset_names=["lakh-midi-v1"],
        source_dataset_modes=["full"],
        force_overwrite=True,
        allow_invalid_checksum=False,
        logging_level="INFO"
    ).run_pipeline()

    CanonicalizeDatasetPipeline(
        input_path=Paths.RAW_DATASETS_DIR,
        output_path=Paths.DATASETS_DIR,
        source_dataset_names=["lakh-midi-v1"],
        source_dataset_modes=["full"],
        source_dataset_file_types=["midi"],
        force_overwrite=True,
        logging_level="INFO",
        debug=False,
        debug_file_pattern=".*",
        pipeline_options={
            "runner": "DirectRunner",
            "direct_running_mode": "multi_processing",
            "direct_num_workers": 8
        }
    ).run_pipeline()
