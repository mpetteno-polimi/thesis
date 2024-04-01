from resolv_pipelines.pipelines.datasets import CanonicalizeDatasetPipeline

from scripts.utilities.constants import Paths


if __name__ == '__main__':
    CanonicalizeDatasetPipeline(
        input_path=Paths.RAW_DATASETS_DIR,
        output_path=Paths.DATASETS_DIR,
        source_dataset_names=["lakh-midi-v1"],
        source_dataset_modes=["clean"],
        source_dataset_file_types=["midi"],
        force_overwrite=True,
        logging_level="INFO",
        debug=False,
        debug_file_pattern=".*",
        pipeline_options={
            "runner": "DirectRunner",
            "direct_running_mode": "multi_processing",
            "direct_num_workers": 8,
            "direct_runner_bundle_repeat": 0
        }
    ).run_pipeline()
