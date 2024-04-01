import resolv_pipelines.pipelines as resolv_pipelines
from resolv_mir import NoteSequence
from resolv_pipelines.pipelines.datasets.utilities import DrawHistogramPipeline

from scripts.utilities.constants import Paths

if __name__ == '__main__':
    DrawHistogramPipeline(
        canonical_format=NoteSequence,
        allowed_attributes_map=resolv_pipelines.SUPPORTED_NOTE_SEQ_ATTRIBUTES,
        bins=[20, 30, 40, 50, 60, 70, 80],
        source_dataset_names=["lakh-midi-v1"],
        source_dataset_modes=["clean"],
        source_dataset_file_types=["midi"],
        input_path=Paths.GENERATED_DATASETS_DIR / "4bars_melodies_distinct",
        input_path_prefix="attributes",
        force_overwrite=True,
        logging_level="INFO",
        pipeline_options={
            "runner": "DirectRunner",
            "direct_running_mode": "multi_processing",
            "direct_num_workers": 8,
            "direct_runner_bundle_repeat": 0
        }
    ).run_pipeline()
