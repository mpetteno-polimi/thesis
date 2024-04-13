import resolv_pipelines.pipelines as resolv_pipelines
from resolv_mir import NoteSequence
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation
from resolv_pipelines.pipelines.datasets import RepresentationDatasetPipeline

from scripts.utilities.constants import Paths


def augmenters_config():
    return {
        "transposer": {
            "order": 1,
            "threshold": 0.7,
            "min_transpose_amount": -12,
            "max_transpose_amount": 12,
            "max_allowed_pitch": 108,
            "min_allowed_pitch": 21,
            "transpose_chords": False,
            "delete_notes": False,
            "in_place": True
        }
    }


if __name__ == '__main__':
    RepresentationDatasetPipeline(
        representation=PitchSequenceRepresentation(sequence_length=64, keep_attributes=True),
        canonical_format=NoteSequence,
        augmenters_config=augmenters_config(),
        allowed_augmenters_map=resolv_pipelines.SUPPORTED_NOTE_SEQ_AUGMENTERS,
        source_dataset_names=["lakh-midi-v1"],
        source_dataset_modes=["clean"],
        source_dataset_file_types=["midi"],
        input_path=Paths.GENERATED_DATASETS_DIR / "4bars_melodies_distinct",
        input_path_prefix="attributes",
        output_path=Paths.DATASETS_DIR,
        output_path_prefix="pitchseq",
        output_dataset_name="4bars_melodies_distinct",
        split_ratios={"train": 0.8, "validation": 0.15, "test": 0.05},
        force_overwrite=True,
        logging_level="INFO",
        debug=False,
        debug_file_pattern=".*",
        pipeline_options={
            "runner": "DirectRunner",
            "direct_running_mode": "multi_processing",
            "direct_num_workers": 6,
            "direct_runner_bundle_repeat": 0
        }
    ).run_pipeline()
