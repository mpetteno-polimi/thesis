import resolv_pipelines.pipelines as resolv_pipelines
from resolv_mir import note_sequence, NoteSequence
from resolv_pipelines.pipelines.datasets import ProcessDatasetPipeline

from scripts.utilities.constants import Paths


def processors_config():
    return {
        "time_change_splitter": {
            "order": 1,
            "skip_splits_inside_notes": False,
            "keep_time_signatures": [
                "4/4"
            ]
        },
        "quantizer": {
            "order": 2,
            "steps_per_quarter": 4
        },
        "melody_extractor": {
            "order": 3,
            "filter_drums": True,
            "gap_bars": 1,
            "ignore_polyphonic_notes": True,
            "min_pitch": 21,
            "max_pitch": 108,
            "min_bars": 4,
            "min_unique_pitches": 3,
            "search_start_step": 0,
            "valid_programs": note_sequence.constants.MEL_PROGRAMS
        },
        "slicer": {
            "order": 4,
            "allow_cropped_slices": False,
            "hop_size_bars": 1,
            "skip_splits_inside_notes": False,
            "slice_size_bars": 4,
            "start_time": 0,
            "keep_shorter_slices": False
        }
    }


if __name__ == '__main__':
    ProcessDatasetPipeline(
        canonical_format=NoteSequence,
        allowed_processors_map=resolv_pipelines.SUPPORTED_NOTE_SEQ_PROCESSORS,
        processors_config=processors_config(),
        input_path=Paths.CANONICAL_DATASETS_DIR,
        output_path=Paths.DATASETS_DIR,
        source_dataset_names=["lakh-midi-v1"],
        source_dataset_modes=["clean"],
        source_dataset_file_types=["midi"],
        output_dataset_name="4bars_melodies_distinct",
        distinct=True,
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
