import resolv_pipelines.pipelines as resolv_pipelines
from resolv_mir import NoteSequence
from resolv_pipelines.pipelines.datasets import ProcessDatasetPipeline

from scripts.utilities.constants import Paths


def attributes_config():
    return {
        "toussaint": {
            "order": 1,
            "bars": 4,
            "binary": True
        },
        "note_density": {
            "order": 6,
            "bars": 4,
            "binary": True
        },
        "pitch_range": {
            "order": 7,
            "num_midi_pitches": 88
        },
        "contour": {
            "order": 2,
            "num_midi_pitches": 88
        },
        "note_change_ratio": {
            "order": 5
        },
        "dynamic_range": {
            "order": 3
        },
        "longest_repetitive_section": {
            "order": 4,
            "min_repetitions": 4
        },
        "repetitive_section_ratio": {
            "order": 10,
            "min_repetitions": 4
        },
        "ratio_hold_note_steps": {
            "order": 8
        },
        "ratio_note_off_steps": {
            "order": 9
        },
        "unique_bigrams_ratio": {
            "order": 11,
            "num_midi_pitches": 88
        },
        "unique_notes_ratio": {
            "order": 12,
            "num_midi_pitches": 88
        },
        "unique_trigrams_ratio": {
            "order": 13,
            "num_midi_pitches": 88
        }
    }


if __name__ == '__main__':
    ProcessDatasetPipeline(
        canonical_format=NoteSequence,
        allowed_processors_map=resolv_pipelines.SUPPORTED_NOTE_SEQ_ATTRIBUTES,
        processors_config=attributes_config(),
        input_path=Paths.GENERATED_DATASETS_DIR / "4bars_melodies_distinct",
        output_path=Paths.DATASETS_DIR,
        source_dataset_names=["lakh-midi-v1"],
        source_dataset_modes=["clean"],
        source_dataset_file_types=["midi"],
        input_path_prefix="data",
        output_path_prefix="attributes",
        output_dataset_name="4bars_melodies_distinct",
        distinct=True,
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
