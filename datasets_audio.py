"""Per-dataset source walk + label rules (AudioMNIST, Speech Commands v0.02).

Stdlib-only on purpose: `prepare_audio_data.py` and `eval/convergence_curve.py`
both import it, and the unit tests exercise it without torch/librosa.

Nothing here belongs in `audio/contract.py` — that module is the byte-identical
mel contract copied into dlg-sonic and stays dataset-agnostic
(`GAN_TRAINING_SPEC.md` §6). Per `SPEECH_COMMANDS_SPEC.md` §0 the *only*
per-dataset things are: this walk, the speaker/content parse, `m_hi`, the
speaker lists and `split_seed`. The mel config, canvas, offset and
`pad_value_normalized` are identical across datasets.
"""

import re
from pathlib import Path

AUDIOMNIST = "audiomnist"
SPEECH_COMMANDS = "speech_commands"
DATASETS = (AUDIOMNIST, SPEECH_COMMANDS)

# AudioMNIST clip naming: <digit>_<speaker_id>_<recording_id>.wav, one folder per speaker.
AUDIOMNIST_FILENAME_RE = re.compile(r"^(?P<digit>\d+)_(?P<speaker>[^_]+)_(?P<rec>\d+)\.wav$")

# Speech Commands v0.02: one folder per WORD (the folder name is the content
# label); the speaker is the 8-hex prefix of the filename, i.e. everything before
# the first underscore (SPEECH_COMMANDS_SPEC.md §1). Everything after `_nohash_`
# is an utterance index and carries no identity.
SC_FILENAME_RE = re.compile(r"^[0-9a-f]+_nohash_\d+\.wav$")
SC_BACKGROUND_DIR = "_background_noise_"  # minutes-long noise files — never 1 s speech (§1, §9)
SC_NUM_WORDS = 35
SC_NUM_SPEAKERS = 2618
SC_NUM_CLIPS = 105829
# The spoken digits among the 35 words — the sub-manifold that makes an SC prior
# usable OOD against AudioMNIST (§5 "optional OOD add"). Order = digit value.
SC_DIGIT_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
)

CONTENT_LABEL_TYPE = {AUDIOMNIST: "digit", SPEECH_COMMANDS: "word"}


def content_label_type(dataset):
    return CONTENT_LABEL_TYPE.get(dataset, "content")


def digit_words(dataset):
    """Content classes that name a spoken digit, for the OOD digit-coverage read.

    AudioMNIST is digits end to end, so the digit subset is the whole class set
    and a separate read would be redundant — returns () there.
    """
    return SC_DIGIT_WORDS if dataset == SPEECH_COMMANDS else ()


# --------------------------------------------------------------------------- #
# Source walks — both return (clips, content):
#   clips   {speaker_id: [wav Path, ...]}       drives the id-disjoint split
#   content {wav Path: content_label or None}   digit / word, carried to labels
# --------------------------------------------------------------------------- #


def collect_clips(wav_root, dataset, log=print):
    if dataset == SPEECH_COMMANDS:
        return collect_clips_speech_commands(wav_root, log=log)
    return collect_clips_audiomnist(wav_root, log=log)


def collect_clips_audiomnist(wav_root, log=print):
    """Speaker = parent dir name, cross-checked against the filename convention."""
    clips, content = {}, {}
    for path in sorted(Path(wav_root).rglob("*.wav")):
        speaker = path.parent.name
        m = AUDIOMNIST_FILENAME_RE.match(path.name)
        if m and m.group("speaker") != speaker:
            log(
                f"WARNING: {path.name}: filename speaker '{m.group('speaker')}' "
                f"!= parent dir '{speaker}' — using parent dir"
            )
        clips.setdefault(speaker, []).append(path)
        content[path] = m.group("digit") if m else None
    return clips, content


def collect_clips_speech_commands(wav_root, log=print):
    """Walk the raw GSC v0.02 tarball layout (SPEECH_COMMANDS_SPEC.md §1, §2.0).

    One folder per word; `_background_noise_/` is skipped entirely. The speaker
    hash is the filename prefix before the first underscore — the parse is
    verified against the real listing here (every offender is logged), because
    the id-disjoint split is only meaningful if it is right.
    """
    root = Path(wav_root)
    if not root.is_dir():
        raise NotADirectoryError(f"{wav_root} is not a directory")

    word_dirs, skipped = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name == SC_BACKGROUND_DIR:
            skipped.append(d.name)
            continue
        word_dirs.append(d)
    if skipped:
        log(f"speech_commands: skipping {skipped} (long noise files, not 1 s speech)")
    if not word_dirs:
        raise FileNotFoundError(
            f"{wav_root} contains no word folders — point --dataset speech_commands at the "
            "directory the v0.02 tarball was extracted into (one folder per word)"
        )

    clips, content, malformed = {}, {}, []
    for word_dir in word_dirs:
        word = word_dir.name
        for path in sorted(word_dir.glob("*.wav")):
            if not SC_FILENAME_RE.match(path.name):
                malformed.append(f"{word}/{path.name}")
            speaker = path.name.split("_")[0]
            clips.setdefault(speaker, []).append(path)
            content[path] = word

    if not clips:
        raise FileNotFoundError(f"{wav_root}: word folders contain no .wav files")
    if malformed:
        log(
            f"WARNING: {len(malformed)} filename(s) do not match "
            f"{SC_FILENAME_RE.pattern} — speaker ids for these came from the "
            f"before-first-underscore parse anyway; first few: {malformed[:10]}"
        )
    if len(word_dirs) != SC_NUM_WORDS:
        log(
            f"WARNING: found {len(word_dirs)} word folders, expected {SC_NUM_WORDS} "
            "for GSC v0.02 — check the extraction"
        )
    log(
        f"speech_commands: {len(word_dirs)} words, {sum(len(v) for v in clips.values())} clips, "
        f"{len(clips)} speakers"
    )
    return clips, content


# --------------------------------------------------------------------------- #
# .npy stem naming
# --------------------------------------------------------------------------- #


def clip_stem(dataset, speaker, path):
    """Output `.npy` stem for one source clip. The speaker is ALWAYS the first
    '__'-token (eval/convergence_curve.py relies on that).

    Speech Commands needs the word in the stem: the same speaker says different
    words with the same `_nohash_<n>` index, so `<speaker>__<wav stem>` alone
    collides across word folders and would silently overwrite `.npy` files.
    AudioMNIST keeps its original `<speaker>__<wav stem>` naming unchanged.
    """
    if dataset == SPEECH_COMMANDS:
        return f"{speaker}__{path.parent.name}_{path.stem}"
    return f"{speaker}__{path.stem}"


def speaker_from_stem(stem):
    """Speaker id of a prepared `.npy` stem (`<speaker>__<...>[__aug{k}]`)."""
    parts = stem.split("__")
    if len(parts) < 2 or not parts[0]:
        raise ValueError(
            f"unrecognized .npy stem {stem!r}: expected '<speaker>__<orig_stem>[__aug{{k}}]' "
            "(prepare_audio_data.py naming)"
        )
    return parts[0]


def content_from_stem(stem, dataset):
    """Fallback content label parsed from the stem, for `.npy` dirs prepared
    before `clip_labels.json` existed. Prefer the labels file when present."""
    parts = stem.split("__")
    if len(parts) < 2:
        raise ValueError(
            f"unrecognized .npy stem {stem!r}: expected '<speaker>__<orig_stem>[__aug{{k}}]' "
            "(prepare_audio_data.py naming) or a labels file"
        )
    orig = parts[1]
    if dataset == AUDIOMNIST:
        return orig.split("_")[0]                        # <digit>_<speaker>_<rec>
    if dataset == SPEECH_COMMANDS:
        head = orig.rsplit("_nohash_", 1)[0]             # <word>_<speaker hash>
        word = head.rsplit("_", 1)[0]
        if not word:
            raise ValueError(f"cannot parse a word label out of .npy stem {stem!r}")
        return word                                      # clip_stem() = <spk>__<word>_<wav stem>
    raise NotImplementedError(
        f"no content-label rule for dataset {dataset!r}: re-run prepare_audio_data.py so it "
        "emits clip_labels.json, or add a rule here"
    )
