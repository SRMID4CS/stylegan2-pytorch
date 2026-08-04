"""Speech Commands data path (SPEECH_COMMANDS_SPEC.md).

CPU-only. The real GSC v0.02 tarball is never needed: the end-to-end tests build
a miniature corpus with the same *layout* (word folders, `<hash>_nohash_<n>.wav`,
16 kHz, sub-second clips, a `_background_noise_` folder) and run the real
prepare_audio_data.py over it. That exercises everything the spec locks —
the walk, the speaker parse, the id-disjoint split, the collision-safe stems,
the derived T, `m_hi` over train only, clip_labels.json — without 2.3 GB of audio.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import datasets_audio as da
import prepare_audio_data as prep
from audio.contract import MEL_CONFIG, clip_num_samples, expected_mel_frames

SC_SR = 16000                      # GSC v0.02 source rate (spec §1)
NUM_SAMPLES = clip_num_samples(MEL_CONFIG)

# A miniature corpus: 4 words (2 of them digits) x 5 speakers, so the seeded
# hold-out has something to bite on.
WORDS = ["yes", "no", "one", "two"]
SPEAKERS = ["0a7c2a8d", "1b8d3b9e", "2c9e4caf", "3daf5db0", "4eb06ec1"]


def write_wav(path, seconds, freq, sr=SC_SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(sr * seconds))
    t = np.arange(n, dtype=np.float32) / sr
    wav = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    torchaudio.save(str(path), torch.from_numpy(wav).unsqueeze(0), sr)


def make_sc_tree(root, words=WORDS, speakers=SPEAKERS, utterances=2):
    """GSC v0.02 layout: <word>/<hash>_nohash_<n>.wav, plus _background_noise_."""
    root = Path(root)
    for wi, word in enumerate(words):
        for si, spk in enumerate(speakers):
            for u in range(utterances):
                # Most real SC clips are SHORTER than 1 s — that is the case the
                # waveform-level zero padding has to handle (spec §2.2).
                write_wav(root / word / f"{spk}_nohash_{u}.wav",
                          seconds=0.6 + 0.1 * u, freq=200 + 40 * wi + 7 * si)
    # Minutes-long noise files that must never enter the mel .npy set (spec §1, §9).
    write_wav(root / da.SC_BACKGROUND_DIR / "doing_the_dishes.wav", seconds=3.0, freq=90)
    (root / "validation_list.txt").write_text("yes/0a7c2a8d_nohash_0.wav\n")
    (root / "testing_list.txt").write_text("no/1b8d3b9e_nohash_0.wav\n")
    (root / "LICENSE").write_text("test fixture\n")
    return root


# --------------------------------------------------------------------------- #
# Source walk + parsing
# --------------------------------------------------------------------------- #


def test_walk_groups_by_speaker_and_labels_by_word(tmp_path):
    root = make_sc_tree(tmp_path / "sc")
    clips, content = da.collect_clips(root, da.SPEECH_COMMANDS, log=lambda *_: None)

    assert sorted(clips) == sorted(SPEAKERS)
    # Every speaker says every word twice — the speaker spans word folders, which
    # is exactly why the split has to be by hash and not by folder.
    for spk in SPEAKERS:
        assert len(clips[spk]) == len(WORDS) * 2
        assert {p.parent.name for p in clips[spk]} == set(WORDS)
    assert set(content.values()) == set(WORDS)
    for path, word in content.items():
        assert path.parent.name == word


def test_walk_excludes_background_noise(tmp_path):
    root = make_sc_tree(tmp_path / "sc")
    clips, content = da.collect_clips(root, da.SPEECH_COMMANDS, log=lambda *_: None)

    assert da.SC_BACKGROUND_DIR not in {p.parent.name for p in content}
    assert "doing" not in clips and "_background_noise_" not in clips
    assert all(da.SC_BACKGROUND_DIR not in str(p) for v in clips.values() for p in v)


def test_walk_flags_malformed_filenames(tmp_path):
    root = make_sc_tree(tmp_path / "sc")
    write_wav(root / "yes" / "NOTAHASH_nohash_x.wav", seconds=0.5, freq=300)
    logged = []
    clips, _ = da.collect_clips(root, da.SPEECH_COMMANDS, log=logged.append)

    assert any("do not match" in m for m in logged)
    # Logged, not dropped: the before-first-underscore parse still applies, so a
    # naming surprise shows up loudly instead of silently shrinking the corpus.
    assert "NOTAHASH" in clips


def test_walk_rejects_a_non_sc_layout(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        da.collect_clips(empty, da.SPEECH_COMMANDS, log=lambda *_: None)


def test_speaker_is_the_prefix_before_the_first_underscore():
    for name in ("0a7c2a8d_nohash_0.wav", "ffffffff_nohash_17.wav"):
        assert da.SC_FILENAME_RE.match(name)
        assert name.split("_")[0] == name[:8]
    assert not da.SC_FILENAME_RE.match("0a7c2a8d_hash_0.wav")
    assert not da.SC_FILENAME_RE.match("0a7c2a8d_nohash_x.wav")


def test_clip_stem_is_collision_free_across_word_folders(tmp_path):
    """Same speaker, same utterance index, different word — the case that would
    silently overwrite .npy files if the word were left out of the stem."""
    yes = tmp_path / "yes" / "0a7c2a8d_nohash_0.wav"
    no = tmp_path / "no" / "0a7c2a8d_nohash_0.wav"
    s_yes = da.clip_stem(da.SPEECH_COMMANDS, "0a7c2a8d", yes)
    s_no = da.clip_stem(da.SPEECH_COMMANDS, "0a7c2a8d", no)

    assert s_yes != s_no
    assert da.speaker_from_stem(s_yes) == da.speaker_from_stem(s_no) == "0a7c2a8d"
    assert da.content_from_stem(s_yes, da.SPEECH_COMMANDS) == "yes"
    assert da.content_from_stem(s_no, da.SPEECH_COMMANDS) == "no"
    # ... and it survives the augmentation suffix.
    assert da.speaker_from_stem(f"{s_yes}__aug1") == "0a7c2a8d"
    assert da.content_from_stem(f"{s_yes}__aug1", da.SPEECH_COMMANDS) == "yes"


def test_audiomnist_stem_naming_is_unchanged(tmp_path):
    path = tmp_path / "01" / "0_01_3.wav"
    assert da.clip_stem(da.AUDIOMNIST, "01", path) == "01__0_01_3"
    assert da.content_from_stem("01__0_01_3", da.AUDIOMNIST) == "0"
    assert da.speaker_from_stem("01__0_01_3__aug2") == "01"


def test_digit_words_only_for_speech_commands():
    assert da.digit_words(da.SPEECH_COMMANDS) == da.SC_DIGIT_WORDS
    assert len(da.SC_DIGIT_WORDS) == 10
    assert da.digit_words(da.AUDIOMNIST) == ()


# --------------------------------------------------------------------------- #
# id-disjoint split (spec §3): hold out 200 by seeded shuffle of the hashes
# --------------------------------------------------------------------------- #


def test_split_holds_out_200_disjoint_and_seeded():
    hashes = [f"{i:08x}" for i in range(da.SC_NUM_SPEAKERS)]
    train, held = prep.split_speakers(hashes, 200, seed=0)

    assert len(held) == 200
    assert len(train) == da.SC_NUM_SPEAKERS - 200
    assert not set(train) & set(held)
    assert sorted(train + held) == sorted(hashes)

    again, _ = prep.split_speakers(hashes, 200, seed=0)
    assert again == train
    other, _ = prep.split_speakers(hashes, 200, seed=1)
    assert other != train


def test_split_is_speaker_level_not_sample_level(tmp_path):
    """Every utterance of a held-out speaker stays held out, across all words."""
    root = make_sc_tree(tmp_path / "sc")
    clips, _ = da.collect_clips(root, da.SPEECH_COMMANDS, log=lambda *_: None)
    train, held = prep.split_speakers(clips, 2, seed=0)

    held_paths = {p for spk in held for p in clips[spk]}
    train_paths = {p for spk in train for p in clips[spk]}
    assert held_paths and not held_paths & train_paths
    for spk in held:
        assert len({p.parent.name for p in clips[spk]}) == len(WORDS)


# --------------------------------------------------------------------------- #
# m_hi over ~97k clips: the bounded-memory percentile must equal np.percentile
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", [99.9, 99.0, 50.0, 100.0])
@pytest.mark.parametrize("n", [1, 7, 1000, 30011])
def test_streaming_percentile_matches_numpy(p, n):
    rng = np.random.default_rng(0)
    values = rng.normal(-6.0, 3.0, size=n).astype(np.float32)

    acc = prep.StreamingPercentile(n, p, chunk=997)
    for i in range(0, n, 137):
        acc.update(values[i:i + 137])

    np.testing.assert_allclose(acc.value(), np.percentile(values, p), rtol=0, atol=1e-5)


def test_streaming_percentile_refuses_a_short_stream():
    acc = prep.StreamingPercentile(100, 99.9)
    acc.update(np.zeros(50, dtype=np.float32))
    with pytest.raises(RuntimeError):
        acc.value()


def test_mel_store_backends_agree(tmp_path):
    mels = [torch.from_numpy(np.random.default_rng(i).normal(size=(4, 5)).astype(np.float32))
            for i in range(6)]
    out = {}
    for backend in ("memory", "disk"):
        store = prep.MelStore(len(mels), (4, 5), 99.9, backend, tmp_path / f".tmp_{backend}")
        for i, m in enumerate(mels):
            store.append(f"s{i}", m)
        out[backend] = (
            store.percentile(99.9),
            [(stem, t.clone()) for stem, t in store],
        )
        store.close()
        assert not (tmp_path / f".tmp_{backend}").exists()

    np.testing.assert_allclose(out["memory"][0], out["disk"][0], rtol=0, atol=1e-5)
    for (sm, tm), (sd, td) in zip(out["memory"][1], out["disk"][1]):
        assert sm == sd and torch.equal(tm, td)


def test_choose_mel_backend_thresholds():
    # AudioMNIST scale stays in RAM; Speech Commands scale spills to disk.
    assert prep.choose_mel_backend("auto", 25_000, (80, 86), 2.0) == "memory"
    assert prep.choose_mel_backend("auto", 97_000, (80, 86), 2.0) == "disk"
    assert prep.choose_mel_backend("memory", 97_000, (80, 86), 2.0) == "memory"
    assert prep.choose_mel_backend("disk", 10, (80, 86), 2.0) == "disk"


def test_expected_mel_frames_matches_the_extractor():
    """T is derived from the extractor's framing, never hardcoded — and because
    the contract and the 1.0 s clip are shared, it is the same for every dataset."""
    from audio.contract import extract_mel

    mel = extract_mel(torch.zeros(NUM_SAMPLES), MEL_CONFIG)
    assert tuple(mel.shape) == (MEL_CONFIG["n_mels"], expected_mel_frames(NUM_SAMPLES, MEL_CONFIG))


# --------------------------------------------------------------------------- #
# End-to-end prep over the miniature corpus
# --------------------------------------------------------------------------- #


def run_prep(wav_root, out_dir, extra=()):
    cmd = [
        sys.executable, str(REPO / "prepare_audio_data.py"),
        "--dataset", "speech_commands",
        "--out", str(out_dir),
        "--holdout", "2", "--split-seed", "0",
        *extra,
        str(wav_root),
    ]
    res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert res.returncode == 0, f"prep failed:\n{res.stdout}\n{res.stderr}"
    return out_dir, res.stdout


@pytest.fixture(scope="module")
def sc_prep(tmp_path_factory):
    root = make_sc_tree(tmp_path_factory.mktemp("sc_src") / "sc")
    out, stdout = run_prep(root, tmp_path_factory.mktemp("sc_out") / "npy")
    return out, stdout


def test_prep_writes_train_speaker_canvases_only(sc_prep):
    out, _ = sc_prep
    split = json.loads((out / "speaker_split.json").read_text())
    files = sorted(p.name for p in out.glob("*.npy"))

    assert len(split["held_out_speakers"]) == 2
    assert not set(split["train_speakers"]) & set(split["held_out_speakers"])
    assert files
    for name in files:
        speaker = name.split("__")[0]
        assert speaker in split["train_speakers"]
        assert speaker not in split["held_out_speakers"]

    # 3 train speakers x 4 words x 2 utterances, no collisions lost along the way.
    assert len(files) == len(split["train_speakers"]) * len(WORDS) * 2
    assert len(set(files)) == len(files)


def test_prep_npy_invariants_and_derived_T(sc_prep):
    out, _ = sc_prep
    manifest = json.loads((out / "prep_manifest.json").read_text())

    # T derived empirically, and identical to what AudioMNIST gets: same contract,
    # same 1.0 s clip (SC spec §0).
    assert manifest["mel_shape"] == [MEL_CONFIG["n_mels"], expected_mel_frames(NUM_SAMPLES, MEL_CONFIG)]
    assert manifest["canvas"] == [128, 128]
    assert manifest["offset"] == [0, 0]
    assert manifest["pad_value_normalized"] == -1.0
    assert manifest["channels"] == 1
    assert manifest["affine"]["m_hi"] > manifest["affine"]["m_lo"]
    assert manifest["audio_aug"]["enabled"] is False

    for path in out.glob("*.npy"):
        arr = np.load(path)
        assert arr.shape == (1, 128, 128)
        assert arr.dtype == np.float32
        assert arr.min() >= -1.0 and arr.max() <= 1.0
        # Sub-second sources: the waveform was zero-padded, so the tail of the real
        # region sits on the mel log-floor and maps to the normalized floor.
        assert np.isclose(arr[0, :, -1].max(), -1.0, atol=1e-5)   # canvas padding column


def test_prep_emits_clip_labels_with_speaker_and_word(sc_prep):
    out, _ = sc_prep
    labels = json.loads((out / "clip_labels.json").read_text())
    manifest = json.loads((out / "prep_manifest.json").read_text())
    files = sorted(p.stem for p in out.glob("*.npy"))

    assert labels["dataset"] == "speech_commands"
    assert labels["content_label_type"] == "word"
    assert labels["content_classes"] == sorted(WORDS)
    assert labels["digit_classes"] == ["one", "two"]     # in digit order, not alphabetical
    assert sorted(labels["clips"]) == files

    for stem, entry in labels["clips"].items():
        assert entry["speaker"] == da.speaker_from_stem(stem)
        assert entry["content"] == da.content_from_stem(stem, da.SPEECH_COMMANDS)

    # The manifest carries the summary (it lands in every checkpoint sidecar); the
    # 97k-entry per-clip map deliberately stays out of it.
    assert manifest["content_labels"]["classes"] == sorted(WORDS)
    assert manifest["content_labels"]["num_classes"] == len(WORDS)
    assert manifest["content_labels"]["digit_classes"] == ["one", "two"]
    assert "clips" not in manifest["content_labels"]


def test_prep_logs_the_background_skip_and_no_temp_left(sc_prep):
    out, stdout = sc_prep
    assert da.SC_BACKGROUND_DIR in stdout
    assert not list(out.glob(".mel_cache*"))


def test_prep_is_deterministic_and_disk_backend_agrees(tmp_path):
    """Same seed -> byte-identical .npy, and the memmap path produces exactly the
    same data as the in-RAM path (only the buffer differs)."""
    root = make_sc_tree(tmp_path / "sc")
    a, _ = run_prep(root, tmp_path / "a", ("--mel-cache", "memory"))
    b, _ = run_prep(root, tmp_path / "b", ("--mel-cache", "memory"))
    c, _ = run_prep(root, tmp_path / "c", ("--mel-cache", "disk"))

    names = sorted(p.name for p in a.glob("*.npy"))
    assert names and names == sorted(p.name for p in b.glob("*.npy")) == sorted(p.name for p in c.glob("*.npy"))
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
        np.testing.assert_allclose(np.load(a / name), np.load(c / name), rtol=0, atol=1e-5)

    m_a = json.loads((a / "prep_manifest.json").read_text())["affine"]["m_hi"]
    m_c = json.loads((c / "prep_manifest.json").read_text())["affine"]["m_hi"]
    np.testing.assert_allclose(m_a, m_c, rtol=0, atol=1e-5)


def test_prep_reference_manifest_catches_contract_drift(tmp_path):
    root = make_sc_tree(tmp_path / "sc")
    ref, _ = run_prep(root, tmp_path / "ref")

    # Matching contract: accepted.
    run_prep(root, tmp_path / "ok", ("--reference-manifest", str(ref / "prep_manifest.json")))

    # Drifted canvas geometry: must stop, not silently produce a second contract.
    bad = subprocess.run(
        [sys.executable, str(REPO / "prepare_audio_data.py"),
         "--dataset", "speech_commands", "--out", str(tmp_path / "bad"),
         "--holdout", "2", "--split-seed", "0",
         "--offset", "4", "4",
         "--reference-manifest", str(ref / "prep_manifest.json"), str(root)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    assert bad.returncode != 0
    assert "offset" in bad.stdout + bad.stderr


def test_prep_expect_speakers_is_fatal_when_wrong(tmp_path):
    root = make_sc_tree(tmp_path / "sc")
    res = subprocess.run(
        [sys.executable, str(REPO / "prepare_audio_data.py"),
         "--dataset", "speech_commands", "--out", str(tmp_path / "out"),
         "--holdout", "2", "--split-seed", "0",
         "--expect-speakers", str(da.SC_NUM_SPEAKERS), str(root)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "--expect-speakers" in res.stdout + res.stderr


# --------------------------------------------------------------------------- #
# Eval label provider (spec §5): speaker + 35-word + digit-subset coverage
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def eval_mod():
    sys.path.insert(0, str(REPO / "eval"))
    import convergence_curve

    return convergence_curve


def test_eval_label_fns_read_clip_labels(sc_prep, eval_mod):
    out, _ = sc_prep
    clip_labels = eval_mod.load_clip_labels(out)
    assert clip_labels is not None

    content_fn, K, meta = eval_mod.build_label_fn(
        "content", str(out), "speech_commands", None, clip_labels
    )
    assert K == len(WORDS)
    assert meta["classes"] == sorted(WORDS)
    # digit indices point at the digit words inside the full class list
    assert [meta["classes"][i] for i in meta["digit_indices"]] == ["one", "two"]

    speaker_fn, Kspk, _ = eval_mod.build_label_fn(
        "speaker", str(out), "speech_commands", None, clip_labels
    )
    split = json.loads((out / "speaker_split.json").read_text())
    assert Kspk == len(split["train_speakers"])

    for path in out.glob("*.npy"):
        stem = path.stem
        assert meta["classes"][content_fn(stem)] == clip_labels["clips"][stem]["content"]
        assert sorted(split["train_speakers"])[speaker_fn(stem)] == clip_labels["clips"][stem]["speaker"]


def test_eval_falls_back_to_stem_parsing_without_clip_labels(sc_prep, eval_mod):
    out, _ = sc_prep
    with_labels, _, meta_l = eval_mod.build_label_fn(
        "content", str(out), "speech_commands", None, eval_mod.load_clip_labels(out)
    )
    without, K, meta_n = eval_mod.build_label_fn(
        "content", str(out), "speech_commands", None, None
    )
    assert K == len(WORDS) and meta_n["classes"] == meta_l["classes"]
    for path in out.glob("*.npy"):
        assert without(path.stem) == with_labels(path.stem)


def test_digit_subset_coverage(eval_mod):
    classes = sorted(WORDS)                     # ['no', 'one', 'two', 'yes']
    digits = [classes.index("one"), classes.index("two")]

    # Half the samples land on digit words, evenly split between them.
    counts = np.array([10, 5, 5, 10])
    frac, ent = eval_mod.subset_coverage(counts, digits)
    assert frac == pytest.approx(10 / 30)
    assert ent == pytest.approx(1.0)            # perfectly even across the subset

    # All digit mass on one word => coverage collapses even though the fraction is fine.
    frac, ent = eval_mod.subset_coverage(np.array([0, 10, 0, 0]), digits)
    assert frac == pytest.approx(1.0) and ent == pytest.approx(0.0)

    # No digit subset (AudioMNIST) => the read is simply not defined.
    frac, ent = eval_mod.subset_coverage(counts, [])
    assert np.isnan(frac) and np.isnan(ent)
