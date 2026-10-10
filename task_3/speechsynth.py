import gc
from IPython.display import Audio, HTML, display
import json
import librosa
import librosa.display
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, welch
import soundfile as sf
import time
import torch
from trainer import Trainer, TrainerArgs
from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.vits import Vits


LANGUAGES = ["en", "fr-fr", "pt-br", "lt"]


def play_audio(signal, rate, label=None):
    if label is not None:
        print(label)
    display(Audio(signal, rate=rate))


def play_side_by_side(signals):
    players = "".join(
        f"<div><p>{label}</p>{Audio(y, rate=sr)._repr_html_()}</div>"
        for label, (y, sr) in signals.items()
    )
    display(HTML(f'<div style="display:flex; gap:24px; align-items:flex-end;">{players}</div>'))


def load_recording(path, sr=None):
    path = Path(path)
    return librosa.load(path, sr=sr)


def recording_details(path):
    path = Path(path)
    
    info = sf.info(path)
    bit_depths = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64}
    return {
        "File": path.name,
        "Duration (s)": round(info.duration, 3),
        "Sampling frequency (Hz)": info.samplerate,
        "Bit depth": bit_depths.get(info.subtype),
        "Channels": info.channels,
        "Format": info.format,
        "Encoding": info.subtype_info,
    }


def plot_waveforms(signals, colours):
    fig, axes = plt.subplots(1, len(signals), figsize=(6 * len(signals), 3.5),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (label, (y, sr)), colour in zip(axes[0, :], signals.items(), colours):
        librosa.display.waveshow(y / np.max(np.abs(y)), sr=sr, ax=ax, color=colour)
        ax.set_title(f"{label} ({len(y) / sr:.2f} s, {sr} Hz)")
        ax.set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude (relative to peak)")

    plt.tight_layout()
    plt.show()


def plot_magnitude_spectra(signals, colours, frame_s=0.04, floor_db=-80):
    fig, ax = plt.subplots(figsize=(12, 4))
    for (label, (y, sr)), colour in zip(signals.items(), colours):
        freqs, power = welch(y, fs=sr, nperseg=int(frame_s * sr))
        db = 10 * np.log10(power / power.max() + 1e-12)
        ax.plot(freqs, db, color=colour, linewidth=1.5, label=f"{label} (up to {sr // 2} Hz)")

    ax.set_xlim(0, max(sr for _, sr in signals.values()) / 2)
    ax.set_ylim(floor_db, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB off peak)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_spectrograms(signals, cmap, frame_s=0.025, hop_s=0.01, floor_db=-80):
    fig, axes = plt.subplots(1, len(signals), figsize=(6 * len(signals), 4),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for ax, (label, (y, sr)) in zip(axes[0, :], signals.items()):
        win_length = int(frame_s * sr)
        n_fft = int(2 ** np.ceil(np.log2(win_length)))
        hop_length = int(hop_s * sr)
        magnitude = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        image = librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), sr=sr,
                                         hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax,
                                         cmap=cmap, vmin=floor_db, vmax=0)
        ax.set_title(f"{label} ({len(y) / sr:.2f} s, {sr} Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("")
    axes[0, 0].set_ylabel("Frequency (Hz)")

    axes[0, 0].set_xlim(0, max(len(y) / sr for y, sr in signals.values()))
    axes[0, 0].set_ylim(0, max(sr for _, sr in signals.values()) / 2)
    fig.colorbar(image, ax=axes[0, :], format="%+.0f dB", label="Level (dB off peak)")
    plt.show()


def f0_contours(signals, fmin=75, fmax=300, lowpass_hz=1000, min_voiced_s=0.04):
    contours = {}
    for label, (y, sr) in signals.items():
        y = sosfiltfilt(butter(6, lowpass_hz, "low", fs=sr, output="sos"), y)
        frame_length = int(2 ** np.ceil(np.log2(0.04 * sr)))
        hop_length = frame_length // 4
        f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, beta_parameters=(2, 8),
                                frame_length=frame_length, hop_length=hop_length)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], ~np.isnan(f0), [0]]).astype(int)))
        for start, end in zip(edges[::2], edges[1::2]):
            if (end - start) * hop_length / sr < min_voiced_s:
                f0[start:end] = np.nan
        contours[label] = (librosa.times_like(f0, sr=sr, hop_length=hop_length), f0)
    return contours


def normalised_f0(y, sr, contour, top_db=30):
    times, f0 = contour
    _, (start, end) = librosa.effects.trim(y, top_db=top_db)
    position = (times - start / sr) / ((end - start) / sr)
    inside = (position >= 0) & (position <= 1)
    return position[inside], 12 * np.log2(f0[inside] / np.nanmedian(f0))


def plot_f0_normalised(signals, contours, colours, top_db=30):
    fig, ax = plt.subplots(figsize=(12, 4))
    for (label, (y, sr)), colour in zip(signals.items(), colours):
        position, semitones = normalised_f0(y, sr, contours[label], top_db)
        ax.plot(position, semitones, color=colour, linewidth=2, marker=".", markersize=4, label=label)

    limit = max(abs(y) for y in ax.get_ylim())
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalised time (0 = start of speech, 1 = end of speech)")
    ax.set_ylabel("F0 (semitones off median)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def signal_comparison(signals, contours, top_db=30, common_sr=None):
    common_sr = common_sr or min(sr for _, sr in signals.values())
    grid = np.linspace(0, 1, 101)
    patterns = {}
    rows = {}
    for label, (y, sr) in signals.items():
        speech, _ = librosa.effects.trim(y, top_db=top_db)
        speech_common = librosa.resample(speech, orig_sr=sr, target_sr=common_sr)
        f0 = contours[label][1]
        position, semitones = normalised_f0(y, sr, contours[label], top_db)
        
        patterns[label] = np.interp(grid, position, semitones)

        rows[label] = {
            "Duration (s)": len(y) / sr,
            "Speech duration (s)": len(speech) / sr,
            "Sampling frequency (Hz)": sr,
            "RMS level (dBFS)": 20 * np.log10(np.sqrt(np.mean(speech ** 2))),
            "Spectral centroid (Hz)": librosa.feature.spectral_centroid(y=speech_common, sr=common_sr).mean(),
            "F0 mean (Hz)": np.nanmean(f0),
            "F0 std (Hz)": np.nanstd(f0),
            "F0 min (Hz)": np.nanmin(f0),
            "F0 max (Hz)": np.nanmax(f0),
            "F0 range (semitones)": 12 * np.log2(np.nanmax(f0) / np.nanmin(f0)),
        }

    return pd.DataFrame(rows)


def to_16k(sources, target_dir, sr=16000):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    for source in sources:
        target = target_dir / Path(source).name
        if not target.exists():
            y, _ = librosa.load(source, sr=sr, mono=True)
            sf.write(target, y, sr, subtype="PCM_16")
        targets.append(target)
    return targets


def samples_to_16k(samples, target_dir):
    targets = to_16k([sample["audio_file"] for sample in samples], target_dir)
    return [{**sample, "audio_file": str(target)} for sample, target in zip(samples, targets)]


def compute_d_vectors(samples, speaker_manager, file_path):
    d_vectors = {s["audio_unique_name"]: {"name": s["speaker_name"],
                                          "embedding": speaker_manager.compute_embedding_from_clip(s["audio_file"])}
                 for s in samples}
    torch.save(d_vectors, file_path)


def missing_characters(model_dir, texts):
    config = load_config(str(Path(model_dir) / "config.json"))
    known = set(config.characters.characters + config.characters.punctuations)
    return "".join(sorted(set("".join(texts).lower()) - known - {" "}))


def make_config(model_dir, d_vector_file, new_chars, params, epochs, output_path, run_name, seed, save=False):
    config = load_config(str(Path(model_dir) / "config.json"))
    config.characters.characters += new_chars
    config.datasets = [BaseDatasetConfig(language=language) for language in LANGUAGES]
    config.d_vector_file = config.model_args.d_vector_file = [str(d_vector_file)]
    config.use_language_weighted_sampler = False
    config.test_sentences = []
    config.epochs = epochs
    config.output_path = str(output_path)
    config.run_name = run_name
    config.training_seed = seed
    config.target_loss = "loss_mel"
    config.num_loader_workers = config.num_eval_loader_workers = 0
    config.print_step, config.print_eval = 50, False
    config.save_checkpoints = save
    config.save_best_after = 0 if save else 10**9

    for name, value in params.items():
        if name == "lr_decay":
            config.lr_scheduler_gen_params["gamma"] = config.lr_scheduler_disc_params["gamma"] = value
        elif name == "weight_decay":
            config.optimizer_params["weight_decay"] = value
        elif name == "grad_clip":
            config.grad_clip = [value, value]
        elif name == "batch_size":
            config.batch_size = config.eval_batch_size = value
        elif hasattr(config.model_args, name):
            setattr(config.model_args, name, value)
        else:
            setattr(config, name, value)
    return config


def load_pretrained(config, model_dir, new_chars):
    model = Vits.init_from_config(config, verbose=False)
    pretrained = torch.load(Path(model_dir) / "model_file.pth", map_location="cpu", weights_only=False)["model"]
    pretrained = {k: v for k, v in pretrained.items() if not k.startswith("speaker_encoder.")}

    emb = model.text_encoder.emb.weight.detach().clone()
    kept = [row for row, char in enumerate(model.tokenizer.characters.vocab) if char not in new_chars]
    emb[kept] = pretrained["text_encoder.emb.weight"]
    emb_l = model.emb_l.weight.detach().clone()
    old_languages = json.loads((Path(model_dir) / "language_ids.json").read_text())
    for language, row in model.language_manager.name_to_id.items():
        if language in old_languages:
            emb_l[row] = pretrained["emb_l.weight"][old_languages[language]]
    pretrained.update({"text_encoder.emb.weight": emb, "emb_l.weight": emb_l})

    model.load_state_dict(pretrained)
    return model


def fine_tune(config, model_dir, new_chars, train_samples, eval_samples):
    model = load_pretrained(config, model_dir, new_chars)
    history = []

    def log_epoch(trainer):
            history.append({**{"train_" + k[4:]: v for k, v in trainer.keep_avg_train.avg_values.items()},
                            **{"val_" + k[4:]: v for k, v in trainer.keep_avg_eval.avg_values.items()}})

    trainer = Trainer(TrainerArgs(), config, config.output_path, model=model, train_samples=train_samples,
                      eval_samples=eval_samples, parse_command_line_args=False, callbacks={"on_epoch_end": log_epoch})
    run_path = trainer.output_path
    start = time.time()
    try:
        trainer._fit()
    except torch.cuda.OutOfMemoryError:
        print(" > Out of GPU memory, run skipped")
        history.clear()
    finally:
        training_time = time.time() - start
        trainer.dashboard_logger.finish()
        trainer_logger = logging.getLogger("trainer")
        for handler in trainer_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                trainer_logger.removeHandler(handler)
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(history), training_time, run_path
