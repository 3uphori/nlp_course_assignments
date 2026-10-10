import numpy as np
import pandas as pd
import librosa
import librosa.display
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, precision_score, recall_score
from IPython.display import Audio, HTML, display


def play_audio(signal, rate, label=None):
    if label is not None:
        print(label)
    display(Audio(signal, rate=rate))


def play_word_examples(dt, numbers):
    examples = {}
    for data, rate, word, *_ in dt:
        if word in numbers and word not in examples:
            examples[word] = (data, rate)

    for word in numbers:
        data, rate = examples[word]
        play_audio(data, rate, label=word)


def log_mel_spectrogram(signal, rate, mel_params):
    mel = librosa.feature.melspectrogram(y=signal, sr=rate, **mel_params)
    return librosa.power_to_db(mel, ref=np.max)

def plot_audio_analysis(dt, numbers, plots, cmap, ncols=3, colours=("salmon", "skyblue"), mel_params=None):
    examples = {}
    for data, rate, word, *_ in dt:
        if word in numbers and word not in examples:
            examples[word] = (data, rate)

    nrows = -(-len(numbers) // ncols)
    line_colours = {"wave": colours[0], "freq": colours[1]}

    for plot in plots:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
        axes = axes.ravel()

        for word, ax in zip(numbers, axes):
            data, rate = examples[word]
            if plot == "wave":
                librosa.display.waveshow(data, sr=rate, ax=ax, color=line_colours[plot])
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude")
                ax.set_title(word)
            elif plot == "freq":
                ax.plot(np.fft.rfftfreq(len(data), 1 / rate), np.abs(np.fft.rfft(data)),
                        color=line_colours[plot])
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Magnitude")
                ax.set_title(word)
            else:
                img = librosa.display.specshow(data, sr=rate, hop_length=mel_params["hop_length"],
                                               fmin=mel_params["fmin"], fmax=mel_params["fmax"],
                                               x_axis="time", y_axis="mel", cmap=cmap, ax=ax)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Mel Frequency")
                ax.set_title(word)
                fig.colorbar(img, ax=ax, format="%+2.0f dB")

        for ax in axes[len(numbers):]:
            ax.set_visible(False)

        fig.tight_layout()
        plt.show()


def add_noise(signal, kind, snr_db, rng):
    noise = rng.standard_normal(len(signal))
    if kind == "pink":
        spectrum = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(signal))
        freqs[0] = freqs[1]
        noise = np.fft.irfft(spectrum / np.sqrt(freqs), n=len(signal))

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    noise = noise * np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))

    return (signal + noise).astype(signal.dtype)


def random_test_recording(audio_dataset, test_speaker, seed=0):
    rows = [row for row in audio_dataset if row[3] == test_speaker]
    return rows[np.random.default_rng(seed).integers(len(rows))]


def compare_clean_noisy(signal, noisy, rate, word, noise_label, mel_params, cmap,
                        colours=("skyblue", "salmon")):
    players = "".join(
        f"<div><p>{label}</p>{Audio(data, rate=rate)._repr_html_()}</div>"
        for label, data in ((f"Clean — word: {word}", signal),
                            (f"{noise_label} — word: {word}", noisy))
    )
    display(HTML(f'<div style="display:flex; gap:24px; align-items:flex-end;">{players}</div>'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    for ax, data, title, colour in zip(axes, (signal, noisy), ("Clean", noise_label), colours):
        librosa.display.waveshow(data, sr=rate, ax=ax, color=colour)
        ax.set_title(f"{title} — word: {word}")
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True, layout="constrained")
    for ax, data, title in zip(axes, (signal, noisy), ("Clean", noise_label)):
        img = librosa.display.specshow(log_mel_spectrogram(data, rate, mel_params), sr=rate,
                                       hop_length=mel_params["hop_length"], fmin=mel_params["fmin"],
                                       fmax=mel_params["fmax"], x_axis="time", y_axis="mel",
                                       cmap=cmap, vmin=-80, vmax=0, ax=ax)
        ax.set_title(f"{title} — log-mel spectrogram — word: {word}")
        ax.set_xlabel("Time (s)")
    axes[1].set_ylabel("")
    fig.colorbar(img, ax=axes, format="%+2.0f dB")
    plt.show()


def preview_synthetic_noise(signal, rate, word, kind, snr_db, mel_params, cmap, seed=0):
    rng = np.random.default_rng(seed)
    noisy = add_noise(signal, kind, snr_db, rng)
    compare_clean_noisy(signal, noisy, rate, word, f"{kind.capitalize()} noise ({snr_db} dB)",
                        mel_params, cmap)


def noisy_test_loader(audio_dataset, features, numbers, test_speaker, kind, snr_db, mel_params,
                      batch_size=16, seed=0):
    rng = np.random.default_rng(seed)
    images, labels = [], []

    for signal, rate, word, speaker in audio_dataset:
        if speaker != test_speaker:
            continue

        noisy = add_noise(signal, kind, snr_db, rng)
        images.append(torch.from_numpy(log_mel_spectrogram(noisy, rate, mel_params)).float()[None])
        labels.append(numbers.index(word))

    x = (torch.stack(images) - features["mean"]) / features["std"]
    return DataLoader(TensorDataset(x, torch.tensor(labels)), batch_size=batch_size, shuffle=False)


def segment_noise(noise, segment_length, rng):
    n_segments = max(1, len(noise) // segment_length)

    needed = n_segments * segment_length
    offset = rng.integers(len(noise)) if rng is not None else 0
    tiled = np.tile(noise, -(-(needed + offset) // len(noise)))[offset:]

    return [tiled[i * segment_length:(i + 1) * segment_length] for i in range(n_segments)]


def add_real_noise(signal, noise, snr_db):
    if len(noise) < len(signal):
        noise = np.tile(noise, -(-len(signal) // len(noise)))
    noise = noise[:len(signal)]

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    noise = noise * np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))

    return (signal + noise).astype(signal.dtype)


def preview_real_noise(signal, rate, word, noise_segments, snr_db, mel_params, cmap, seed):
    rng = np.random.default_rng(seed)
    segment = noise_segments[rng.integers(len(noise_segments))]
    noisy = add_real_noise(signal, segment, snr_db)
    compare_clean_noisy(signal, noisy, rate, word, f"Rain noise ({snr_db} dB)", mel_params, cmap)


def real_noise_test_loader(audio_dataset, features, numbers, test_speaker, noise_segments, snr_db,
                           mel_params, seed, batch_size=16):
    rng = np.random.default_rng(seed)
    images, labels = [], []

    for signal, rate, word, speaker in audio_dataset:
        if speaker != test_speaker:
            continue

        segment = noise_segments[rng.integers(len(noise_segments))]
        noisy = add_real_noise(signal, segment, snr_db)
        images.append(torch.from_numpy(log_mel_spectrogram(noisy, rate, mel_params)).float()[None])
        labels.append(numbers.index(word))

    x = (torch.stack(images) - features["mean"]) / features["std"]
    return DataLoader(TensorDataset(x, torch.tensor(labels)), batch_size=batch_size, shuffle=False)


def predict(model, loader, device):
    model.eval()
    actual, probabilities = [], []
    with torch.no_grad():
        for data, targets in loader:
            logits = model(data.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu())
            actual.append(targets)

    probabilities = torch.cat(probabilities)
    return torch.cat(actual).numpy(), probabilities.argmax(dim=1).numpy(), probabilities.numpy()


def plot_confusion_matrix(actual, predicted, numbers, cmap, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        actual, predicted, labels=list(range(len(numbers))), display_labels=numbers,
        cmap=cmap, xticks_rotation=45, ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_probability_distributions(actual, predicted, probabilities, numbers, colours, title, ncols=5):
    confidence = probabilities.max(axis=1)
    nrows = -(-len(numbers) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    for label, (word, ax) in enumerate(zip(numbers, axes)):
        of_word = actual == label
        correct = predicted[of_word] == label
        word_confidence = confidence[of_word]

        ax.hist(
            [word_confidence[correct], word_confidence[~correct]],
            bins=10,
            range=(0, 1),
            color=colours,
            label=["Correct", "Wrong"],
            edgecolor="w",
            alpha=0.8,
            stacked=True
        )
        ax.set_title(f"{word} ({correct.sum()}/{len(correct)} correct)")
        sns.despine(ax=ax)

    for ax in axes[len(numbers):]:
        ax.set_visible(False)

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right")
    fig.supylabel("Number of recordings")
    fig.suptitle(f"Prediction Probability Distribution per Word ({title})")
    fig.tight_layout()
    plt.show()


def model_output(model, loader, numbers, colours, cmap, device, title):
    actual, predicted, probabilities = predict(model, loader, device)

    print()
    print("Test Performance:")
    print(classification_report(actual, predicted, labels=list(range(len(numbers))),
                                target_names=numbers, digits=4, zero_division=0))

    plot_confusion_matrix(actual, predicted, numbers, cmap, title)
    plot_probability_distributions(actual, predicted, probabilities, numbers, colours, title)

    return pd.DataFrame({
        "Actual": [numbers[i] for i in actual],
        "Predicted": [numbers[i] for i in predicted],
        "Predicted_Probability": probabilities.max(axis=1),
    })


def results_summary(model_results, numbers):
    rows = {}
    for name, results in model_results.items():
        actual, predicted = results["Actual"], results["Predicted"]

        row = {
            "Accuracy": (actual == predicted).mean(),
            "Precision": precision_score(actual, predicted, labels=numbers, average="macro", zero_division=0),
            "Recall": recall_score(actual, predicted, labels=numbers, average="macro", zero_division=0),
            "Average Probability": results["Predicted_Probability"].mean(),
        }
        for word in numbers:
            row[f"Accuracy: {word}"] = (predicted[actual == word] == word).mean()

        guesses = predicted.value_counts()
        row["Most Guessed"] = guesses.index[0]
        row["Most Guessed Share"] = guesses.iloc[0] / len(predicted)
        rows[name] = row

    return pd.DataFrame.from_dict(rows, orient="index")


def confusion_pairs(model_results, top=3):
    rows = {}
    for name, results in model_results.items():
        errors = results[results["Actual"] != results["Predicted"]]
        pairs = errors.groupby(["Actual", "Predicted"]).size().sort_values(ascending=False, kind="stable")

        row = {"Errors": f"{len(errors)}/{len(results)}"}
        for rank in range(top):
            if rank < len(pairs):
                (actual, predicted), count = pairs.index[rank], pairs.iloc[rank]
                row[f"Confusion {rank + 1}"] = f"{actual} predicted as {predicted} ({count})"
            else:
                row[f"Confusion {rank + 1}"] = ""
        rows[name] = row

    return pd.DataFrame.from_dict(rows, orient="index")


def plot_results_comparison(comparison, numbers):
    groups = comparison.index
    x = range(len(groups))
    metrics = ["Accuracy", "Precision", "Recall", "Average Probability"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    word_colours = plt.get_cmap("tab10").colors

    fig, (ax_metrics, ax_words) = plt.subplots(2, 1, figsize=(12, 10))

    styles = {
        "Accuracy": dict(linewidth=5, markersize=10, alpha=0.6),
        "Recall": dict(linewidth=2, linestyle="--", markerfacecolor="none", markersize=8),
    }
    for i, metric in enumerate(metrics):
        style = styles.get(metric, dict(linewidth=2))
        ax_metrics.plot(x, comparison[metric], marker=markers[i], label=metric, **style)
    ax_metrics.set_title("Overall scores by test group")

    for i, word in enumerate(numbers):
        ax_words.plot(x, comparison[f"Accuracy: {word}"], marker=markers[i],
                      color=word_colours[i], linewidth=2, label=word)
    ax_words.set_title("Accuracy per word by test group")

    for ax in (ax_metrics, ax_words):
        ax.set_xticks(list(x))
        ax.set_xticklabels(groups, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    plt.show()
