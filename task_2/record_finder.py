from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

root = Path(r"C:\Users\Lenovo\Downloads\archive")

number_words = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
]

# speaker -> {word: recording_count}
speaker_words = {}

for word in number_words:
    folder = root / word

    if not folder.is_dir():
        print(f"Missing folder: {word}")
        continue

    # Count recordings per speaker
    speaker_counts = Counter(
        f.name[:8]
        for f in folder.iterdir()
        if f.is_file()
    )

    # Keep speakers with 5+ recordings
    for speaker, count in speaker_counts.items():
        if count >= 5:
            if speaker not in speaker_words:
                speaker_words[speaker] = {}

            speaker_words[speaker][word] = count

    print(
        f"{word}: "
        f"{sum(count >= 5 for count in speaker_counts.values())} "
        f"qualifying speakers"
    )


# Sort speakers by number of words they qualify for
results = sorted(
    speaker_words.items(),
    key=lambda x: len(x[1]),
    reverse=True
)


print("\nSpeakers across the 10 number words:\n")

for speaker, words in results:
    print(
        f"{speaker}: {len(words)}/10 words | "
        f"{words}"
    )