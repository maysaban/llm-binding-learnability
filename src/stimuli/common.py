import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SEED = 42
random.seed(SEED)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def output_dir(experiment_name: str) -> Path:
    return DATA_ROOT / experiment_name / "generated"


def balanced_sequence(values, n_items):
    """Build a list of length n_items with counts differing by at most 1."""
    base = n_items // len(values)
    remainder = n_items % len(values)

    sequence = []
    for index, value in enumerate(values):
        count = base + (1 if index < remainder else 0)
        sequence.extend([value] * count)

    assert len(sequence) == n_items
    return sequence


def balanced_list(values, repetitions):
    """Repeat each value exactly `repetitions` times."""
    items = []
    for value in values:
        items.extend([value] * repetitions)

    assert len(items) == len(values) * repetitions
    return items


def balanced_list_to_count(values, n_items):
    """Alias kept for experiment modules: exact near-equal counts, deterministic."""
    return balanced_sequence(values, n_items)


def balanced_name_pair_sequences(names, n_items):
    """Two balanced name lists where matrix and local names never match."""
    matrix_names = balanced_sequence(names, n_items)
    local_names = balanced_sequence(names, n_items)

    shift = len(names) // 2 + 1
    local_names = local_names[shift:] + local_names[:shift]

    for index in range(n_items):
        if matrix_names[index] == local_names[index]:
            swap_with = next(
                candidate
                for candidate in range(n_items)
                if candidate != index
                and local_names[candidate] != matrix_names[index]
                and local_names[candidate] != matrix_names[candidate]
            )
            local_names[index], local_names[swap_with] = (
                local_names[swap_with],
                local_names[index],
            )

    assert all(
        matrix_name != local_name
        for matrix_name, local_name in zip(matrix_names, local_names)
    )
    return matrix_names, local_names


def balanced_unique_name_pairs(names, n_items, *, seed):
    """Select unique (matrix, local) pairs with balanced name counts."""
    matrix_roles = shuffle_deterministic(
        balanced_sequence(names, n_items),
        seed=seed,
    )
    local_limits = Counter(balanced_sequence(names, n_items))

    selected_locals = []
    local_counts = Counter()
    used_pairs = set()

    def assign(index):
        if index == n_items:
            return True

        matrix = matrix_roles[index]
        candidates = sorted(
            name
            for name in names
            if name != matrix
        )

        for local in candidates:
            if local_counts[local] >= local_limits[local]:
                continue
            if (matrix, local) in used_pairs:
                continue

            selected_locals.append(local)
            local_counts[local] += 1
            used_pairs.add((matrix, local))

            if assign(index + 1):
                return True

            selected_locals.pop()
            local_counts[local] -= 1
            used_pairs.remove((matrix, local))

        return False

    assert assign(0), f"Could not build {n_items} unique name pairs from {names}"

    pairs = list(zip(matrix_roles, selected_locals))
    assert len(set(pairs)) == n_items
    assert_balanced(
        [matrix for matrix, _ in pairs],
        label="matrix names",
        expected_total=n_items,
    )
    assert_balanced(
        [local for _, local in pairs],
        label="local names",
        expected_total=n_items,
    )
    return pairs


def shuffle_deterministic(sequence, seed):
    """Shuffle a copy of sequence with a fixed seed; counts stay identical."""
    items = list(sequence)
    random.Random(seed).shuffle(items)
    return items


def assert_balanced(sequence, *, label="", expected_total=None):
    """Verify counts differ by at most 1; raise if not."""
    counts = Counter(sequence)
    if expected_total is not None:
        assert sum(counts.values()) == expected_total

    if not counts:
        return

    count_values = list(counts.values())
    assert max(count_values) - min(count_values) <= 1, (
        f"{label} is not balanced: {dict(counts)}"
    )


def pluralize(noun: str) -> str:
    return noun + "s"


def items_from_rows(field_names, rows):
    """Build numbered item dicts from tuples of row values."""
    return [
        {"item_id": index + 1, **dict(zip(field_names, row))}
        for index, row in enumerate(rows)
    ]


def assert_gender_balance(items, *, target_field="target", n_per_gender=16):
    assert len(items) == n_per_gender * 2
    assert sum(item[target_field] == "himself" for item in items) == n_per_gender
    assert sum(item[target_field] == "herself" for item in items) == n_per_gender


def assert_unique_templates(items, key_fn, *, label=""):
    keys = [key_fn(item) for item in items]
    assert len(set(keys)) == len(items), (
        f"{label}: duplicate lexical templates found"
    )


def assert_unique_sentences(items, expand_fn, *, label=""):
    sentences = [
        condition["sentence"]
        for item in items
        for condition in expand_fn(item)
    ]
    assert len(set(sentences)) == len(sentences), (
        f"{label}: duplicate sentences found"
    )


def save_base_items(items, output_path):
    fieldnames = list(items[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)


def save_expanded_stimuli(items, output_path, expand_fn, base_fields):
    rows = []

    for item in items:
        for condition in expand_fn(item):
            rows.append({
                "item_id": item["item_id"],
                **{field: item[field] for field in base_fields},
                **condition,
            })

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_sentences(items, output_path, expand_fn):
    rows = []

    for item in items:
        for condition in expand_fn(item):
            rows.append({
                "item_id": item["item_id"],
                "condition": condition["condition"],
                "sentence": condition["sentence"],
                "target": item.get("target"),
            })

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_and_save(experiment_name, generate_fn, expand_fn, base_fields):
    out = output_dir(experiment_name)
    out.mkdir(parents=True, exist_ok=True)

    items = generate_fn()

    assert_unique_sentences(items, expand_fn, label=experiment_name)

    save_base_items(items, out / "base_items.csv")
    save_expanded_stimuli(items, out / "stimuli.csv", expand_fn, base_fields)
    save_sentences(items, out / "sentences.csv", expand_fn)

    return items


@dataclass(frozen=True)
class Experiment:
    name: str
    base_fields: list[str]
    generate_base_items: Callable[[], list[dict]]
    expand_conditions: Callable[[dict], list[dict]]

    def generate(self):
        return generate_and_save(
            self.name,
            self.generate_base_items,
            self.expand_conditions,
            self.base_fields,
        )
