from .common import (
    Experiment,
    assert_balanced,
    assert_gender_balance,
    assert_unique_templates,
    balanced_list,
    balanced_sequence,
    balanced_unique_name_pairs,
    shuffle_deterministic,
)
from .lexicon import (
    EMBEDDED_VERBS,
    FEMALE_NAMES,
    MALE_NAMES,
    MATRIX_FRAMES,
)

BASE_FIELDS = ["target", "matrix_frame", "embedded_verb"]
N_ITEMS = 32


def _template_key(item):
    return (
        item["target"],
        item["matrix_male"],
        item["matrix_female"],
        item["local_male"],
        item["local_female"],
        item["matrix_frame"],
        item["embedded_verb"],
    )


def generate_base_items():
    targets = shuffle_deterministic(
        balanced_sequence(["himself", "herself"], N_ITEMS),
        seed=101,
    )
    verbs = shuffle_deterministic(
        balanced_list(EMBEDDED_VERBS, repetitions=4),
        seed=102,
    )
    matrix_frames = shuffle_deterministic(
        balanced_list(MATRIX_FRAMES, repetitions=8),
        seed=103,
    )

    male_pairs = balanced_unique_name_pairs(MALE_NAMES, N_ITEMS, seed=104)
    female_pairs = balanced_unique_name_pairs(FEMALE_NAMES, N_ITEMS, seed=105)

    name_quadruples = shuffle_deterministic(
        list(zip(male_pairs, female_pairs)),
        seed=106,
    )

    assert_gender_balance([{"target": target} for target in targets])
    assert_balanced(verbs, label="embedded verbs", expected_total=N_ITEMS)
    assert_balanced(matrix_frames, label="matrix frames", expected_total=N_ITEMS)

    items = [
        {
            "item_id": index + 1,
            "target": targets[index],
            "matrix_male": name_quadruples[index][0][0],
            "local_male": name_quadruples[index][0][1],
            "matrix_female": name_quadruples[index][1][0],
            "local_female": name_quadruples[index][1][1],
            "matrix_frame": matrix_frames[index],
            "embedded_verb": verbs[index],
        }
        for index in range(N_ITEMS)
    ]

    assert_unique_templates(items, _template_key, label="experiment1")

    return items


def _gendered_roles(item, target):
    if target == "himself":
        return (
            item["matrix_male"],
            item["matrix_female"],
            item["local_male"],
            item["local_female"],
        )

    return (
        item["matrix_female"],
        item["matrix_male"],
        item["local_female"],
        item["local_male"],
    )


def expand_conditions(item):
    target = item["target"]
    matching_matrix, mismatching_matrix, matching_local, mismatching_local = (
        _gendered_roles(item, target)
    )
    frame = item["matrix_frame"]
    verb = item["embedded_verb"]

    conditions = []
    for local_match, local_name in ((1, matching_local), (0, mismatching_local)):
        for matrix_match, matrix_name in (
            (1, matching_matrix),
            (0, mismatching_matrix),
        ):
            local_label = "+" if local_match else "-"
            matrix_label = "+" if matrix_match else "-"
            conditions.append({
                "condition": f"{local_label}Local{matrix_label}Matrix",
                "local_match": local_match,
                "matrix_match": matrix_match,
                "sentence": f"{matrix_name} {frame} {local_name} {verb} {target}.",
            })

    return conditions


EXPERIMENT = Experiment(
    name="experiment1",
    base_fields=BASE_FIELDS,
    generate_base_items=generate_base_items,
    expand_conditions=expand_conditions,
)
generate = EXPERIMENT.generate
