from .common import (
    Experiment,
    assert_balanced,
    assert_unique_templates,
    balanced_list_to_count,
    pluralize,
    shuffle_deterministic,
)

EXPERIMENT2_VERBS = [
    "blamed",
    "criticized",
    "praised",
    "defended",
    "embarrassed",
    "questioned",
    "congratulated",
    "protected",
    "reassured",
    "challenged",
    "humiliated",
]

HEAD_DISTRACTOR_PAIRS = [
    ("mother", "boy"),
    ("teacher", "student"),
    ("friend", "actor"),
    ("neighbor", "doctor"),
    ("sister", "manager"),
    ("colleague", "lawyer"),
    ("assistant", "professor"),
    ("cousin", "musician"),
    ("fan", "singer"),
    ("relative", "patient"),
    ("father", "girl"),
    ("brother", "dancer"),
    ("partner", "designer"),
    ("lawyer", "politician"),
    ("advisor", "scientist"),
    ("teacher", "researcher"),
    ("friend", "doctor"),
    ("neighbor", "student"),
    ("sister", "actor"),
    ("colleague", "manager"),
    ("assistant", "researcher"),
    ("cousin", "teacher"),
    ("fan", "actor"),
    ("relative", "doctor"),
    ("father", "student"),
    ("brother", "musician"),
    ("partner", "lawyer"),
    ("advisor", "professor"),
    ("mother", "dancer"),
    ("teacher", "singer"),
    ("friend", "designer"),
    ("neighbor", "musician"),
]

BASE_FIELDS = [
    "head_singular",
    "head_plural",
    "distractor_singular",
    "distractor_plural",
    "embedded_verb",
    "target",
]


def generate_base_items():
    assert len(HEAD_DISTRACTOR_PAIRS) == 32
    assert all(head != distractor for head, distractor in HEAD_DISTRACTOR_PAIRS)

    verbs = shuffle_deterministic(
        balanced_list_to_count(EXPERIMENT2_VERBS, 32),
        seed=201,
    )
    assert_balanced(verbs, label="embedded verbs", expected_total=32)

    items = []

    for index, ((head, distractor), verb) in enumerate(
        zip(HEAD_DISTRACTOR_PAIRS, verbs)
    ):
        head_plural = pluralize(head)
        distractor_plural = pluralize(distractor)

        assert head_plural != head
        assert distractor_plural != distractor

        items.append({
            "item_id": index + 1,
            "head_singular": head,
            "head_plural": head_plural,
            "distractor_singular": distractor,
            "distractor_plural": distractor_plural,
            "embedded_verb": verb,
            "target": "themselves",
        })

    assert_unique_templates(
        items,
        lambda item: (
            item["head_singular"],
            item["distractor_singular"],
            item["embedded_verb"],
        ),
        label="experiment2",
    )

    return items


def expand_conditions(item):
    target = item["target"]
    verb = item["embedded_verb"]

    conditions = []

    for head_match, head in [
        (1, item["head_plural"]),
        (0, item["head_singular"]),
    ]:
        for distractor_match, distractor in [
            (1, item["distractor_plural"]),
            (0, item["distractor_singular"]),
        ]:
            conditions.append({
                "condition": (
                    f"{'+' if head_match else '-'}Head"
                    f"{'+' if distractor_match else '-'}Distractor"
                ),
                "head_match": head_match,
                "distractor_match": distractor_match,
                "sentence": (
                    f"The {head} of the {distractor} "
                    f"{verb} {target}."
                ),
            })

    assert len(conditions) == 4
    return conditions


EXPERIMENT = Experiment(
    name="experiment2",
    base_fields=BASE_FIELDS,
    generate_base_items=generate_base_items,
    expand_conditions=expand_conditions,
)

generate = EXPERIMENT.generate
