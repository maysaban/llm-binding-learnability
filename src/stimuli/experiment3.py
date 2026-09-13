from .common import (
    Experiment,
    assert_gender_balance,
    assert_unique_templates,
    items_from_rows,
)

BASE_FIELDS = [
    "target",
    "matrix_antecedent",
    "local_matching",
    "local_mismatching",
    "matrix_frame",
    "coarg_verb",
    "picture_frame",
]

BASE_ITEMS = [
    ("himself", "John", "Bill", "Mary", "believes that", "criticized", "saw a picture of"),
    ("herself", "Sarah", "Emma", "Daniel", "thinks that", "praised", "liked a picture of"),
    ("himself", "Peter", "James", "Laura", "was glad that", "defended", "found a picture of"),
    ("herself", "Rachel", "Hannah", "Michael", "believes that", "embarrassed", "bought a picture of"),
    ("himself", "David", "Tom", "Susan", "thinks that", "blamed", "looked at a picture of"),
    ("herself", "Laura", "Mary", "Robert", "thought that", "admired", "saw a picture of"),
    ("himself", "Michael", "Daniel", "Anna", "believes that", "humiliated", "liked a picture of"),
    ("herself", "Rebecca", "Emily", "Ben", "was glad that", "photographed", "found a picture of"),
    ("himself", "James", "Peter", "Hannah", "was upset that", "praised", "bought a picture of"),
    ("herself", "Emma", "Susan", "David", "thinks that", "defended", "looked at a picture of"),
    ("himself", "Robert", "Ben", "Rachel", "believes that", "embarrassed", "saw a picture of"),
    ("herself", "Mary", "Anna", "Tom", "was glad that", "blamed", "liked a picture of"),
    ("himself", "Daniel", "Michael", "Laura", "thinks that", "admired", "found a picture of"),
    ("herself", "Hannah", "Rachel", "James", "thought that", "criticized", "bought a picture of"),
    ("himself", "Ben", "John", "Emily", "believes that", "photographed", "looked at a picture of"),
    ("herself", "Susan", "Rebecca", "Peter", "was glad that", "humiliated", "saw a picture of"),
    ("himself", "Tom", "Robert", "Emma", "thinks that", "defended", "liked a picture of"),
    ("herself", "Anna", "Laura", "Michael", "was upset that", "praised", "found a picture of"),
    ("himself", "Peter", "Daniel", "Susan", "believes that", "blamed", "bought a picture of"),
    ("herself", "Emily", "Hannah", "Ben", "was glad that", "embarrassed", "looked at a picture of"),
    ("himself", "John", "James", "Rebecca", "thought that", "humiliated", "saw a picture of"),
    ("herself", "Rachel", "Mary", "David", "thinks that", "admired", "liked a picture of"),
    ("himself", "Michael", "Tom", "Anna", "believes that", "criticized", "found a picture of"),
    ("herself", "Laura", "Emma", "Robert", "was glad that", "photographed", "bought a picture of"),
    ("himself", "David", "Ben", "Hannah", "thinks that", "praised", "looked at a picture of"),
    ("herself", "Rebecca", "Susan", "Peter", "was upset that", "defended", "saw a picture of"),
    ("himself", "James", "Michael", "Emily", "believes that", "embarrassed", "liked a picture of"),
    ("herself", "Emma", "Rachel", "Tom", "was glad that", "blamed", "found a picture of"),
    ("himself", "Robert", "Daniel", "Mary", "thought that", "admired", "bought a picture of"),
    ("herself", "Mary", "Anna", "John", "thinks that", "humiliated", "looked at a picture of"),
    ("himself", "Daniel", "Peter", "Laura", "believes that", "photographed", "saw a picture of"),
    ("herself", "Hannah", "Rebecca", "James", "was glad that", "criticized", "liked a picture of"),
]


def generate_base_items():
    items = items_from_rows(BASE_FIELDS, BASE_ITEMS)
    assert_gender_balance(items)
    assert_unique_templates(
        items,
        lambda item: tuple(item[field] for field in BASE_FIELDS),
        label="experiment3",
    )
    return items


def expand_conditions(item):
    matrix = item["matrix_antecedent"]
    frame = item["matrix_frame"]
    target = item["target"]

    conditions = []
    for local_match, local_key, local_label in (
        (1, "local_matching", "+Local"),
        (0, "local_mismatching", "-Local"),
    ):
        local = item[local_key]
        for context_type, middle_key, context_label in (
            ("coarg", "coarg_verb", "+Coarg"),
            ("picture_np", "picture_frame", "+Picture"),
        ):
            conditions.append({
                "condition": f"{local_label}{context_label}",
                "local_match": local_match,
                "context_type": context_type,
                "sentence": (
                    f"{matrix} {frame} {local} {item[middle_key]} {target}."
                ),
            })

    return conditions


EXPERIMENT = Experiment(
    name="experiment3",
    base_fields=BASE_FIELDS,
    generate_base_items=generate_base_items,
    expand_conditions=expand_conditions,
)
generate = EXPERIMENT.generate
