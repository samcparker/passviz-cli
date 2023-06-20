animals = [
    {
        "type": "cat",
        "age": 5,
    },
    {
        "type": "dog",
        "age": 15,
    },
    {
        "type": "bear",
        "age": 12,
    },
]


sorted_by_age = sorted(
    animals, key=lambda x: x["age"]
)  # [{'type': 'cat', 'age': 5}, {'type': 'bear', 'age': 12}, {'type': 'dog', 'age': 15}]
sorted_by_type = sorted(
    animals, key=lambda x: x["type"]
)  # [{'type': 'bear', 'age': 12}, {'type': 'cat', 'age': 5}, {'type': 'dog', 'age': 15}]
