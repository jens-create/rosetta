import datasets


def load_hf_dataset(
    name_or_path: str,
    split: str,
    subset: str | None = None,
    num_samples: int | None = None,
    seed: int | None = 42,
    cache_dir: str | None = None,
) -> datasets.Dataset:
    """Load a HuggingFace dataset and transform to a given task."""
    dataset = datasets.load_dataset(name_or_path, subset, split=split, cache_dir=cache_dir)
    if isinstance(
        dataset,
        (datasets.DatasetDict, datasets.IterableDataset, datasets.IterableDatasetDict),
    ):
        raise NotImplementedError(f"`{type(dataset)}` not supported.")

    if num_samples:
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    return dataset
