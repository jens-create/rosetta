import datasets
from rich.console import Console
from rich.table import Table


def print_as_table(dataset: datasets.Dataset, num_samples: int = 10, seed: int = 42) -> None:
    """Pretty print a random subset of a dataset in a table form."""
    console = Console()

    num_samples = min(num_samples, len(dataset))

    subset = dataset.shuffle(seed=seed).select(range(num_samples))

    table = Table(show_header=True, header_style="bold magenta")
    for column_name in dataset.column_names:
        table.add_column(column_name, style="dim")

    for item, _ in enumerate(subset):
        row_data = [str(subset[item][col]) for col in dataset.column_names]
        table.add_row(*row_data)

    console.print(table)
