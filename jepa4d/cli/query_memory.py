"""Query object records in a JEPA-4D SQLite memory artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from jepa4d.memory.persistence import MemoryPersistence

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    db: Annotated[Path, typer.Option("--db")],
    query: Annotated[str, typer.Option("--query", "-q")],
    limit: Annotated[int, typer.Option("--limit")] = 10,
) -> None:
    terms = query.lower().split()
    records = MemoryPersistence(db).list_kind("object")
    matches = [
        value
        for value in records
        if any(term in f"{value.get('category', '')} {value.get('description', '')}".lower() for term in terms)
    ][:limit]
    typer.echo(json.dumps({"query": query, "count": len(matches), "objects": matches}, indent=2))


if __name__ == "__main__":
    app()
