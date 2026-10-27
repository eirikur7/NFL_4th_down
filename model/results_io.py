"""Save and load simulation results as JSON files in results/."""
import json

import pandas as pd

from . import constants


def save_results(results, filename="results.json"):
    """Save a dict of numbers, lists and DataFrames. A DataFrame is stored as {"table": {...}}."""
    def encode(value):
        if isinstance(value, pd.Series):
            value = value.to_frame()
        if isinstance(value, pd.DataFrame):
            return {"table": value.to_dict(orient="split")}
        return value

    constants.RESULTS_DIR.mkdir(exist_ok=True)
    with open(constants.RESULTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump({key: encode(value) for key, value in results.items()}, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_results(filename="results.json"):
    """Load a results file (tables come back as DataFrames)."""
    def decode(value):
        if isinstance(value, dict) and set(value) == {"table"}:
            table = value["table"]
            return pd.DataFrame(table["data"], index=table["index"], columns=table["columns"])
        return value

    with open(constants.RESULTS_DIR / filename, encoding="utf-8") as f:
        return {key: decode(value) for key, value in json.load(f).items()}
