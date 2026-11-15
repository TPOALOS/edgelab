"""Append-only JSON-lines trial ledger."""
import json
import os


def read(path):
    """Return the ledger entries (a missing file is an empty ledger). Corrupt lines are reported, not skipped silently."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("%s line %d is not valid JSON: %s" % (path, n, exc))
    return out


def append(path, entry):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def trial_key(entry):
    """What makes two runs the same hypothesis: strategy name, data, split and the columns used."""
    return (entry["name"], entry["data_sha256"], entry["split"], entry["ret_col"], entry.get("turnover_col"))
