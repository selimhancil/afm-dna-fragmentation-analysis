"""Step 4: PROVISIONAL chronological order, taken from the supervisor's written rule.

Source of the rule
------------------
Email from Dr. Abdelhady to Selim Cil, 2026-10-29 12:54 PM:
* The numbering is meant to order the images by acquisition time.
* Plain numbers come first (1, 11, 111, ...), then the "A" group (A1 before A11),
  then the "B" group (B1 before B11), and so on.
* Some low-quality images were discarded, so the time gap between consecutive
  files is NOT constant (about 30 s is only typical).
* The raw data files with exact scan times are still to be sent.

What this module does and does not claim
----------------------------------------
* The order is marked "provisional" until it is checked against the raw
  acquisition timestamps.
* The order is built by parsing each filename (group letter, then the numeric
  value of the digits). It is NOT plain alphabetical sorting.
* No time axis is invented: acquisition_time_s and time_gap_to_previous_s stay
  empty. Do not assume evenly spaced frames.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from afm_pipeline.inventory import (
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
    InventoryError,
    find_image_files,
)

DEFAULT_ORDER_CSV = PROJECT_ROOT / "outputs" / "chronological_order_provisional.csv"

ORDER_SOURCE = "Dr. Abdelhady email 2026-10-29 12:54 PM (filename rule)"
ORDER_STATUS = "provisional: not yet verified against raw acquisition timestamps"

_NAME_PATTERN = re.compile(r"^(?P<prefix>[A-Z]?)(?P<digits>\d+)$")

ORDER_COLUMNS = [
    "chronological_rank",
    "filename",
    "name_prefix",
    "name_number",
    "acquisition_time_s",
    "time_gap_to_previous_s",
    "order_source",
    "order_status",
]


class OrderError(InventoryError):
    """The filenames do not follow the rule that was described."""


def parse_filename(filename: str) -> tuple[int, str, int]:
    """Return (group_rank, prefix_label, number) for a filename like 'A11.jpg'.

    group_rank: 0 for plain numbers, 1 for 'A', 2 for 'B', ...
    """
    stem = Path(filename).stem
    match = _NAME_PATTERN.match(stem)
    if match is None:
        raise OrderError(
            f"Filename '{filename}' does not follow the described pattern "
            "(optional single capital letter + digits, e.g. 11, A1, B11).\n"
            "Do not guess its position; ask the supervisor where it belongs."
        )
    prefix = match.group("prefix")
    group_rank = 0 if prefix == "" else ord(prefix) - ord("A") + 1
    label = prefix if prefix else "numeric_only"
    return group_rank, label, int(match.group("digits"))


def build_provisional_order(filenames: list[str]) -> pd.DataFrame:
    parsed = []
    for name in filenames:
        group_rank, label, number = parse_filename(name)
        parsed.append((group_rank, number, name, label))

    keys = [(g, n) for g, n, _, _ in parsed]
    if len(set(keys)) != len(keys):
        seen: dict = {}
        for g, n, name, _ in parsed:
            if (g, n) in seen:
                raise OrderError(
                    f"Files '{seen[(g, n)]}' and '{name}' have the same position "
                    "under the ordering rule. Resolve this with the supervisor."
                )
            seen[(g, n)] = name

    parsed.sort(key=lambda item: (item[0], item[1]))
    rows = []
    for rank, (_, number, name, label) in enumerate(parsed, start=1):
        rows.append(
            {
                "chronological_rank": rank,
                "filename": name,
                "name_prefix": label,
                "name_number": number,
                "acquisition_time_s": pd.NA,
                "time_gap_to_previous_s": pd.NA,
                "order_source": ORDER_SOURCE,
                "order_status": ORDER_STATUS,
            }
        )
    return pd.DataFrame(rows, columns=ORDER_COLUMNS)


def write_order_csv(df: pd.DataFrame, out_path: Path | str, data_dir: Path | str) -> Path:
    out_path = Path(out_path).resolve()
    data_resolved = Path(data_dir).resolve()
    if out_path == data_resolved or data_resolved in out_path.parents:
        raise InventoryError(
            f"Refusing to write the order file inside the data folder ({data_resolved}).\n"
            "Original images must stay untouched; choose an output path in outputs/."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the PROVISIONAL chronological order (supervisor's filename rule)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_ORDER_CSV)
    args = parser.parse_args(argv)

    try:
        images, _ = find_image_files(args.data_dir)
        names = [p.name for p in images]
        df = build_provisional_order(names)
        out = write_order_csv(df, args.output, args.data_dir)
    except InventoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    same_as_listing = df["filename"].tolist() == sorted(names)
    print(f"Files ordered: {len(df)}")
    print(
        "Same as plain filename sorting: "
        + ("yes (coincidence of this naming scheme, not the reason for the order)"
           if same_as_listing else "no")
    )
    print("")
    print("STATUS: PROVISIONAL. Order follows the supervisor's written rule and has")
    print("        not been verified against the raw acquisition timestamps yet.")
    print("NOTE:   Time gaps between images are NOT uniform (discarded frames).")
    print("        No time axis was assigned; time columns are empty.")
    print(f"\nOrder file written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
