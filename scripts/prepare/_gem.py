"""Losslessly export GEM workbook cells, including cached formulas, to JSONL."""

import json
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from _common import sha256

from agentgs.datasets.parsers.energy import (
    MAIN_NS,
    column_index,
    xlsx_shared_strings,
    xlsx_sheet_target,
)


def convert_workbook(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".prepare")
    counts = {}
    try:
        with zipfile.ZipFile(source) as archive, temporary.open("w", encoding="utf-8") as out:
            strings = xlsx_shared_strings(archive)
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            sheets = [sheet.attrib["name"] for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet")]

            def write(value):
                out.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

            write(
                {
                    "record": "workbook",
                    "schema_version": 1,
                    "source_filename": source.name,
                    "source_sha256": sha256(source),
                    "sheets": sheets,
                }
            )
            for sheet in sheets:
                count = 0
                with archive.open(xlsx_sheet_target(archive, sheet)) as incoming:
                    for _, element in ET.iterparse(incoming, events=("end",)):
                        if element.tag != f"{{{MAIN_NS}}}row":
                            continue
                        values = []
                        formulas = {}
                        for cell in element.findall(f"{{{MAIN_NS}}}c"):
                            position = column_index(cell.attrib.get("r", "A1"))
                            while len(values) <= position:
                                values.append(None)
                            kind = cell.attrib.get("t", "n")
                            node = cell.find(f"{{{MAIN_NS}}}v")
                            raw = node.text if node is not None else None
                            value = raw
                            if kind == "s" and raw is not None:
                                value = strings[int(raw)]
                            elif kind == "inlineStr":
                                inline = cell.find(f"{{{MAIN_NS}}}is")
                                value = (
                                    "".join(
                                        (part.text or "") for part in inline.iter(f"{{{MAIN_NS}}}t")
                                    )
                                    if inline is not None
                                    else None
                                )
                            elif kind == "b" and raw is not None:
                                value = raw == "1"
                            elif kind == "n" and raw is not None:
                                value = int(raw) if raw.lstrip("-+").isdigit() else float(raw)
                                if isinstance(value, float) and not math.isfinite(value):
                                    raise ValueError(
                                        f"Non-finite Excel numeric cell: {sheet}!{cell.attrib.get('r')}"
                                    )
                            values[position] = value
                            formula = cell.find(f"{{{MAIN_NS}}}f")
                            if formula is not None:
                                formulas[cell.attrib["r"]] = {
                                    "text": formula.text,
                                    "attributes": formula.attrib,
                                    "cached_value": value,
                                }
                        record = {
                            "record": "row",
                            "sheet": sheet,
                            "row": int(element.attrib.get("r", count + 1)),
                            "values": values,
                        }
                        if formulas:
                            record["formulas"] = formulas
                        write(record)
                        count += 1
                        element.clear()
                counts[sheet] = count
            write({"record": "summary", "rows_by_sheet": counts})
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
