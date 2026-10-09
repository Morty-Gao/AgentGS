#!/usr/bin/env python3
"""Extract a selected manuscript's images and accepted-revision inventory."""

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

PAPER_ROOT = Path(__file__).resolve().parents[1]
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
ASSETS = {
    "word/media/image1.png": "overview.png",
    "word/media/image2.png": "architecture.png",
    "word/media/image3.png": "interface.png",
    "word/media/image4.png": "performance.png",
    "word/media/image5.png": "case_study.png",
}


def accepted_nodes(element):
    """Read the final revision view without editing the source document."""
    if element.tag in {f"{{{NS['w']}}}del", f"{{{NS['w']}}}moveFrom"}:
        return
    yield element
    for child in element:
        yield from accepted_nodes(child)


def extract_text(element):
    tags = {f"{{{NS['w']}}}t", f"{{{NS['m']}}}t"}
    return "".join(node.text or "" for node in accepted_nodes(element) if node.tag in tags)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
    )
    args = parser.parse_args()
    figures = PAPER_ROOT / "figures"
    inventory_dir = PAPER_ROOT / "build" / "source"
    figures.mkdir(exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.source) as archive:
        for source, destination in ASSETS.items():
            path = figures / destination
            path.write_bytes(archive.read(source))
            print(path.relative_to(PAPER_ROOT))
        document = ET.fromstring(archive.read("word/document.xml"))
        records = []
        paragraph_index = 0
        table_index = 0
        for index, element in enumerate(document.find("w:body", NS)):
            record = {
                "index": index,
                "kind": element.tag.rsplit("}", 1)[-1],
                "text": extract_text(element),
                "math_count": len(element.findall(".//m:oMath", NS)),
            }
            if record["kind"] == "p":
                record["paragraph_index"] = paragraph_index
                paragraph_index += 1
            if record["kind"] == "tbl":
                record["table_index"] = table_index
                table_index += 1
                record["rows"] = [
                    [extract_text(cell) for cell in row.findall("w:tc", NS)]
                    for row in element.findall("w:tr", NS)
                ]
            record["math_xml"] = [
                ET.tostring(node, encoding="unicode")
                for node in accepted_nodes(element)
                if node.tag == f"{{{NS['m']}}}oMath"
            ]
            records.append(record)
    path = inventory_dir / "docx_inventory.json"
    path.write_text(
        json.dumps(
            {
                "source": args.source.name,
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "revision_view": "accept insertions; exclude deletions and moveFrom",
                "tracked_insertions": len(document.findall(".//w:ins", NS)),
                "tracked_deletions": len(document.findall(".//w:del", NS)),
                "assets": ASSETS,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(path.relative_to(PAPER_ROOT))


if __name__ == "__main__":
    main()
