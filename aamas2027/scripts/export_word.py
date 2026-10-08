#!/usr/bin/env python3
"""Export the complete English translation to an editable Word manuscript."""

import argparse
import json
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)


def group(text, start):
    """Read a balanced LaTeX argument, retaining nested braces."""
    while text[start].isspace():
        start += 1
    assert text[start] == "{", text[start:start + 80]
    depth = 1
    end = start + 1
    while depth:
        if text[end] in "{}" and text[end - 1] != "\\":
            depth += 1 if text[end] == "{" else -1
        end += 1
    return text[start + 1:end - 1], end


def commands(text, name, nargs, transform):
    pattern = re.compile(r"\\" + re.escape(name) + r"(?![A-Za-z])")
    while match := pattern.search(text):
        end = match.end()
        arguments = []
        for _ in range(nargs):
            argument, end = group(text, end)
            arguments.append(argument)
        text = text[:match.start()] + transform(*arguments) + text[end:]
    return text


def expand(path):
    text = path.read_text()
    if path.name == "interface.tex":
        text = re.sub(r"\\ifdefined\\anonymizeinterface.*?\\else(.*?)\\fi", r"\1", text, flags=re.S)
    return re.sub(r"\\input\{([^}]+)\}", lambda m: expand(ROOT / (m[1] + ".tex")), text)


def label(text):
    match = re.search(r"\\label\{([^}]+)\}", text)
    return match[1] if match else None


def caption(text):
    match = re.search(r"\\caption\s*", text)
    return group(text, match.end())[0]


def prepare_latex():
    aux = (ROOT / "build/full.aux").read_text()
    labels = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux))
    citations = dict(re.findall(r"\\bibcite\{([^}]+)\}\{\{(\d+)\}", aux))
    full = expand(ROOT / "full.tex")
    full = re.sub(r"(?<!\\)%[^\n]*", "", full)
    title = group(full, full.index("{", full.index(r"\title[")))[0]
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", full, re.S)[1]
    keywords = group(full, full.index("{", full.index(r"\keywords")))[0]
    body = full.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    body = re.sub(r"\\(?:maketitle|FloatBarrier|balance)\b", "", body)
    for name in ("bibliographystyle", "bibliography", "nocite", "Description"):
        body = commands(body, name, 1, lambda _: "")
    body = commands(body, "paragraph", 1, lambda x: r"\textbf{" + x + "} ")

    def float_block(match):
        kind, content = match[1], match[2]
        key = label(content)
        number = labels[key]
        cap = caption(content)
        content = commands(content, "caption", 1, lambda _: "")
        content = commands(content, "label", 1, lambda _: "")
        content = content.replace(r"\centering", "")
        if key == "fig:bridging":
            content = r"\textbf{[TODO: directed-bridging diagram]}"
        if kind.startswith("figure"):
            content = re.sub(r"\\includegraphics\[[^]]*\]", r"\\includegraphics[width=6.4in]", content)
            return "\n" + content + "\n\\textbf{Figure " + number + ". " + cap + "}\n"
        return "\n\\textbf{Table " + number + ". " + cap + "}\n" + content + "\n"

    body = re.sub(r"\\begin\{(figure\*?|table\*?)\}(?:\[[^]]*\])?(.*?)\\end\{\1\}", float_block, body, flags=re.S)

    def algorithm(match):
        content = match[1]
        cap = caption(content)
        number = labels[label(content)]
        steps = []
        for line in content.splitlines():
            if line.startswith(r"\Require "):
                inputs = line.removeprefix(r"\Require ")
            elif line.startswith(r"\Ensure "):
                outputs = line.removeprefix(r"\Ensure ")
            elif line.startswith(r"\State "):
                steps.append(r"\item " + line.removeprefix(r"\State "))
        assert len(steps) == 7
        return "\n\\textbf{Algorithm " + number + ". " + cap + "}\n\n\\textbf{Input:} " + inputs + "\n\n\\textbf{Output:} " + outputs + "\n\\begin{enumerate}\n" + "\n".join(steps) + "\n\\end{enumerate}\n"

    body = re.sub(r"\\begin\{algorithm\}(?:\[[^]]*\])?(.*?)\\end\{algorithm\}", algorithm, body, flags=re.S)

    def equation(match):
        content = match[1]
        number = labels[label(content)]
        content = commands(content, "label", 1, lambda _: "").strip()
        return "\n\\[" + content + "\\]\n\\begin{flushright}(" + number + ")\\end{flushright}\n"

    body = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", equation, body, flags=re.S)
    body = commands(body, "ref", 1, lambda key: labels[key])
    body = commands(body, "cite", 1, lambda keys: "[" + ", ".join(citations[k.strip()] for k in keys.split(",")) + "]")
    body = commands(body, "label", 1, lambda _: "")
    # A normal tabular avoids format-specific float/caption handling in Word.
    body = re.sub(r"\\begin\{tabularx\}\{[^}]*\}", r"\\begin{tabular}", body)
    body = body.replace(r"\end{tabularx}", r"\end{tabular}")

    bbl = (ROOT / "build/full.bbl").read_text()
    entries = re.split(r"\\bibitem\[[^]]*\]%?\s*\{([^}]+)\}", bbl)[1:]
    assert len(entries) == 38
    references = []
    for key, text in zip(entries[0::2], entries[1::2]):
        text = text.split(r"\end{thebibliography}")[0]
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
        text = re.sub(r"\\urldef\\tempurl\s*", "", text)
        text = commands(text, "showURL", 1, lambda _: "")
        for name in ("bibinfo", "bibfield"):
            text = commands(text, name, 2, lambda _, value: value)
        for name in ("showarticletitle", "shownote"):
            text = commands(text, name, 1, lambda value: value)
        text = commands(text, "natexlab", 1, lambda _: "")
        text = text.replace(r"\newblock", " ")
        text = re.sub(r"\\showeprint\[[^]]*\]\{([^}]+)\}", r"arXiv:\1. ", text)
        references.append("[" + citations[key] + "] " + " ".join(text.split()))
    return "\\documentclass{article}\n\\newcommand{\\drafttodo}[1]{\\textbf{[TODO: #1]}}\n\\title{" + title + "}\n\\begin{document}\n\\maketitle\n\\section*{Abstract}\n" + abstract + "\n\\textbf{Keywords:} " + keywords + "\n" + body + "\n\\section*{References}\n" + "\n\n".join(references) + "\n\\end{document}\n"


def child(parent, tag, attributes=None):
    node = parent.find(f"{{{W}}}{tag}")
    if node is None:
        node = ET.SubElement(parent, f"{{{W}}}{tag}")
    if attributes:
        node.attrib.update({f"{{{W}}}{k}": str(v) for k, v in attributes.items()})
    return node


def reference_doc(path):
    data = subprocess.check_output(["pandoc", "--print-default-data-file", "reference.docx"])
    path.write_bytes(data)
    with ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    styles = ET.fromstring(parts["word/styles.xml"])
    defaults = child(styles, "docDefaults")
    props = child(child(defaults, "rPrDefault"), "rPr")
    child(props, "rFonts", {"ascii": "Times New Roman", "hAnsi": "Times New Roman"})
    child(props, "sz", {"val": 22})
    for style in styles.findall(f"{{{W}}}style"):
        sid = style.get(f"{{{W}}}styleId")
        if sid in {"Normal", "BodyText", "FirstParagraph", "Title", "Heading1", "Heading2", "Caption"}:
            rp = child(style, "rPr")
            child(rp, "rFonts", {"ascii": "Times New Roman", "hAnsi": "Times New Roman"})
            child(rp, "color", {"val": "000000"})
            child(rp, "sz", {"val": {"Title": 36, "Heading1": 28, "Heading2": 24, "Caption": 20}.get(sid, 22)})
            pp = child(style, "pPr")
            child(pp, "spacing", {"after": 100, "line": 276, "lineRule": "auto"})
            if sid in {"Normal", "BodyText", "FirstParagraph"}:
                child(pp, "jc", {"val": "both"})
    document = ET.fromstring(parts["word/document.xml"])
    section = document.find(f".//{{{W}}}sectPr")
    child(section, "pgSz", {"w": 11906, "h": 16838})
    child(section, "pgMar", {"top": 1134, "bottom": 1134, "left": 1134, "right": 1134, "header": 567, "footer": 567, "gutter": 0})
    parts["word/styles.xml"] = ET.tostring(styles, encoding="utf-8", xml_declaration=True)
    parts["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)


def format_document(path):
    """Apply page and table settings directly for consistent Word layout."""
    with ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    document = ET.fromstring(parts["word/document.xml"])
    section = document.find(f".//{{{W}}}sectPr")
    child(section, "pgSz", {"w": 11906, "h": 16838})
    child(section, "pgMar", {"top": 1134, "bottom": 1134, "left": 1134, "right": 1134, "header": 567, "footer": 567, "gutter": 0})
    widths = [[2313, 4145, 795, 795, 795, 795], [7400, 2238], [4500, 2569, 2569], [5200, 1500, 1700, 1238]]
    tables = document.findall(f".//{{{W}}}tbl")
    assert len(tables) == 4
    for index, (table, columns) in enumerate(zip(tables, widths)):
        child(child(table, "tblPr"), "tblW", {"w": sum(columns), "type": "dxa"})
        for node, width in zip(table.findall(f"{{{W}}}tblGrid/{{{W}}}gridCol"), columns):
            node.set(f"{{{W}}}w", str(width))
        for row_index, row in enumerate(table.findall(f"{{{W}}}tr")):
            child(child(row, "trPr"), "cantSplit")
            if row_index < (2 if index == 2 else 1):
                child(child(row, "trPr"), "tblHeader")
            for column, (cell, width) in enumerate(zip(row.findall(f"{{{W}}}tc"), columns)):
                child(child(cell, "tcPr"), "tcW", {"w": width, "type": "dxa"})
                for para in cell.findall(f".//{{{W}}}p"):
                    child(child(para, "pPr"), "jc", {"val": "left" if column == 0 or (index == 0 and column == 1) else "right"})
                    for run in para.findall(f"{{{W}}}r"):
                        child(child(run, "rPr"), "sz", {"val": 20})
    for para in document.findall(f".//{{{W}}}p"):
        text = "".join(para.itertext())
        if para.find(f".//{{{W}}}drawing") is not None:
            props = child(para, "pPr")
            child(props, "keepNext")
            child(props, "jc", {"val": "center"})
        elif text.startswith("Table "):
            child(child(para, "pPr"), "keepNext")
        elif re.match(r"^\[\d+\] ", text):
            child(child(para, "pPr"), "ind", {"left": 360, "hanging": 360})
    parts["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "AgentGS-English.docx")
    args = parser.parse_args()
    work = ROOT / "build/word"
    work.mkdir(parents=True, exist_ok=True)
    tex = work / "english-word.tex"
    tex.write_text(prepare_latex())
    reference = work / "reference.docx"
    reference_doc(reference)
    result = subprocess.check_output(["pandoc", str(tex), "-f", "latex", "-t", "json"])
    ast = json.loads(result)
    counts = [0, 0]
    for block in ast["blocks"]:
        if block["t"] == "Header":
            level, attributes, inlines = block["c"]
            if inlines != [{"t": "Str", "c": "Abstract"}] and inlines != [{"t": "Str", "c": "References"}]:
                counts[level - 1] += 1
                if level == 1:
                    counts[1] = 0
                prefix = ".".join(map(str, counts[:level]))
                block["c"][2] = [{"t": "Str", "c": prefix}, {"t": "Space"}] + inlines
    ast_path = work / "english-word.json"
    ast_path.write_text(json.dumps(ast))
    subprocess.run(["pandoc", str(ast_path), "-f", "json", "-t", "docx", "--standalone", "--resource-path=" + str(ROOT / "figures"), "--reference-doc=" + str(reference), "-o", str(args.output.resolve())], check=True)
    format_document(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
