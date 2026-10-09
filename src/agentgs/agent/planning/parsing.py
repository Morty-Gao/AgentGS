from __future__ import annotations

import re
from typing import Optional, Tuple


def extract_answer_parts(answer_text: str) -> Tuple[Optional[str], Optional[str]]:
    is_completed = extract_regex_group("IsCompleted:\\s*([^\\n]+)", answer_text)
    final_answer = extract_regex_group("FinalAnswer:\\s*(.+)", answer_text, re.DOTALL)
    return (is_completed, final_answer)


def extract_regex_group(pattern: str, text: str, flags=0) -> Optional[str]:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None
