from __future__ import annotations

import ast
import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_tool_path(response_text: str) -> List[str]:
    try:
        match = re.search("Plan Chain:\\s*(\\[[\\s\\S]*?\\])", response_text)
        list_str = match.group(1) if match else None
        if list_str is None:
            generic_match = re.search("\\[[\\s\\S]*?\\]", response_text)
            list_str = generic_match.group(0) if generic_match else None
        if list_str is not None:
            cleaned_tools_string = re.sub(",\\s*//.*?(?=[\\],])", "", list_str)
            try:
                parsed = ast.literal_eval(cleaned_tools_string)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except (SyntaxError, ValueError):
                quoted_items = re.findall("['\\\"]([^'\\\"]+)['\\\"]", cleaned_tools_string)
                if quoted_items:
                    return [item.strip() for item in quoted_items if item.strip()]
        logger.warning("No tool path found in response.")
        return []
    except (SyntaxError, ValueError, re.error) as e:
        logger.error(f"Error extracting tool path: {e}")
        return []


def extract_answer_parts(answer_text: str) -> Tuple[Optional[str], Optional[str]]:
    is_completed = extract_regex_group("IsCompleted:\\s*([^\\n]+)", answer_text)
    final_answer = extract_regex_group("FinalAnswer:\\s*(.+)", answer_text, re.DOTALL)
    return (is_completed, final_answer)


def extract_regex_group(pattern: str, text: str, flags=0) -> Optional[str]:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None
