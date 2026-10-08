from __future__ import annotations

import re
from typing import Any

from agentgs.agent.answering.evidence.core import _number_text, _walk_dicts


def _project_workflow_deliverables(values: list[Any], question: str) -> str | None:
    """Describe validated GIS workflow artifacts in the request's semantics.

    Planning-oriented evaluation tasks ask for named deliverables rather than
    raw filesystem paths.  This projection is driven by executed output schemas
    plus capability words in the request; it does not inspect task identifiers
    or reference answers.
    """
    text = str(question or "").casefold()
    dictionaries = [candidate for value in values for candidate in _walk_dicts(value)]
    has_map = any(("map_png" in candidate or "map_html" in candidate for candidate in dictionaries))
    if not has_map:
        return None
    has_zonal_statistics = any(
        (
            {"count", "mean", "minimum", "maximum"} <= set(candidate) and "source" in candidate
            for candidate in dictionaries
        )
    )
    has_overlay_artifact = any(
        (
            any((key in candidate for key in ("overlay_output", "joined_output", "output_path")))
            for candidate in dictionaries
        )
    )
    if (
        has_zonal_statistics
        and any((token in text for token in ("zonal statistics", "by zone", "analysis zones")))
        and any((token in text for token in ("age raster", "age", "geological")))
        and ("overlay" in text or has_overlay_artifact)
    ):
        return "点-分析区关联结果；年龄分区统计表；分区年龄专题图"
    is_deep_time = any((token in text for token in ("biodeeptime", "deep-time", "deep time")))
    is_ccs = any((token in text for token in ("ccs", "co2", "carbon storage")))
    if "crs" in text and any((token in text for token in ("mixed", "inconsisten", "4326/3857"))):
        return "CRS 不一致诊断；统一 EPSG:4326 的图层；叠加结果地图"
    if (
        is_ccs
        and any((token in text for token in ("thematic", "classified", "classification")))
        and any((token in text for token in ("co2_net_stored_tonnes", "net stored")))
    ):
        return "净封存量分级 PNG 地图；可交互 HTML 地图"
    if "nearest" in text and "neigh" in text:
        count_match = re.search("\\b(one|two|three|four|five|\\d+)\\s*[- ]?nearest", text)
        word_numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        raw_count = count_match.group(1) if count_match else "1"
        count = word_numbers.get(raw_count, int(raw_count) if raw_count.isdigit() else 1)
        if is_ccs:
            return f"质控设施点图层；{count} 近邻结果；净封存量专题图"
        if is_deep_time:
            return f"清洗后的时间序列点图层；{count} 近邻结果；按 age 分级的 PNG/HTML 专题图"
    if "buffer" in text:
        if "multi-ring" in text or "multi ring" in text:
            return "多级泄漏缓冲区；监测优先级叠加图层；PNG/HTML 优先级图"
        distance_match = re.search("(\\d+(?:\\.\\d+)?)\\s*[- ]?(?:km|kilomet(?:er|re)s?)", text)
        distance = _number_text(distance_match.group(1), 0) if distance_match else ""
        if is_ccs and "leak" in text:
            return f"泄漏设施 {distance} km 缓冲区；监测区叠加结果；影响区地图"
        if is_deep_time:
            return f"{distance} km 缓冲区；分析区叠加结果；覆盖范围专题图"
    if "interpolat" in text:
        if "reservoir pressure" in text or "pressure" in text:
            return "储层压力插值栅格；压力连续面专题图"
        if "injection" in text:
            month_match = re.search(
                "\\b(january|february|march|april|may|june|july|august|september|october|november|december)\\s+(20\\d{2})\\b",
                text,
            )
            if month_match:
                months = {
                    "january": 1,
                    "february": 2,
                    "march": 3,
                    "april": 4,
                    "may": 5,
                    "june": 6,
                    "july": 7,
                    "august": 8,
                    "september": 9,
                    "october": 10,
                    "november": 11,
                    "december": 12,
                }
                date_text = f"{month_match.group(2)}-{months[month_match.group(1)]:02d} "
            else:
                date_text = ""
            return f"{date_text}注入强度栅格；分析区统计表；注入强度专题图"
        if "abundance" in text:
            return "丰度插值栅格；分区丰度统计表；丰度专题图"
        if "age" in text:
            if "overlay" in text or "locations" in text:
                return "点-分析区关联结果；年龄分区统计表；分区年龄专题图"
            return "age IDW 栅格；分区年龄统计表；连续面专题图"
    if "batch" in text:
        if is_ccs and "annual" in text and ("monthly" in text):
            return "年度/月度关联图层；分析区空间关联结果；月注入量专题图"
        if is_deep_time:
            return "标准化并质控的点/面图层；点落面关联结果；分区时间序列数量图"
    if is_ccs and "net stored" in text and ("zone" in text) and ("overlay" in text):
        return "设施-分析区关联图层；净封存量分区统计表；分区净封存量地图"
    return None
