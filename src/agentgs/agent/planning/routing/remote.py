from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentgs.agent.planning.routing.common import (
    _operation_text,
    _requested_months,
    _task_text,
    _unique,
)


def _remote_tools(text: str) -> list[str]:
    text = _task_text(text).casefold()
    operation_text = _operation_text(text)
    tools = ["RemoteSensingListFiles"]
    if any((token in text for token in ("ndsi", "snow index", "snow cover"))):
        tools.append("RemoteSensingBatchNDSI")
    if any((token in text for token in ("ndwi", "water index", "water body"))):
        tools.append("RemoteSensingNDWI")
    if any((token in text for token in ("nbr", "burn ratio", "wildfire"))):
        tools.append("RemoteSensingNBR")
    if "ndvi" in text and "ndsi" not in text:
        tools.append("RemoteSensingNDVI")
    if any((token in text for token in ("ndti", "turbidity"))):
        tools.append("RemoteSensingNDTI")
    if any(
        (token in text for token in ("cloud mask", "cloud-masked", "remove the clouds", "qa_pixel"))
    ):
        tools.append("RemoteSensingCloudMask")
    if any(
        (
            token in text
            for token in (
                "pixelwise average",
                "pixel-wise average",
                "average geotiff",
                "average map",
                "mean map",
                "composite",
            )
        )
    ):
        tools.append("RemoteSensingTiffAverage")
    if any(
        (
            token in text
            for token in (
                "mean",
                "average",
                "intensity",
                "rainfall",
                "temperature",
                "coverage",
                "vegetation",
                "biomass",
                "thermal",
                "land cover",
                "index",
            )
        )
    ):
        tools.append("RemoteSensingBatchImageMean")
    if any((token in text for token in ("average", "mean"))) and any(
        (
            token in operation_text
            for token in ("compare", "difference", "between", "each period", "during this period")
        )
    ):
        tools.append("RemoteSensingMean")
    if any(
        (
            token in text
            for token in (
                "hotspot proportion",
                "above the average",
                "above a threshold",
                "above the mean",
                "water body",
                "vegetation cover",
                "deep snow",
                "extreme loss area",
                "extreme loss proportion",
                "severe water loss",
            )
        )
    ) or re.search(
        "greater\\s+than\\s+[-+]?\\d+(?:\\.\\d+)?\\s+as\\s+(?:vegetation|water)|above\\s+(?:the\\s+)?(?:[a-z0-9_-]+\\s+)?mean",
        text,
    ):
        tools.append("RemoteSensingBatchHotspotPercentage")
    if any((token in text for token in ("binary raster", "extreme snow", "hotspot raster"))):
        tools.append("RemoteSensingBatchHotspotRaster")
    if "extreme snow and ice loss" in text and any(
        (token in text for token in ("proportion", "percentage"))
    ):
        tools.append("RemoteSensingExtremeSnowLossPercentage")
    if any((token in text for token in ("trend", "linear regression"))):
        tools.append("RemoteSensingLinearTrend")
    if "mann-kendall" in text:
        tools.append("RemoteSensingMannKendallTest")
    if re.search("\\bsen(?:['’]s)?\\s+slope\\b", text):
        tools.append("RemoteSensingSensSlope")
    if "kurtosis" in text:
        tools.append("RemoteSensingKurtosis")
    if "coefficient of variation" in text:
        tools.append("RemoteSensingCoefficientOfVariation")
    if any(
        (
            token in text
            for token in (
                "maximum",
                "maximum percentage",
                "peak",
                "greatest",
                "highest",
                "largest proportion",
                "most severe",
            )
        )
    ):
        tools.append("RemoteSensingMaxValueAndIndex")
    if any(
        (
            token in text
            for token in (
                "minimum",
                "coldest",
                "greatest loss",
                "largest loss",
                "maximum decrease",
                "minimum change",
            )
        )
    ):
        tools.append("RemoteSensingMinValueAndIndex")
    if any((token in text for token in ("kelvin", "°c", "celsius"))):
        tools.append("RemoteSensingKelvinToCelsius")
    if any(
        (
            token in text
            for token in (
                "percentage change",
                "percent change",
                "percentage increase",
                "percentage decrease",
                "percent loss",
                "vegetation loss",
            )
        )
    ):
        tools.append("RemoteSensingPercentageChange")
    if any(
        (
            token in operation_text
            for token in (
                "difference",
                "compare",
                "change between",
                "change",
                "loss",
                "decrease",
                "increase",
            )
        )
    ):
        tools.append("RemoteSensingDifference")
    if any((token in text for token in ("ratio", "divide"))):
        tools.append("RemoteSensingDivision")
    if any(
        (
            token in text
            for token in (
                "building volume",
                "built_volume",
                "non-residential",
                "energy saving",
                "per unit",
            )
        )
    ):
        tools.extend(
            ("RemoteSensingBatchImageSum", "RemoteSensingDivision", "RemoteSensingPercentageChange")
        )
    if "subtract" in text:
        tools.append("RemoteSensingSubtract")
    if any(
        (
            token in text
            for token in (
                "maximum percentage decrease",
                "greatest vegetation loss",
                "maximum vegetation loss",
            )
        )
    ):
        tools = [tool for tool in tools if tool != "RemoteSensingMaxValueAndIndex"]
        tools.append("RemoteSensingMinValueAndIndex")
    elif any(
        (token in text for token in ("maximum percentage increase", "greatest percentage increase"))
    ):
        tools = [tool for tool in tools if tool != "RemoteSensingMinValueAndIndex"]
        tools.append("RemoteSensingMaxValueAndIndex")
    if (
        "consecutive dates" in text
        or "between consecutive" in text
        or ("adjacent dates" in text and "change" in operation_text)
    ):
        tools.append("RemoteSensingTiffDifference")
        tools = [tool for tool in tools if tool != "RemoteSensingDifference"]
    if any(
        (
            token in text
            for token in (
                "percentage change",
                "percentage of change",
                "percentage decrease",
                "percentage increase",
            )
        )
    ) and (not any((token in text for token in ("difference between", "compare the difference")))):
        tools = [tool for tool in tools if tool != "RemoteSensingDifference"]
    if "energy saving" in text:
        tools = [tool for tool in tools if tool != "RemoteSensingDifference"]
    if "RemoteSensingSensSlope" in tools or "RemoteSensingMannKendallTest" in tools:
        tools = [tool for tool in tools if tool != "RemoteSensingLinearTrend"]
    threshold_definition = bool(
        "define" in text
        and re.search(
            "(?:decrease|increase|drop|loss|area|hotspot)[^.]{0,80}(?:greater|higher|lower|above|below|by)\\s+(?:than\\s+)?[-+]?\\d",
            text,
        )
    )
    if threshold_definition and (
        not any((token in operation_text for token in ("compare", "between", "difference between")))
    ):
        tools = [tool for tool in tools if tool != "RemoteSensingDifference"]
    if re.search(
        "(?:mean|average)\\s+(?:was|is|as)\\s+(?:set\\s+)?as\\s+(?:the\\s+)?threshold", text
    ):
        tools.append("RemoteSensingMean")
    if (
        any((token in text for token in ("mean map", "pixelwise average", "pixel-wise average")))
        and "building volume" in text
    ):
        tools = [tool for tool in tools if tool != "RemoteSensingBatchImageMean"]
        tools.insert(0, "RemoteSensingTiffAverage")
    for specialized in (
        "RemoteSensingBatchNDSI",
        "RemoteSensingNDWI",
        "RemoteSensingNBR",
        "RemoteSensingNDVI",
        "RemoteSensingNDTI",
    ):
        if specialized in tools and "RemoteSensingBatchImageMean" in tools:
            tools.remove(specialized)
            tools.insert(tools.index("RemoteSensingBatchImageMean"), specialized)
    if (
        "RemoteSensingMean" in tools
        and "RemoteSensingBatchHotspotPercentage" in tools
        and ("threshold" in text)
    ):
        tools.remove("RemoteSensingMean")
        tools.insert(tools.index("RemoteSensingBatchHotspotPercentage"), "RemoteSensingMean")
    return _unique(tools)


def _remote_asset_groups(resource: Any, text: str = "") -> list[list[Path]]:
    """Inspect available raster assets to infer independent cohorts.

    This uses only file metadata (names and dates), never benchmark answers or
    question identifiers.  It lets the planner instantiate repeated temporal
    operations without asking an LLM to enumerate a long directory listing.
    """
    if not isinstance(resource, str):
        return []
    name = Path(resource.replace("\\", "/")).name
    if not name:
        return []
    roots = (
        Path(resource),
        Path("data/benchmarks/earthbench_geotoolagent/assets") / name,
        Path("benchmark/data") / name,
    )
    directory = next((root for root in roots if root.is_dir()), None)
    if directory is None:
        return []
    files = [path for path in directory.iterdir() if path.is_file()]
    grouped: dict[str, list[Path]] = {}
    year_values: dict[Path, str] = {}
    month_values: dict[Path, int] = {}
    for path in sorted(files):
        date_match = re.search("(?:^|[_ -])(\\d{4})[-_](\\d{2})(?:[-_]|$)", path.stem)
        year_match = date_match or re.search("(?:^|[_ -])(\\d{4})(?:[-_]|$)", path.stem)
        if year_match:
            year_values[path] = year_match.group(1)
        if date_match:
            month_values[path] = int(date_match.group(2))
        stem = re.sub("[_ -]\\d{4}[-_]\\d{2}[-_]\\d{2}", "", path.stem, flags=re.IGNORECASE)
        stem = re.sub("[_ -]\\d{4}", "", stem, flags=re.IGNORECASE)
        stem = re.sub("[^a-z0-9\\u3400-\\u9fff]+", "_", stem.casefold()).strip("_")
        grouped.setdefault(stem or "default", []).append(path)
    requested_months = _requested_months(text)
    month_comparison = any(
        (
            phrase in str(text or "").casefold()
            for phrase in (
                "which month",
                "compare the month",
                "compare monthly",
                "difference between the two",
                "each month",
            )
        )
    )
    if len(requested_months) > 1 and month_values and month_comparison:
        by_month: dict[int, list[Path]] = {}
        requested = set(requested_months)
        for path in files:
            month = month_values.get(path)
            if month in requested:
                by_month.setdefault(month, []).append(path)
        if len(by_month) > 1:
            return [by_month[month] for month in requested_months if month in by_month]
    requested_quarters = list(dict.fromkeys(re.findall("\\bq([1-4])\\b", text, re.IGNORECASE)))
    if len(requested_quarters) > 1 and month_values:
        by_quarter: dict[str, list[Path]] = {}
        requested = set(requested_quarters)
        for path in files:
            month = month_values.get(path)
            if month is None:
                continue
            quarter = str((month - 1) // 3 + 1)
            if quarter in requested:
                by_quarter.setdefault(f"Q{quarter}", []).append(path)
        if len(by_quarter) > 1:
            return [
                by_quarter[f"Q{quarter}"]
                for quarter in requested_quarters
                if f"Q{quarter}" in by_quarter
            ]
    years = sorted(set(year_values.values()))
    if len(years) > 1 and any(
        (
            token in text
            for token in ("period", "between", "compare", "difference", "change between")
        )
    ):
        by_year: dict[str, list[Path]] = {}
        for path in files:
            by_year.setdefault(year_values.get(path, "unknown"), []).append(path)
        grouped = by_year
    return [items for items in grouped.values() if items]


def _remote_plan(text: str, resource: Any) -> list[str]:
    text = _task_text(text).casefold()
    operation_text = _operation_text(text)
    tools = _remote_tools(text)
    groups = _remote_asset_groups(resource, text)
    if not groups:
        return tools
    flat_assets = [path for group in groups for path in group]
    if (
        any((token in text for token in ("maxfrp", "fire intensity")))
        and any(
            (token in text for token in ("significant increase", "increase threshold", "+20 mw"))
        )
        and ("map" in text)
    ):
        return [
            "RemoteSensingListFiles",
            "RemoteSensingTiffAverage",
            "RemoteSensingTiffAverage",
            "RemoteSensingSubtract",
            "RemoteSensingFireIncreaseMap",
            "RemoteSensingCountAboveThreshold",
        ]
    if "built_volume" in text and "ratio" in text and ("trend" in text):
        return [
            "RemoteSensingListFiles",
            "RemoteSensingBatchImageMean",
            "RemoteSensingBatchImageMean",
            "RemoteSensingDivision",
            "RemoteSensingLinearTrend",
        ]
    derived_generators = {
        "RemoteSensingNDVI": "ndvi",
        "RemoteSensingBatchNDSI": "ndsi",
        "RemoteSensingNDWI": "ndwi",
        "RemoteSensingNBR": "nbr",
        "RemoteSensingNDTI": "ndti",
    }
    for generator, marker in derived_generators.items():
        if (
            generator in tools
            and flat_assets
            and all((marker in path.stem.casefold() for path in flat_assets))
        ):
            tools = [tool for tool in tools if tool != generator]
    active_generator = next((tool for tool in derived_generators if tool in tools), None)
    if active_generator:
        years = sorted(
            {
                match.group(1)
                for path in flat_assets
                for match in [re.search("(?:^|[_ -])((?:19|20)\\d{2})(?:[-_]|$)", path.stem)]
                if match
            }
        )
        comparison = any(
            (
                token in operation_text
                for token in (
                    "compare",
                    "between",
                    "difference",
                    "change",
                    "increase",
                    "decrease",
                    "over this period",
                    "across the two years",
                )
            )
        )
        cohort_count = len(years) if comparison and len(years) > 1 else 1
        if (
            active_generator == "RemoteSensingBatchNDSI"
            and cohort_count > 1
            and ("snow cover" in text)
            and any((token in operation_text for token in ("change", "variation")))
            and ("extreme" not in text)
            and ("coefficient of variation" not in text)
            and ("RemoteSensingPercentageChange" not in tools)
        ):
            tools.append("RemoteSensingPercentageChange")
        explicit_linear_trend = any(
            (
                phrase in text
                for phrase in ("linear trend analysis", "linear regression", "least-squares")
            )
        )
        if cohort_count > 1 and "RemoteSensingDifference" in tools and (not explicit_linear_trend):
            tools = [tool for tool in tools if tool != "RemoteSensingLinearTrend"]
        terminal_series_tools = {
            "RemoteSensingCoefficientOfVariation",
            "RemoteSensingKurtosis",
            "RemoteSensingLinearTrend",
            "RemoteSensingSensSlope",
            "RemoteSensingMannKendallTest",
            "RemoteSensingMaxValueAndIndex",
            "RemoteSensingMinValueAndIndex",
            "RemoteSensingArgmax",
        }
        if (
            any((tool in tools for tool in terminal_series_tools))
            and "RemoteSensingBatchHotspotPercentage" not in tools
            and ("RemoteSensingBatchImageMean" not in tools)
        ):
            tools.append("RemoteSensingBatchImageMean")
        if "RemoteSensingCoefficientOfVariation" in tools:
            tools = [tool for tool in tools if tool != "RemoteSensingMean"]
        plan = ["RemoteSensingListFiles"]
        if "RemoteSensingCloudMask" in tools:
            plan.append("RemoteSensingCloudMask")
        plan.append(active_generator)
        if "RemoteSensingTiffAverage" in tools:
            plan.extend(["RemoteSensingTiffAverage"] * cohort_count)
        if "RemoteSensingBatchImageMean" in tools:
            plan.extend(["RemoteSensingBatchImageMean"] * cohort_count)
        needs_period_scalar = (
            cohort_count > 1
            and "RemoteSensingBatchImageMean" in tools
            and any(
                (
                    tool in tools
                    for tool in ("RemoteSensingDifference", "RemoteSensingPercentageChange")
                )
            )
            and ("RemoteSensingCoefficientOfVariation" not in tools)
        )
        if "RemoteSensingMean" in tools or needs_period_scalar:
            plan.extend(["RemoteSensingMean"] * cohort_count)
        elif any(
            (tool in tools for tool in ("RemoteSensingSensSlope", "RemoteSensingMannKendallTest"))
        ) and any(
            (
                phrase in text
                for phrase in ("average of the daily", "average value of the daily", "mean nbr")
            )
        ):
            plan.append("RemoteSensingMean")
        if (
            "RemoteSensingBatchHotspotRaster" in tools
            and "RemoteSensingExtremeSnowLossPercentage" in tools
        ):
            plan = [
                tool
                for tool in plan
                if tool not in {"RemoteSensingBatchImageMean", "RemoteSensingMean"}
            ]
        repeated_reducers = {
            "RemoteSensingCoefficientOfVariation",
            "RemoteSensingBatchHotspotPercentage",
            "RemoteSensingBatchHotspotRaster",
            "RemoteSensingExtremeSnowLossPercentage",
        }
        for tool in tools:
            if tool in {
                "RemoteSensingListFiles",
                "RemoteSensingCloudMask",
                active_generator,
                "RemoteSensingTiffAverage",
                "RemoteSensingBatchImageMean",
                "RemoteSensingMean",
                "RemoteSensingLinearTrend",
                "RemoteSensingSensSlope",
                "RemoteSensingMannKendallTest",
                "RemoteSensingDifference",
                "RemoteSensingPercentageChange",
                "RemoteSensingMaxValueAndIndex",
                "RemoteSensingMinValueAndIndex",
                "RemoteSensingArgmax",
            }:
                continue
            if tool in repeated_reducers:
                plan.extend([tool] * cohort_count)
            elif tool not in plan:
                plan.append(tool)
        for tool in (
            "RemoteSensingLinearTrend",
            "RemoteSensingSensSlope",
            "RemoteSensingMannKendallTest",
            "RemoteSensingDifference",
            "RemoteSensingPercentageChange",
            "RemoteSensingMaxValueAndIndex",
            "RemoteSensingMinValueAndIndex",
            "RemoteSensingArgmax",
        ):
            if tool in tools:
                plan.append(tool)
        return plan
    is_thermal = any((token in text for token in ("temperature", "thermal", "b10", "band 10")))
    has_extremum = any(
        (
            tool in tools
            for tool in ("RemoteSensingMinValueAndIndex", "RemoteSensingMaxValueAndIndex")
        )
    )
    if is_thermal and has_extremum:
        tools = [
            tool
            for tool in tools
            if tool not in {"RemoteSensingMean", "RemoteSensingKelvinToCelsius"}
        ]
    elif is_thermal and "RemoteSensingDifference" in tools:
        tools = [tool for tool in tools if tool != "RemoteSensingKelvinToCelsius"]
    elif (
        is_thermal
        and "RemoteSensingKelvinToCelsius" in tools
        and ("kelvin" not in text)
        and (len(flat_assets) == 1)
    ):
        tools = [tool for tool in tools if tool != "RemoteSensingKelvinToCelsius"]
    elif (
        is_thermal
        and "RemoteSensingKelvinToCelsius" in tools
        and ("RemoteSensingBatchImageMean" in tools)
        and (len(groups) == 1)
        and (len(groups[0]) > 1)
        and ("RemoteSensingMean" not in tools)
    ):
        tools.insert(tools.index("RemoteSensingKelvinToCelsius"), "RemoteSensingMean")
    if (
        "RemoteSensingBatchImageMean" in tools
        and "RemoteSensingPercentageChange" in tools
        and ("consecutive dates" in text or "between consecutive" in text)
    ):
        count = max(1, sum((max(0, len(group) - 1) for group in groups)))
        tail = [
            tool
            for tool in tools
            if tool
            not in {
                "RemoteSensingListFiles",
                "RemoteSensingBatchImageMean",
                "RemoteSensingPercentageChange",
                "RemoteSensingDifference",
            }
        ]
        return (
            ["RemoteSensingListFiles", "RemoteSensingBatchImageMean"]
            + ["RemoteSensingPercentageChange"] * count
            + tail
        )
    if "RemoteSensingTiffDifference" in tools:
        terminal = [
            tool
            for tool in tools
            if tool
            not in {
                "RemoteSensingListFiles",
                "RemoteSensingBatchImageMean",
                "RemoteSensingDifference",
                "RemoteSensingTiffDifference",
            }
        ]
        return ["RemoteSensingListFiles", "RemoteSensingTiffDifference"] + terminal
    if (
        len(groups) > 1
        and "RemoteSensingBatchImageMean" in tools
        and ("RemoteSensingDifference" in tools)
        and ("RemoteSensingMean" not in tools)
        and (len(_requested_months(text)) > 1)
    ):
        tools.append("RemoteSensingMean")
    if (
        len(groups) > 1
        and any(
            (tool in tools for tool in ("RemoteSensingBatchImageMean", "RemoteSensingTiffAverage"))
        )
        and ("RemoteSensingMean" in tools or "RemoteSensingTiffAverage" in tools)
    ):
        count = len(groups)
        producers: list[str] = []
        if "RemoteSensingTiffAverage" in tools:
            producers.extend(["RemoteSensingTiffAverage"] * count)
        if "RemoteSensingBatchImageMean" in tools:
            producers.extend(["RemoteSensingBatchImageMean"] * count)
        reducers = (
            ["RemoteSensingMean"] * count
            if "RemoteSensingMean" in tools and "RemoteSensingTiffAverage" not in tools
            else []
        )
        terminal = [
            tool
            for tool in tools
            if tool
            not in {
                "RemoteSensingListFiles",
                "RemoteSensingBatchImageMean",
                "RemoteSensingTiffAverage",
                "RemoteSensingMean",
            }
        ]
        if "RemoteSensingBatchImageSum" in terminal:
            terminal = [tool for tool in terminal if tool != "RemoteSensingBatchImageSum"] + [
                "RemoteSensingBatchImageSum"
            ] * (2 * count)
            order = {
                "RemoteSensingBatchImageSum": 10,
                "RemoteSensingDivision": 20,
                "RemoteSensingPercentageChange": 30,
            }
            terminal.sort(key=lambda tool: order.get(tool, 15))
        if "RemoteSensingDivision" in terminal:
            terminal = [tool for tool in terminal if tool != "RemoteSensingDivision"] + [
                "RemoteSensingDivision"
            ] * count
            order = {
                "RemoteSensingBatchImageSum": 10,
                "RemoteSensingDivision": 20,
                "RemoteSensingPercentageChange": 30,
            }
            terminal.sort(key=lambda tool: order.get(tool, 15))
        per_group_reducers = {
            "RemoteSensingBatchHotspotPercentage",
            "RemoteSensingBatchHotspotRaster",
        }
        for reducer in per_group_reducers:
            if reducer in terminal:
                terminal = [tool for tool in terminal if tool != reducer] + [reducer] * count
        terminal.sort(
            key=lambda tool: {
                "RemoteSensingBatchImageSum": 10,
                "RemoteSensingBatchHotspotPercentage": 12,
                "RemoteSensingBatchHotspotRaster": 12,
                "RemoteSensingDivision": 20,
                "RemoteSensingPercentageChange": 30,
                "RemoteSensingDifference": 40,
            }.get(tool, 15)
        )
        return ["RemoteSensingListFiles"] + producers + reducers + terminal
    return tools
