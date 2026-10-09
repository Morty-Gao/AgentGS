"""Shared contracts at the agent/runtime boundary.

``models`` defines serialized data and dependency protocols; ``tool_inputs``
declares invocation schemas; ``catalog`` reads the unchanged tool metadata.
``arguments`` and ``resources`` provide shared normalization and path access.
``remote_sensing`` owns raster metadata, validation and step preparation.

These modules do not depend on the agent, scheduler, web layer or evaluation.
"""
