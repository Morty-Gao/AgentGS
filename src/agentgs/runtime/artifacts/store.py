import hashlib
import json
from pathlib import Path
from uuid import uuid4

from agentgs.config.paths import within
from agentgs.contracts.models import ArtifactRef


class LocalArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name, data, source):
        target = within(self.root, name)
        if target.exists():
            raise ValueError("Artifact path already exists; producer call ids must be unique")
        target.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2).encode()
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(target)
        return ArtifactRef(
            id=uuid4().hex,
            path=str(target.relative_to(self.root)),
            media_type="application/json",
            sha256=hashlib.sha256(content).hexdigest(),
            bytes=len(content),
            source=source,
        )
