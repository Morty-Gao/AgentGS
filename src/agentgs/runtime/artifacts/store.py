import hashlib
import json
import mimetypes
from pathlib import Path
from uuid import uuid4

from agentgs.config.paths import within
from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import ArtifactRef


class LocalArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    async def write_json(self, name, data, source):
        return await invoke(self._write_json, name, data, source)

    def _write_json(self, name, data, source):
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

    async def register(self, path, source):
        return await invoke(self._register, path, source)

    def _register(self, path, source):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Only produced files in the task artifact directory can be registered")
        with path.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        return ArtifactRef(
            id=uuid4().hex,
            path=str(path.relative_to(self.root)),
            media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            sha256=fingerprint,
            bytes=path.stat().st_size,
            source=source,
        )
