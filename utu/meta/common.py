from dataclasses import dataclass

from ..agents.common import DataClassWithStreamEvents


@dataclass
class GeneratorTaskRecorder(DataClassWithStreamEvents):
    requirements: str = None
    selected_tools: dict[str, list[str]] = None
    instructions: str = None
    name: str = None
