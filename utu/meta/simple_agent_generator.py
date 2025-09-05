"""
- [ ] integrate into UI
    use `interaction_toolkit.set_ask_function()` to config ask function
- [ ] bug fix: `task_recorder.stream_events()` cannot stop
    ref: `OrchestraAgent`
"""

import asyncio
import json
from collections import defaultdict

from agents import RunResultStreaming, StopAtTools, trace

from ..agents import SimpleAgent
from ..tools import UserInteractionToolkit, get_toolkits_map
from ..utils import DIR_ROOT, AgentsUtils, get_jinja_env, get_logger
from .common import GeneratorTaskRecorder

logger = get_logger(__name__)

TOOL_SELECTION_TEMPLATE = """<available_tools>
{available_tools}
</available_tools>
<requirement>
{requirement}
</requirement>"""

CONFIG_TEMPLATE = """
# @package _global_
defaults:
  - /model/base@model
{toolkits_includes}
  - _self_

toolkits:
{toolkits_configs}

agent:
  name: {agent_name}
  instructions: |
{instructions}
"""


def add_indented_lines(lines: str | list[str], indent: int = 2) -> str:
    if isinstance(lines, str):
        lines = lines.split("\n")
    return "\n".join(" " * indent + line for line in lines)


class SimpleAgentGenerator:
    def __init__(self):
        self.jinja_env = get_jinja_env(DIR_ROOT / "utu/prompts/meta")
        self.output_dir = DIR_ROOT / "configs/agents/generated"
        self.output_dir.mkdir(exist_ok=True)

        self.mode = "local"  # local | webui
        self._initialized = False

    async def build(self):
        if self._initialized:
            return
        self.interaction_toolkit = UserInteractionToolkit()
        self.agent_1 = SimpleAgent(
            name="clarification_agent",
            instructions=self.jinja_env.get_template("requirements_clarification.j2").render(),
            tools=await self.interaction_toolkit.get_tools_in_agents(),
            tool_use_behavior=StopAtTools(stop_at_tool_names=["final_answer"]),
        )
        self.agent_2 = SimpleAgent(
            name="tool_selection_agent",
            instructions=self.jinja_env.get_template("tools_selection.j2").render(),
        )
        self.agent_3 = SimpleAgent(
            name="instructions_generation_agent",
            instructions=self.jinja_env.get_template("instructions_generation.j2").render(),
        )
        self.agent_4 = SimpleAgent(
            name="name_generation_agent",
            instructions=self.jinja_env.get_template("name_generation.j2").render(),
        )
        self._initialized = True

    async def run(self):
        await self.build()
        with trace("simple_agent_generator"):
            task_recorder = GeneratorTaskRecorder()
            await self.step1(task_recorder)
            await self.step2(task_recorder)
            await self.step3(task_recorder)
            await self.step4(task_recorder)
            ofn = self.format_config(task_recorder)
            print(f"Config saved to {ofn}")

    def run_streamed(self):
        with trace("simple_agent_generator"):
            task_recorder = GeneratorTaskRecorder()
            task_recorder._run_impl_task = asyncio.create_task(self._start_streaming(task_recorder))
        return task_recorder

    async def _start_streaming(self, task_recorder: GeneratorTaskRecorder):
        await self.build()
        await self.step1(task_recorder)
        await self.step2(task_recorder)
        await self.step3(task_recorder)
        await self.step4(task_recorder)
        ofn = self.format_config(task_recorder)
        task_recorder._is_complete = True
        print(f"Config saved to {ofn}")
        print(f"task_recorder: {task_recorder}")
        return ofn

    def format_config(self, task_recorder: GeneratorTaskRecorder) -> str:
        toolkits_includes = []
        toolkits_configs = []
        for toolkit_name, tool_names in task_recorder.selected_tools.items():
            toolkits_includes.append(f"- /tools/{toolkit_name}@toolkits.{toolkit_name}")
            toolkits_configs.append(f"{toolkit_name}: {json.dumps({'activated_tools': tool_names})}")
        config = CONFIG_TEMPLATE.format(
            agent_name=task_recorder.name,
            instructions=add_indented_lines(task_recorder.instructions, 4),
            toolkits_includes=add_indented_lines(toolkits_includes, 2),
            toolkits_configs=add_indented_lines(toolkits_configs, 2),
        )
        ofn = self.output_dir / f"{task_recorder.name}.yaml"
        ofn.write_text(config)
        return ofn

    async def step1(self, task_recorder: GeneratorTaskRecorder) -> None:
        user_input = await self.interaction_toolkit.ask_user("Please enter your requirements:")
        async with self.agent_1 as agent:
            result = agent.run_streamed(user_input)
            await self._process_streamed(result, task_recorder)
            task_recorder.requirements = result.final_output

    async def step2(self, task_recorder: GeneratorTaskRecorder) -> None:
        """Select useful tools from available toolkits. Return: {toolkit_name: [tool_name, ...]}"""
        available_toolkits = ["search", "document", "image", "audio", "bash", "tabular"]
        toolkits_map = get_toolkits_map(names=available_toolkits)
        tools_descs = []
        tool_to_toolkit_name = {}
        for toolkit_name, toolkit in toolkits_map.items():
            tools = await toolkit.get_tools_in_agents()
            tools_descs.extend(f"- {tool.name}: {tool.description}" for tool in tools)
            tool_to_toolkit_name.update({tool.name: toolkit_name for tool in tools})
        tools_str = "\n".join(tools_descs)
        query = TOOL_SELECTION_TEMPLATE.format(
            available_tools=tools_str,
            requirement=task_recorder.requirements,
        )
        async with self.agent_2 as agent:
            result = agent.run_streamed(query)
            await self._process_streamed(result, task_recorder)
            selected_tools = result.final_output
            selected_tool_names = json.loads(selected_tools)
        selected_tools = defaultdict(list)
        for tool_name in selected_tool_names:
            selected_tools[tool_to_toolkit_name[tool_name]].append(tool_name)
        task_recorder.selected_tools = selected_tools

    async def step3(self, task_recorder: GeneratorTaskRecorder) -> None:
        """Generate instructions for the agent."""
        async with self.agent_3 as agent:
            result = agent.run_streamed(task_recorder.requirements)
            await self._process_streamed(result, task_recorder)
            task_recorder.instructions = result.final_output

    async def step4(self, task_recorder: GeneratorTaskRecorder) -> None:
        """Generate instructions for the agent."""
        async with self.agent_4 as agent:
            result = agent.run_streamed(task_recorder.requirements)
            await self._process_streamed(result, task_recorder)
            name = result.final_output
            if len(name) > 50 or " " in name:
                logger.warning(f"Generated name is too long or contains spaces: {name}")
                name = name[:50].replace(" ", "_")
            task_recorder.name = name

    async def _process_streamed(self, run_result_streaming: RunResultStreaming, task_recorder: GeneratorTaskRecorder):
        if self.mode == "local":
            await AgentsUtils.print_stream_events(run_result_streaming.stream_events())
        else:
            async for event in run_result_streaming.stream_events():
                task_recorder._event_queue.put_nowait(event)
