import asyncio

from utu.meta import SimpleAgentGenerator
from utu.utils import AgentsUtils

# asyncio.run(SimpleAgentGenerator().run())


async def main():
    generator = SimpleAgentGenerator()
    task_recorder = generator.run_streamed()
    await AgentsUtils.print_stream_events(task_recorder.stream_events())


if __name__ == "__main__":
    asyncio.run(main())
