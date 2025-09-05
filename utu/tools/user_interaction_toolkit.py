from collections.abc import Callable
from typing import Any

from ..config import ToolkitConfig
from ..utils import PrintUtils
from .base import AsyncBaseToolkit


class UserInteractionToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None):
        super().__init__(config)
        self.ask_function = PrintUtils.async_print_input

    def set_ask_function(self, ask_function: Callable[[str], str]):
        self.ask_function = ask_function

    async def ask_user(self, question: str) -> str:
        """Asks for user's input on a specific question

        Args:
            question (str): The question to ask.
        """
        if self.ask_function:
            return await self.ask_function(question)

    async def final_answer(self, answer: Any) -> str:
        """Provides a final answer to the given problem.

        Args:
            answer (any): The answer to ask.
        """
        return answer

    async def get_tools_map(self) -> dict[str, Callable]:
        return {
            "ask_user": self.ask_user,
            "final_answer": self.final_answer,
        }
