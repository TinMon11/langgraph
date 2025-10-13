# # https://python.langchain.com/docs/concepts/callbacks/
# LangChain provides a callback system that allows you to hook into the various stages of your LLM application. This is useful for logging, monitoring, streaming, and other tasks.

# You can subscribe to these events by using the callbacks argument available throughout the API. This argument is a list of handler objects, which are expected to implement one or more of the methods described below in more detail.

# chain.invoke({"number": 25}, {"callbacks": [handler]})

from typing import Any, List

from langchain.schema import LLMResult
from langchain_core.callbacks import BaseCallbackHandler


class AgentCallbackHandler(BaseCallbackHandler):

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print(" *** LLM Started ***")
        print(" **** Prompt: ", prompts[0])
        print(" *********************************** ")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM starts running."""
        print(" *** LLM Ended ***")
        print(" **** Response: ", response.generations[0][0].text)
        print(" ----------------------------------- ")
