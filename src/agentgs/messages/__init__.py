"""Messages are the source of truth; execution information is derived on demand.

    conversation = Conversation.from_openai([{"role": "user", "content": "Hello"}])
    conversation.append(Message(role="assistant", content="Hi", reasoning="..."))
    conversation.save("messages.json")
    wire_messages = conversation.to_request_messages()

JSONL uses to_jsonl()/from_jsonl(); execution_events() and request_messages()
derive tool execution information and original model inputs from the same log.
Request export retains current-turn reasoning and omits earlier-turn reasoning.
"""

from agentgs.messages.conversation import Conversation
from agentgs.messages.models import Function, FunctionCall, LogEntry, Message, RuntimeEvent

__all__ = ["Conversation", "Function", "FunctionCall", "LogEntry", "Message", "RuntimeEvent"]
