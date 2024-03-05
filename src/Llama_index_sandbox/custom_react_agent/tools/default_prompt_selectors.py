"""Default prompt selectors."""
from llama_index.legacy.prompts import SelectorPromptTemplate
from llama_index.legacy.prompts.default_prompts import (
    DEFAULT_REFINE_PROMPT,
    DEFAULT_REFINE_TABLE_CONTEXT_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
    DEFAULT_TREE_SUMMARIZE_PROMPT,
)
from llama_index.legacy.prompts.utils import is_chat_model

from src.Llama_index_sandbox.custom_react_agent.tools.query_engine_prompts import CHAT_TEXT_QA_PROMPT, CHAT_TREE_SUMMARIZE_PROMPT, CHAT_REFINE_PROMPT, CHAT_REFINE_TABLE_CONTEXT_PROMPT

DEFAULT_TEXT_QA_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_TEXT_QA_PROMPT,
    conditionals=[(is_chat_model, CHAT_TEXT_QA_PROMPT)],
)

DEFAULT_TREE_SUMMARIZE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_TREE_SUMMARIZE_PROMPT,
    conditionals=[(is_chat_model, CHAT_TREE_SUMMARIZE_PROMPT)],
)

DEFAULT_REFINE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_REFINE_PROMPT,
    conditionals=[(is_chat_model, CHAT_REFINE_PROMPT)],
)

DEFAULT_REFINE_TABLE_CONTEXT_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_REFINE_TABLE_CONTEXT_PROMPT,
    conditionals=[(is_chat_model, CHAT_REFINE_TABLE_CONTEXT_PROMPT)],
)
