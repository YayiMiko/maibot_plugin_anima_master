from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

WEB_SEARCH_KEYWORDS = (
    "联网",
    "搜索",
    "搜一下",
    "查一下",
    "参考资料",
    "官方图",
    "设定图",
    "资料",
)
DEEP_THINKING_KEYWORDS = (
    "深度思考",
    "认真想",
    "仔细想",
    "严格还原",
    "高度还原",
    "不要跑偏",
    "核心特征",
)

_SPACES_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PromptResearchPlan:
    """Research switches selected for one prompt.

    Args:
        use_web_search: Whether web search should run.
        use_deep_thinking: Whether the LLM should use deep thinking options.
        search_reason: Keyword that triggered web search.
        thinking_reason: Keyword that triggered deep thinking.
    """

    use_web_search: bool
    use_deep_thinking: bool
    search_reason: str
    thinking_reason: str


class PromptResearcher:
    """Handle optional prompt research and strategy keyword checks."""

    def __init__(
        self,
        *,
        context: Any,
        logger: Any,
        get_bool: Callable[[str, bool], bool],
        get_int: Callable[[str, int], int],
        get_str: Callable[[str, str], str],
    ):
        """Store dependencies for prompt research.

        Args:
            context: AstrBot plugin context used for per-chat config lookup.
            logger: Logger compatible with AstrBot logger methods.
            get_bool: Config boolean accessor.
            get_int: Config integer accessor.
            get_str: Config string accessor.
        """
        self.context = context
        self.logger = logger
        self._bool = get_bool
        self._int = get_int
        self._str = get_str

    def keyword_reason(self, prompt: str, keywords: tuple[str, ...]) -> str:
        """Return the first keyword that appears in the prompt.

        Args:
            prompt: User prompt.
            keywords: Candidate keywords.

        Returns:
            Matched keyword, or an empty string.
        """
        text = str(prompt or "").lower()
        for keyword in keywords:
            if keyword.lower() in text:
                return keyword
        return ""

    def plan(self, prompt: str) -> PromptResearchPlan:
        """Build the optional research plan for a prompt.

        Args:
            prompt: User prompt.

        Returns:
            Research plan including trigger reasons.
        """
        search_reason = self.keyword_reason(prompt, WEB_SEARCH_KEYWORDS)
        thinking_reason = self.keyword_reason(prompt, DEEP_THINKING_KEYWORDS)
        return PromptResearchPlan(
            use_web_search=bool(search_reason)
            and self._bool("prompt_builder_web_search_enabled", True),
            use_deep_thinking=bool(thinking_reason)
            and self._bool("prompt_builder_deep_thinking_enabled", True),
            search_reason=search_reason,
            thinking_reason=thinking_reason,
        )

    def search_query(self, prompt: str) -> str:
        """Build the web search query for a prompt.

        Args:
            prompt: User prompt.

        Returns:
            Query text.
        """
        template = self._str(
            "prompt_builder_search_query_template",
            "{prompt} anime game character outfit pose official art visual design",
        ).strip()
        if "{prompt}" in template:
            return template.replace("{prompt}", prompt).strip()
        return f"{prompt} {template}".strip()

    async def search_context(
        self,
        event: Any,
        prompt: str,
        *,
        search_query_prompt: str | None = None,
    ) -> str:
        """Fetch web search context for a prompt.

        Args:
            event: AstrBot message event for per-chat config lookup.
            prompt: User prompt used for logs and summary text.
            search_query_prompt: Optional prompt override used only for the search query.

        Returns:
            Search context text, or an empty string when unavailable.
        """
        if not self._bool("prompt_builder_web_search_enabled", True):
            return ""
        key = self._str("research_api_key", "").strip()
        if not key:
            self.logger.warning("Anima web research skipped: no search credential")
            return ""
        try:
            payload = {
                "query": self.search_query(search_query_prompt or prompt),
                "max_results": max(
                    1, min(self._int("prompt_builder_search_max_results", 5), 8)
                ),
                "search_depth": self._str("prompt_builder_search_depth", "advanced"),
                "topic": "general",
            }
            if payload["search_depth"] not in {"basic", "advanced"}:
                payload["search_depth"] = "advanced"
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
                response.raise_for_status()
                results = response.json().get("results", [])
        except Exception as exc:
            self.logger.warning("[comfyui_agent] prompt web search failed: %s", exc)
            return ""

        lines = [f"用户主题：{prompt}", "搜索结果："]
        for idx, result in enumerate(results, 1):
            title = str(result.get("title", "") or "").strip()
            url = str(result.get("url", "") or "").strip()
            snippet = str(result.get("content", "") or "").strip()
            snippet = _SPACES_RE.sub(" ", snippet)[:500]
            if not title and not snippet:
                continue
            line = f"{idx}. {title}"
            if snippet:
                line += f"\n摘要：{snippet}"
            if url:
                line += f"\nURL：{url}"
            lines.append(line)
        context = "\n".join(lines).strip()
        if len(lines) <= 2:
            return ""
        self.logger.info(
            "[comfyui_agent] prompt web search ok results=%s chars=%s",
            len(lines) - 2,
            len(context),
        )
        return context
