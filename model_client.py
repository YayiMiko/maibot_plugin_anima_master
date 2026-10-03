"""Translate prompt-only calls to the public MaiBot LLM capability."""

from types import SimpleNamespace


class ModelClient:
    """Use isolated prompt messages, never the bot's conversational persona."""

    def __init__(self, ctx, config: dict):
        self.ctx = ctx
        self.config = config

    async def llm_generate(self, **kwargs):
        """Generate prompt text through Host-managed credentials and routing.

        Args:
            **kwargs: Upstream prompt arguments, excluding conversation history.

        Returns:
            A normalized completion for the unchanged prompt pipeline.

        Raises:
            RuntimeError: Host reports an unsuccessful or empty completion.
        """
        if kwargs.get("image_urls"):
            raise ValueError("Vision is not enabled in this implementation stage")
        if kwargs.get("thinking") or kwargs.get("reasoning_effort"):
            self.ctx.logger.info(
                "Anima reasoning parameters are controlled by Host model settings"
            )
        messages = []
        if kwargs.get("system_prompt"):
            messages.append({"role": "system", "content": kwargs["system_prompt"]})
        messages.append({"role": "user", "content": kwargs["prompt"]})
        result = await self.ctx.llm.generate(
            messages,
            task_name=self.config["prompt_model_task"],
            model_name=self.config.get("prompt_model_name", ""),
            max_tokens=kwargs.get("max_tokens"),
        )
        text = str(result.get("response", "") or "").strip()
        if not result.get("success") or not text:
            raise RuntimeError("MaiBot prompt model returned no usable completion")
        return SimpleNamespace(completion_text=text)
