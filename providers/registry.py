"""
Provider registry.

Maps a provider name to its adapter class. ``speech_revolutions`` is our own
system (the "Zephyr" system in the comparison chart), accessed via the
official Python SDK; the rest are the ten external systems being benchmarked
against it. Each external adapter needs its own API key(s) via environment
variables (see the README capability matrix).
"""

from __future__ import annotations

from .assemblyai import AssemblyAIProvider
from .azure import AzureProvider
from .base import Provider
from .cohere import CohereProvider
from .deepgram import DeepgramProvider
from .elevenlabs import ElevenLabsProvider
from .gladia import GladiaProvider
from .grok import GrokProvider
from .mistral import MistralProvider
from .openai import OpenAIProvider
from .qwen import QwenProvider
from .soniox import SonioxProvider
from .speech_revolutions import SpeechRevolutionsProvider

_PROVIDERS: dict[str, type[Provider]] = {
    "speech_revolutions": SpeechRevolutionsProvider,  # our own system ("Zephyr")
    "assemblyai": AssemblyAIProvider,
    "deepgram": DeepgramProvider,
    "openai": OpenAIProvider,
    "elevenlabs": ElevenLabsProvider,
    "gladia": GladiaProvider,
    "mistral": MistralProvider,          # Voxtral Mini
    "soniox": SonioxProvider,
    "cohere": CohereProvider,
    "grok": GrokProvider,                # xAI
    "qwen": QwenProvider,                # Qwen3-ASR (needs URL-hosted audio)
    "azure": AzureProvider,              # Azure Batch (needs URL-hosted audio)
}

# All target providers from the comparison chart are now implemented.
PLANNED_PROVIDERS: list[str] = []


def get_provider(name: str, **kwargs) -> Provider:
    if name not in _PROVIDERS:
        raise KeyError(
            f"Unknown provider '{name}'. Available: {available_providers()}"
        )
    return _PROVIDERS[name](**kwargs)


def available_providers() -> list[str]:
    return list(_PROVIDERS)


def register(name: str, cls: type[Provider]) -> None:
    _PROVIDERS[name] = cls
