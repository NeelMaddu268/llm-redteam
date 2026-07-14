"""redteam — an LLM prompt-injection / jailbreak evaluation harness.

The pipeline is: load payloads -> run them against targets -> classify each
response -> aggregate into a report. Every stage is defined behind a small
interface so pieces are swappable (add a target, add a classifier, add a
model provider) without touching the rest.

Nothing in the core pipeline requires an API key: the default targets and the
default judge all run against a local Ollama model.
"""

__version__ = "0.1.0"
