"""Custom extractor plugin system for pdf-ingest."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from .chunk import Chunk

T = TypeVar("T")


@dataclass
class ExtractorResult(Generic[T]):
    """Result from an extractor plugin."""
    items: list[T]
    errors: list[str]
    metadata: dict[str, Any]


class ExtractorPlugin(ABC, Generic[T]):
    """Base class for custom extractor plugins.

    To create a custom extractor:
    1. Subclass ExtractorPlugin
    2. Implement extract() method
    3. Register with @register_extractor or PluginRegistry.register()

    Example:
        @register_extractor("custom")
        class MyExtractor(ExtractorPlugin[MyType]):
            def extract(self, chunks, chat):
                # Your extraction logic
                return ExtractorResult(items=[], errors=[], metadata={})
    """

    name: str = "base"
    description: str = "Base extractor plugin"
    version: str = "1.0.0"
    supported_formats: list[str] = ["json", "markdown"]

    @abstractmethod
    def extract(
        self,
        chunks: list[Chunk],
        chat: Any,  # RAGChat instance
    ) -> ExtractorResult[T]:
        """Extract structured data from chunks.

        Args:
            chunks: List of text chunks to extract from
            chat: RAGChat instance for LLM calls

        Returns:
            ExtractorResult with extracted items, errors, and metadata
        """

    def to_dict(self, item: T) -> dict[str, Any]:
        """Convert item to dictionary for JSON export."""
        if hasattr(item, "to_dict"):
            return item.to_dict()
        if hasattr(item, "__dict__"):
            return item.__dict__
        return {"value": str(item)}

    def to_markdown(self, item: T) -> str:
        """Convert item to markdown for display."""
        if hasattr(item, "to_markdown"):
            return item.to_markdown()
        return str(item)


class PluginRegistry:
    """Registry for extractor plugins."""

    _plugins: dict[str, type[ExtractorPlugin]] = {}
    _instances: dict[str, ExtractorPlugin] = {}

    @classmethod
    def register(cls, name: str, plugin_class: type[ExtractorPlugin]) -> None:
        """Register a plugin class.

        Args:
            name: Unique name for the plugin
            plugin_class: ExtractorPlugin subclass
        """
        if not issubclass(plugin_class, ExtractorPlugin):
            raise TypeError(f"Plugin must be subclass of ExtractorPlugin, got {plugin_class}")
        cls._plugins[name] = plugin_class
        cls._instances.pop(name, None)

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a plugin.

        Args:
            name: Plugin name to remove

        Returns:
            True if plugin was removed, False if not found
        """
        if name in cls._plugins:
            del cls._plugins[name]
            cls._instances.pop(name, None)
            return True
        return False

    @classmethod
    def get(cls, name: str) -> ExtractorPlugin | None:
        """Get plugin instance by name (singleton)."""
        if name not in cls._plugins:
            return None
        if name not in cls._instances:
            cls._instances[name] = cls._plugins[name]()
        return cls._instances[name]

    @classmethod
    def list_plugins(cls) -> dict[str, dict[str, Any]]:
        """List all registered plugins with metadata."""
        return {
            name: {
                "name": plugin.name,
                "description": plugin.description,
                "version": plugin.version,
                "supported_formats": plugin.supported_formats,
            }
            for name, plugin in cls._plugins.items()
        }

    @classmethod
    def clear(cls) -> None:
        """Clear all registered plugins."""
        cls._plugins.clear()
        cls._instances.clear()


def register_extractor(name: str):
    """Decorator to register an extractor plugin.

    Usage:
        @register_extractor("my_extractor")
        class MyExtractor(ExtractorPlugin[MyType]):
            ...
    """
    def decorator(cls: type[ExtractorPlugin]) -> type[ExtractorPlugin]:
        PluginRegistry.register(name, cls)
        return cls
    return decorator


def get_extractor(name: str) -> ExtractorPlugin | None:
    """Get extractor plugin instance by name."""
    return PluginRegistry.get(name)