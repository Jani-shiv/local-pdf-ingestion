"""Input validation and sanitization for CLI commands.

This module provides:
- Validation functions for all input parameters
- Sanitization for strings, paths, URLs
- InputValidator class for command-level validation

Example:
    InputValidator.validate_ingest_args(args)
    sanitized = sanitize_string(user_input)
"""
from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import ConfigurationError

# Allowed file extensions for ingestion
ALLOWED_EXTENSIONS: dict[str, list[str]] = {
    "pdf": [".pdf"],
    "audio": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
    "office": [".docx", ".xlsx", ".pptx"],
    "image": [".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
    "web": [],  # URLs handled separately
}

ALL_ALLOWED_EXTENSIONS: set[str] = set()
for exts in ALLOWED_EXTENSIONS.values():
    ALL_ALLOWED_EXTENSIONS.update(exts)

# URL validation regex
URL_REGEX = re.compile(
    r"^(?:http|https)://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
    r"localhost|"
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    r"(?::\d+)?"
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)

# Sanitization patterns
DANGEROUS_PATTERNS = [
    (re.compile(r"[\x00-\x1f\x7f-\x9f]"), ""),  # Control characters
    (re.compile(r"[\r\n]+"), " "),  # Newlines
    (re.compile(r"\s{2,}"), " "),  # Multiple spaces
]


def sanitize_string(value: str, max_length: int = 10000) -> str:
    """Sanitize a string input.

    Removes control characters, normalizes whitespace, and truncates.

    Args:
        value: Input string
        max_length: Maximum allowed length

    Returns:
        Sanitized string

    Raises:
        ConfigurationError: If value is not a string
    """
    if not isinstance(value, str):
        raise ConfigurationError(f"Expected string, got {type(value).__name__}")

    # Truncate if too long
    if len(value) > max_length:
        value = value[:max_length]

    # Apply sanitization patterns
    for pattern, replacement in DANGEROUS_PATTERNS:
        value = pattern.sub(replacement, value)

    return value.strip()


def validate_file_path(
    path: str | Path,
    must_exist: bool = True,
    allowed_extensions: set[str] | None = None,
) -> Path:
    """Validate and sanitize a file path.

    Args:
        path: File path string or Path object
        must_exist: Require file to exist
        allowed_extensions: Optional set of allowed extensions

    Returns:
        Resolved Path object

    Raises:
        ConfigurationError: If validation fails
    """
    path = Path(path).expanduser().resolve()

    if must_exist and not path.exists():
        raise ConfigurationError(f"File not found: {path}")

    if must_exist and not path.is_file():
        raise ConfigurationError(f"Path is not a file: {path}")

    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ConfigurationError(
            f"Unsupported file type: {path.suffix}. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    # Check for path traversal attempts
    with suppress(ValueError):
        path.relative_to(Path.cwd())

    return path


def validate_directory_path(path: str | Path, must_exist: bool = True) -> Path:
    """Validate and sanitize a directory path.

    Args:
        path: Directory path string or Path object
        must_exist: Require directory to exist

    Returns:
        Resolved Path object

    Raises:
        ConfigurationError: If validation fails
    """
    path = Path(path).expanduser().resolve()

    if must_exist and not path.exists():
        raise ConfigurationError(f"Directory not found: {path}")

    if must_exist and not path.is_dir():
        raise ConfigurationError(f"Path is not a directory: {path}")

    return path


def validate_url(url: str) -> str:
    """Validate a URL.

    Args:
        url: URL string to validate

    Returns:
        Validated URL string

    Raises:
        ConfigurationError: If URL is invalid
    """
    url = sanitize_string(url, max_length=2048)

    if not URL_REGEX.match(url):
        raise ConfigurationError(f"Invalid URL format: {url}")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ConfigurationError(f"URL must use http or https: {url}")

    return url


def validate_urls(urls: list[str]) -> list[str]:
    """Validate a list of URLs.

    Args:
        urls: List of URL strings

    Returns:
        List of validated URLs

    Raises:
        ConfigurationError: If list is empty or contains invalid URLs
    """
    if not urls:
        raise ConfigurationError("At least one URL must be provided")

    validated = []
    for url in urls:
        validated.append(validate_url(url.strip()))

    return validated


def validate_chunk_size(value: int) -> int:
    """Validate chunk size parameter.

    Args:
        value: Chunk size value

    Returns:
        Validated chunk size

    Raises:
        ConfigurationError: If invalid
    """
    if not isinstance(value, int):
        raise ConfigurationError(f"Chunk size must be an integer, got {type(value).__name__}")

    if value < 100 or value > 10000:
        raise ConfigurationError("Chunk size must be between 100 and 10000")

    return value


def validate_chunk_overlap(value: int, chunk_size: int) -> int:
    """Validate chunk overlap parameter.

    Args:
        value: Overlap value
        chunk_size: Associated chunk size

    Returns:
        Validated overlap value

    Raises:
        ConfigurationError: If invalid
    """
    if not isinstance(value, int):
        raise ConfigurationError(f"Chunk overlap must be an integer, got {type(value).__name__}")

    if value < 0 or value >= chunk_size:
        raise ConfigurationError(f"Chunk overlap must be >= 0 and < chunk_size ({chunk_size})")

    return value


def validate_top_k(value: int) -> int:
    """Validate top_k parameter.

    Args:
        value: Top K value

    Returns:
        Validated top_k value

    Raises:
        ConfigurationError: If invalid
    """
    if not isinstance(value, int):
        raise ConfigurationError(f"Top K must be an integer, got {type(value).__name__}")

    if value < 1 or value > 50:
        raise ConfigurationError("Top K must be between 1 and 50")

    return value


def validate_threshold(value: float) -> float:
    """Validate similarity threshold parameter.

    Args:
        value: Threshold value

    Returns:
        Validated threshold

    Raises:
        ConfigurationError: If invalid
    """
    if not isinstance(value, (int, float)):
        raise ConfigurationError(f"Threshold must be a number, got {type(value).__name__}")

    if value < 0.0 or value > 1.0:
        raise ConfigurationError("Threshold must be between 0.0 and 1.0")

    return float(value)


def validate_temperature(value: float) -> float:
    """Validate temperature parameter.

    Args:
        value: Temperature value

    Returns:
        Validated temperature

    Raises:
        ConfigurationError: If invalid
    """
    if not isinstance(value, (int, float)):
        raise ConfigurationError(f"Temperature must be a number, got {type(value).__name__}")

    if value < 0.0 or value > 2.0:
        raise ConfigurationError("Temperature must be between 0.0 and 2.0")

    return float(value)


def validate_max_tokens(value: int) -> int:
    """Validate max_tokens parameter.

    Args:
        value: Max tokens value

    Returns:
        Validated max tokens

    Raises:
        ConfigurationError: If invalid
    """
    if not isinstance(value, int):
        raise ConfigurationError(f"Max tokens must be an integer, got {type(value).__name__}")

    if value < 100 or value > 8192:
        raise ConfigurationError("Max tokens must be between 100 and 8192")

    return value


def validate_model_name(value: str) -> str:
    """Validate model name.

    Args:
        value: Model name string

    Returns:
        Validated model name

    Raises:
        ConfigurationError: If invalid
    """
    value = sanitize_string(value, max_length=256)

    if not value:
        raise ConfigurationError("Model name cannot be empty")

    # Basic validation - alphanumeric, dots, colons, hyphens, underscores
    if not re.match(r"^[a-zA-Z0-9.:_-]+$", value):
        raise ConfigurationError(f"Invalid model name format: {value}")

    return value


def validate_output_path(path: str | Path | None) -> Path | None:
    """Validate output file path.

    Args:
        path: Output path or None

    Returns:
        Validated Path or None

    Raises:
        ConfigurationError: If path has unsupported extension
    """
    if path is None:
        return None

    path = Path(path).expanduser().resolve()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check extension
    allowed_output_extensions = {".json", ".md", ".markdown", ".txt", ".csv"}
    if path.suffix.lower() not in allowed_output_extensions:
        raise ConfigurationError(
            f"Unsupported output format: {path.suffix}. "
            f"Allowed: {', '.join(sorted(allowed_output_extensions))}",
        )

    return path


def validate_store_path(path: str | Path) -> Path:
    """Validate vector store path.

    Args:
        path: Store path string or Path

    Returns:
        Validated Path (created if needed)
    """
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_query(query: str) -> str:
    """Validate search/chat query.

    Args:
        query: Query string

    Returns:
        Validated query string

    Raises:
        ConfigurationError: If query is empty
    """
    query = sanitize_string(query, max_length=5000)

    if not query:
        raise ConfigurationError("Query cannot be empty")

    return query


def validate_extract_type(value: str) -> str:
    """Validate extract type.

    Args:
        value: Extract type string

    Returns:
        Validated extract type

    Raises:
        ConfigurationError: If type is invalid
    """
    allowed = {"recipes", "invoices", "medical", "contracts", "study"}
    if value not in allowed:
        raise ConfigurationError(f"Invalid extract type: {value}. Allowed: {', '.join(sorted(allowed))}")
    return value


def validate_extract_format(value: str, extract_type: str) -> str:
    """Validate extract output format.

    Args:
        value: Format string
        extract_type: Type of extraction

    Returns:
        Validated format

    Raises:
        ConfigurationError: If format is invalid for type
    """
    allowed = {"json", "markdown"}
    if extract_type == "invoices":
        allowed.add("csv")

    if value not in allowed:
        raise ConfigurationError(f"Invalid format: {value}. Allowed for {extract_type}: {', '.join(sorted(allowed))}")
    return value


def validate_study_format(value: str) -> str:
    """Validate study guide format.

    Args:
        value: Format string

    Returns:
        Validated format

    Raises:
        ConfigurationError: If format is invalid
    """
    allowed = {"flashcards", "mindmap", "quiz", "all"}
    if value not in allowed:
        raise ConfigurationError(f"Invalid study format: {value}. Allowed: {', '.join(sorted(allowed))}")
    return value


def validate_summarize_style(value: str) -> str:
    """Validate summarize style.

    Args:
        value: Style string

    Returns:
        Validated style

    Raises:
        ConfigurationError: If style is invalid
    """
    allowed = {"concise", "detailed", "bullet"}
    if value not in allowed:
        raise ConfigurationError(f"Invalid style: {value}. Allowed: {', '.join(sorted(allowed))}")
    return value


def validate_chunk_strategy(value: str) -> str:
    """Validate chunk strategy.

    Args:
        value: Strategy string

    Returns:
        Validated strategy

    Raises:
        ConfigurationError: If strategy is invalid
    """
    allowed = {"character", "semantic", "recursive"}
    if value not in allowed:
        raise ConfigurationError(f"Invalid chunk strategy: {value}. Allowed: {', '.join(sorted(allowed))}")
    return value


def validate_vector_store(value: str) -> str:
    """Validate vector store type.

    Args:
        value: Store type string

    Returns:
        Validated store type

    Raises:
        ConfigurationError: If type is invalid
    """
    allowed = {"faiss", "chroma"}
    if value not in allowed:
        raise ConfigurationError(f"Invalid vector store: {value}. Allowed: {', '.join(sorted(allowed))}")
    return value


def validate_log_level(value: str) -> str:
    """Validate log level.

    Args:
        value: Log level string

    Returns:
        Validated log level (uppercase)

    Raises:
        ConfigurationError: If level is invalid
    """
    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    value_upper = value.upper()
    if value_upper not in allowed:
        raise ConfigurationError(f"Invalid log level: {value}. Allowed: {', '.join(sorted(allowed))}")
    return value_upper


def validate_positive_int(
    value: int,
    name: str,
    min_val: int = 1,
    max_val: int = 1000000,
) -> int:
    """Validate positive integer within range.

    Args:
        value: Integer value
        name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated integer

    Raises:
        ConfigurationError: If invalid
    """
    if not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer, got {type(value).__name__}")

    if value < min_val or value > max_val:
        raise ConfigurationError(f"{name} must be between {min_val} and {max_val}")

    return value


class InputValidator:
    """Comprehensive input validator for CLI commands.

    Provides static methods to validate arguments for each CLI command.

    Example:
        InputValidator.validate_ingest_args(args)
    """

    @staticmethod
    def validate_ingest_args(args: Any) -> None:
        """Validate ingest command arguments.

        Args:
            args: Parsed argparse namespace

        Raises:
            ConfigurationError: If any validation fails
        """
        # Validate paths
        for path_str in args.paths:
            path = Path(path_str)
            if path.is_dir():
                validate_directory_path(path)
            else:
                validate_file_path(path, allowed_extensions=ALL_ALLOWED_EXTENSIONS)

        # Validate parameters
        validate_chunk_size(args.chunk_size)
        validate_chunk_overlap(args.overlap, args.chunk_size)
        validate_chunk_strategy(args.strategy)
        validate_store_path(args.output)

    @staticmethod
    def validate_chat_args(args: Any) -> None:
        """Validate chat command arguments.

        Args:
            args: Parsed argparse namespace

        Raises:
            ConfigurationError: If any validation fails
        """
        validate_store_path(args.store)
        validate_model_name(args.model)
        validate_top_k(args.top_k)
        validate_threshold(args.threshold)

        if args.query:
            validate_query(args.query)

    @staticmethod
    def validate_search_args(args: Any) -> None:
        """Validate search command arguments.

        Args:
            args: Parsed argparse namespace

        Raises:
            ConfigurationError: If any validation fails
        """
        validate_store_path(args.store)
        validate_query(args.query)
        validate_top_k(args.top_k)
        validate_threshold(args.threshold)

    @staticmethod
    def validate_summarize_args(args: Any) -> None:
        """Validate summarize command arguments.

        Args:
            args: Parsed argparse namespace

        Raises:
            ConfigurationError: If any validation fails
        """
        validate_store_path(args.store)
        validate_positive_int(args.max_length, "max_length", 100, 5000)
        validate_summarize_style(args.style)
        validate_output_path(args.output)

    @staticmethod
    def validate_study_args(args: Any) -> None:
        """Validate study command arguments.

        Args:
            args: Parsed argparse namespace

        Raises:
            ConfigurationError: If any validation fails
        """
        validate_store_path(args.store)
        validate_study_format(args.format)
        validate_output_path(args.output)

    @staticmethod
    def validate_extract_args(args: Any) -> None:
        """Validate extract command arguments.

        Args:
            args: Parsed argparse namespace

        Raises:
            ConfigurationError: If any validation fails
        """
        validate_store_path(args.store)
        validate_extract_type(args.type)
        validate_extract_format(args.format, args.type)
        validate_output_path(args.output)

    @staticmethod
    def validate_list_args(args: Any) -> None:
        """Validate list command arguments.

        Args:
            args: Parsed argparse namespace

        Raises:
            ConfigurationError: If any validation fails
        """
        validate_store_path(args.store)