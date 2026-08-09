"""Application-level exceptions, kept independent of any web framework so
core/ modules never need to know they're being called from Flask."""


class FinancialDocToolError(Exception):
    """Base class for all application-raised errors."""


class EmbeddingServiceError(FinancialDocToolError):
    """Raised when the embeddings provider cannot complete a request."""


class PdfProcessingError(FinancialDocToolError):
    """Raised when a PDF cannot be parsed or exceeds configured limits."""


class DocumentNotFoundError(FinancialDocToolError):
    """Raised when no document is associated with the current session."""
