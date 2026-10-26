class AnimationError(Exception):
    """Base class for all errors raised by this package."""


class NotebookError(Exception):
    """Raised when there is an error related to the notebook environment."""


class MathLatexError(AnimationError):
    """Raised when there is an error related to MathLaTeX rendering."""


class NotebookSidecarError(NotebookError):
    """Raised when there is an error related to the notebook sidecar."""


class NotebookPanelError(NotebookError):
    """Raised when there is an error related to the notebook panel."""
