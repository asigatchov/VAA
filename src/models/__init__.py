"""Models and CLIs for volleyball action detection."""

__all__ = ["ACTION_CLASS_NAMES", "EXPORT_CLASS_NAMES", "VballActionDetector"]


def __getattr__(name: str):
    if name in {"ACTION_CLASS_NAMES", "EXPORT_CLASS_NAMES"}:
        from .constants import ACTION_CLASS_NAMES, EXPORT_CLASS_NAMES

        return {"ACTION_CLASS_NAMES": ACTION_CLASS_NAMES, "EXPORT_CLASS_NAMES": EXPORT_CLASS_NAMES}[name]
    if name == "VballActionDetector":
        from .action_detector import ACTION_CLASS_NAMES, EXPORT_CLASS_NAMES, VballActionDetector

        values = {
            "ACTION_CLASS_NAMES": ACTION_CLASS_NAMES,
            "EXPORT_CLASS_NAMES": EXPORT_CLASS_NAMES,
            "VballActionDetector": VballActionDetector,
        }
        return values[name]
    raise AttributeError(name)
