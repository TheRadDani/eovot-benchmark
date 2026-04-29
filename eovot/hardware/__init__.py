"""Hardware sub-package — platform detection and TDP configuration."""

from .detector import HardwarePlatform, detect_platform, get_recommended_tdp

__all__ = ["HardwarePlatform", "detect_platform", "get_recommended_tdp"]
