"""scanner.hardware.display — TFT SPI display driver for RB-TFT3.2-V2.

Renders text and status messages on the 3.2" SPI TFT display connected to
the Raspberry Pi.  Uses the luma.lcd library which supports ST7789/ILI9341
controllers common in RB-TFT3.2-V2 modules.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Display dimensions for RB-TFT3.2-V2
_DISPLAY_WIDTH = 320
_DISPLAY_HEIGHT = 240
_LINE_HEIGHT_PX = 20
_FONT_SIZE = 16

# State → human-readable descriptions
_STATE_LABELS: dict[str, str] = {
    "IDLE": "Ready",
    "CALIBRATING": "Calibrating...",
    "SCANNING": "Scanning...",
    "PROCESSING": "Processing...",
    "EXPORTING": "Exporting...",
    "COMPLETE": "Scan complete!",
    "ERROR": "ERROR",
}


class Display:
    """RB-TFT3.2-V2 SPI TFT display driver.

    Uses luma.lcd with PIL for text rendering.  The display is cleared on
    initialisation and supports writing text to individual lines.

    Args:
        config: Interface configuration dict.  Optionally:
            - display_spi_port: SPI port (default 0)
            - display_spi_device: SPI device/CS (default 0)
            - display_dc_pin: Data/Command GPIO pin (default 25)
            - display_rst_pin: Reset GPIO pin (default 24)
            - display_backlight_pin: Backlight GPIO pin (default 18)

    Raises:
        HardwareError: if luma.lcd is unavailable or display init fails.
    """

    def __init__(self, config: dict) -> None:
        from scanner.hardware import HardwareError

        try:
            from luma.lcd.device import ili9341  # type: ignore[import]
            from luma.core.interface.serial import spi  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError(
                "luma.lcd not available — install it with: pip install luma.lcd"
            ) from exc

        try:
            from PIL import ImageFont, ImageDraw, Image  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError("Pillow not available") from exc

        spi_port: int = int(config.get("display_spi_port", 0))
        spi_device: int = int(config.get("display_spi_device", 0))
        dc_pin: int = int(config.get("display_dc_pin", 25))
        rst_pin: int = int(config.get("display_rst_pin", 24))
        bl_pin: int = int(config.get("display_backlight_pin", 18))

        try:
            serial = spi(port=spi_port, device=spi_device, gpio_DC=dc_pin, gpio_RST=rst_pin)
            self._device = ili9341(serial, width=_DISPLAY_WIDTH, height=_DISPLAY_HEIGHT)
            self._Image = Image
            self._ImageDraw = ImageDraw
            try:
                self._font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _FONT_SIZE
                )
            except OSError:
                self._font = ImageFont.load_default()

            self._lines: list[str] = [""] * (_DISPLAY_HEIGHT // _LINE_HEIGHT_PX)
            self._refresh()
            logger.info(
                "Display initialised (RB-TFT3.2-V2, %dx%d)", _DISPLAY_WIDTH, _DISPLAY_HEIGHT
            )
        except Exception as exc:
            raise HardwareError(f"Display init failed: {exc}") from exc

    def display_text(self, text: str, line: int = 0) -> None:
        """Write *text* on the display at *line*.

        Args:
            text: String to display.
            line: 0-based line number.
        """
        line = max(0, min(line, len(self._lines) - 1))
        self._lines[line] = text
        self._refresh()
        logger.debug("Display [line %d]: %s", line, text)

    def display_status(self, state: str) -> None:
        """Show the scanner state on the first line.

        Args:
            state: State string key from the machine d'états.
        """
        label = _STATE_LABELS.get(state, state)
        self.display_text(label, line=0)

    def _refresh(self) -> None:
        """Redraw the entire display buffer from *self._lines*."""
        from luma.core.render import canvas  # type: ignore[import]

        try:
            with canvas(self._device) as draw:
                for idx, text in enumerate(self._lines):
                    if text:
                        y = idx * _LINE_HEIGHT_PX
                        draw.text((4, y), text, fill="white", font=self._font)
        except Exception as exc:
            logger.warning("Display refresh failed: %s", exc)

    def close(self) -> None:
        """Clear the display and release resources."""
        try:
            self._device.cleanup()
            logger.info("Display released")
        except Exception as exc:
            logger.warning("Error closing display: %s", exc)
