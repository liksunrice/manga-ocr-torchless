import time
import sys
import threading
from pathlib import Path
from PIL import Image, ImageGrab
import pyperclip
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .ocr import MangaOcr
import logging

logger = logging.getLogger(__name__)

class ClipboardHandler:
    def __init__(self, mocr: MangaOcr):
        self.mocr = mocr
        self.last_img = None

    def run(self):
        logger.info("Reading from clipboard...")
        try:
            while True:
                img = ImageGrab.grabclipboard()
                if isinstance(img, Image.Image) and self._is_new_image(img):
                    self.last_img = img
                    self._process_image(img)
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Clipboard reading stopped.")

    def _is_new_image(self, img: Image.Image) -> bool:
        if self.last_img is None:
            return True
        if img.size != self.last_img.size:
            return True
        return img.tobytes() != self.last_img.tobytes()
        
    def _process_image(self, img: Image.Image):
        try:
            text = self.mocr(img.convert("RGB"))
            print(text)
            pyperclip.copy(text)
        except Exception as e:
            logger.error(f"Error processing image: {e}")


class DirectoryHandler(FileSystemEventHandler):
    def __init__(self, mocr: MangaOcr, read_delay: float = 0.5):
        self.mocr = mocr
        self.read_delay = read_delay

    def on_created(self, event):
        if not event.is_directory:
            # Add a slight delay to ensure file is completely written before reading
            time.sleep(self.read_delay)
            self._process_image(event.src_path)

    def _process_image(self, path: str):
        try:
            # We attempt to open it just to see if it's a valid image
            # watchdog might trigger on non-image files depending on the dir
            img = Image.open(path).convert("RGB")
            text = self.mocr(img)
            print(text)
            pyperclip.copy(text)
        except IOError:
            # Not an image or could not open, ignore it
            pass
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")


def run_directory(mocr: MangaOcr, path: str, delay: float = 0.5):
    target_dir = Path(path)
    if not target_dir.is_dir():
        logger.error(f"Error: Directory '{path}' not found.")
        sys.exit(1)
        
    logger.info(f"Watching directory: {path}")
    event_handler = DirectoryHandler(mocr, delay)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Directory monitoring stopped.")
    observer.join()
