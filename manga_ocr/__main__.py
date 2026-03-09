import argparse
import sys
import logging
from PIL import Image
from pathlib import Path

from .ocr import MangaOcr

def main():
    parser = argparse.ArgumentParser(description="Manga OCR (Torchless/ONNX)")
    parser.add_argument('image_path', type=str, nargs='?', help='Path to an image file to OCR')
    parser.add_argument('-m', '--model', type=str, default='mayocream/manga-ocr-onnx',
                        help='HuggingFace repo ID or local path to ONNX model')
    parser.add_argument('-b', '--background', action='store_true', help='Run in background, reading from clipboard (requires pyperclip)')
    parser.add_argument('-d', '--directory', type=str, default='', help='Watch directory for new images')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay before reading new files (default 0.5)')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage even if GPU accelerators are available')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    group.add_argument('-q', '--quiet', action='store_true', help='Suppress all output except OCR results')
    
    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR

    logging.basicConfig(level=log_level, format='%(message)s')
    logger = logging.getLogger(__name__)

    mocr = MangaOcr(pretrained_model_name_or_path=args.model, force_cpu=args.force_cpu)

    if args.background:
        from .cli import ClipboardHandler
        handler = ClipboardHandler(mocr)
        handler.run()
    elif args.directory:
        from .cli import run_directory
        run_directory(mocr, args.directory, delay=args.delay)
    elif args.image_path:
        img_path = Path(args.image_path)
        if not img_path.exists():
            logger.error(f"Error: Image '{args.image_path}' not found.")
            sys.exit(1)
            
        try:
            img = Image.open(img_path).convert("RGB")
            text = mocr(img)
            print(text)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            sys.exit(1)
    else:
        logger.error("No image provided. Please provide an image path or use a background monitoring arg: -b or -d.")
        parser.print_help()

if __name__ == "__main__":
    main()
