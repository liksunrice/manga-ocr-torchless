import argparse
import sys
from PIL import Image
from pathlib import Path

from .ocr import MangaOcr

def main():
    parser = argparse.ArgumentParser(description="Manga OCR (Torchless/ONNX)")
    parser.add_argument('image_path', type=str, nargs='?', help='Path to an image file to OCR')
    parser.add_argument('--model', type=str, default='l0wgear/manga-ocr-2025-onnx',
                        help='HuggingFace repo ID or local path to ONNX model')
    parser.add_argument('-b', '--background', action='store_true', help='Run in background, reading from clipboard (requires pyperclip)')
    parser.add_argument('-d', '--directory', type=str, default='', help='Watch directory for new images')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay before reading new files (default 0.5)')
    args = parser.parse_args()

    mocr = MangaOcr(pretrained_model_name_or_path=args.model)

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
            print(f"Error: Image '{args.image_path}' not found.", file=sys.stderr)
            sys.exit(1)
            
        try:
            img = Image.open(img_path).convert("RGB")
            text = mocr(img)
            print(text)
        except Exception as e:
            print(f"Error processing image: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("No image provided. Please provide an image path or use a background monitoring arg: -b or -d.", file=sys.stderr)
        parser.print_help()

if __name__ == "__main__":
    main()
