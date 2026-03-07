# manga-ocr-torchless
A lightweight, **torch-free** version of the excellent [manga-ocr](https://github.com/kha-white/manga-ocr) by kha-white. 

This package uses **ONNX Runtime** for inference, making it significantly faster to install and run on machines without a GPU, eliminating the multi-gigabyte dependency on PyTorch.

By default, this package uses [mayocream's onnx export](https://huggingface.co/mayocream/manga-ocr-onnx) of the original manga-ocr package, but you can use any onnx export based on the manga-ocr package such as [l0wgear's onnx export](https://huggingface.co/l0wgear/manga-ocr-onnx) of [jzhang533/manga-ocr-base](https://huggingface.co/jzhang533/manga-ocr-base).

## Features
- **Lightweight:** No PyTorch dependency (~2GB saved).
- **Parity:** Achieves 100% character-level parity on the original test suite.
- **Fast:** Optimized for CPU inference using ONNX.

## Installation

```bash
pip install manga-ocr-torchless
```

*Note: The required ONNX models (~400MB) will be automatically downloaded from HuggingFace on the first run, not during installation.*

## Usage

### CLI

**Process a single image:**
```bash
manga_ocr path/to/image.jpg
```

**Monitor clipboard (auto-OCR every time you copy an image):**
```bash
manga_ocr -b
```

**Watch a directory for new screenshots:**
```bash
manga_ocr -d ./screenshots
```

### Python API

```python
from manga_ocr import MangaOcr
from PIL import Image

mocr = MangaOcr()
img = Image.open('image.jpg')
text = mocr(img)
print(text)
```

### Custom Models

You can use a different ONNX model by providing a HuggingFace repo ID or a local path to the constructor or via the `--model` flag in the CLI:

**Python:**
```python
mocr = MangaOcr(pretrained_model_name_or_path="user/repo-id")
# OR
mocr = MangaOcr(pretrained_model_name_or_path="./local_model_directory")
```

**CLI:**
```bash
manga_ocr --model "user/repo-id" path/to/image.jpg
```

## Acknowledgments
This project is a direct port of [manga-ocr](https://github.com/kha-white/manga-ocr) by **kha-white**. All credit for the model architecture and training belongs to them. This version simply swaps the backend to ONNX for a leaner distribution.

## License
Apache-2.0
