import pytest
from pathlib import Path
from PIL import Image
import json
from manga_ocr import MangaOcr

# Load the test data mapping
ASSETS_DIR = Path(__file__).parent / 'original_tests'
JSON_PATH = ASSETS_DIR / 'expected_results.json'

if JSON_PATH.exists():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
else:
    test_data = []

@pytest.fixture(scope="session")
def mocr():
    return MangaOcr()

@pytest.mark.parametrize("item", test_data)
def test_inference_original_assets(mocr, item):
    """
    Test OCR inference using the original manga-ocr test suite assets.
    """
    img_path = ASSETS_DIR / "images" / item["filename"]
    expected_text = item["result"]
    
    if not img_path.exists():
        pytest.skip(f"Test image not found at {img_path}")
        
    img = Image.open(img_path).convert("RGB")
    result_text = mocr(img)
    
    assert isinstance(result_text, str)
    assert len(result_text) > 0, "OCR returned empty text"
    
    assert result_text == expected_text, f"Mismatch for {item['filename']}. Expected '{expected_text}', got: '{result_text}'"
