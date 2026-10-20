import sys
from types import SimpleNamespace

import pytest

from plan_extractor.ocr_engine import EasyOCREngine, create_ocr_engine


def _stub_easyocr(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "easyocr",
        SimpleNamespace(Reader=lambda *args, **kwargs: object()),
    )


def test_factory_returns_easyocr_engine_when_selected(monkeypatch):
    _stub_easyocr(monkeypatch)

    engine = create_ocr_engine({"ocr": {"engine": "easyocr"}})

    assert isinstance(engine, EasyOCREngine)


def test_factory_defaults_to_easyocr_when_engine_key_is_missing(monkeypatch):
    _stub_easyocr(monkeypatch)

    engine = create_ocr_engine({"ocr": {}})

    assert isinstance(engine, EasyOCREngine)


def test_factory_rejects_unknown_engine():
    with pytest.raises(
        ValueError,
        match=r"Unknown OCR engine 'unsupported'.*easyocr",
    ):
        create_ocr_engine({"ocr": {"engine": "unsupported"}})
