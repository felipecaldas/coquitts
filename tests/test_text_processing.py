import sys
from pathlib import Path

# Ensure project root is on sys.path for module imports like `utils`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from utils import normalize_text, preprocess_pt_text, split_text_into_chunks


def test_normalize_text_quotes_and_spaces():
    raw = "“Amazônia” ‘Z’\u00A0teste"
    norm = normalize_text(raw)
    assert '“' not in norm and '”' not in norm and '‘' not in norm and '’' not in norm
    assert '"Amazônia"' in norm
    assert "'Z'" in norm
    assert "\u00A0" not in norm
    assert "  " not in norm  # no double spaces


def test_preprocess_pt_text_preserves_sentences():
    text = (
        "Um explorador inglês entrou na Amazônia em 1925 e nunca mais voltou. "
        "O coronel Percy Fawcett procurava uma cidade perdida cheia de ouro. "
        "Ele levou o filho e um amigo na expedição. Os três homens sumiram para sempre na selva. "
        "Mais de 100 pessoas morreram tentando encontrá-los. Fawcett havia encontrado ruínas estranhas antes. "
        "Ele chamava a cidade misteriosa de 'Z'. Cartões postais chegaram depois do sumiço, mas eram falsos. "
        "A Amazônia guardou o segredo até hoje. Ninguém sabe se ele encontrou a cidade dourada ou virou comida de onça."
    )
    pre = preprocess_pt_text(text)
    # Ensure key sentences are present after preprocessing
    assert "misteriosa de 'Z'" in pre
    assert "Cartões postais chegaram depois do sumiço, mas eram falsos" in pre
    assert "A Amazônia guardou o segredo até hoje" in pre


def test_split_text_into_chunks_contains_critical_phrases():
    text = (
        "Um explorador inglês entrou na Amazônia em 1925 e nunca mais voltou. "
        "O coronel Percy Fawcett procurava uma cidade perdida cheia de ouro. "
        "Ele levou o filho e um amigo na expedição. Os três homens sumiram para sempre na selva. "
        "Mais de 100 pessoas morreram tentando encontrá-los. Fawcett havia encontrado ruínas estranhas antes. "
        "Ele chamava a cidade misteriosa de 'Z'. Cartões postais chegaram depois do sumiço, mas eram falsos. "
        "A Amazônia guardou o segredo até hoje. Ninguém sabe se ele encontrou a cidade dourada ou virou comida de onça."
    )
    pre = preprocess_pt_text(text)
    chunks = split_text_into_chunks(pre, max_length=120)  # small to enforce multiple chunks
    combined = "\n".join(chunks)
    assert "misteriosa de 'Z'" in combined
    assert "Cartões postais chegaram depois do sumiço, mas eram falsos" in combined
    assert "A Amazônia guardou o segredo até hoje" in combined
