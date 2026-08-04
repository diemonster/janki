from japanese_anki.io import merge_records
from japanese_anki.models import VocabularyRecord


def test_merge_preserves_existing_enrichment() -> None:
    existing = [
        VocabularyRecord(
            id="word:話す:はなす",
            expression="話す",
            reading="はなす",
            furigana="話[はな]す",
            meanings=["to speak"],
            usage_notes="Curated note",
            tags=["curated"],
        )
    ]
    incoming = [
        VocabularyRecord(
            id="word:話す:はなす",
            expression="話す",
            reading="はなす",
            meanings=["to speak", "to talk"],
            tags=["shirabe"],
        )
    ]

    merged, counts = merge_records(existing, incoming)
    assert counts["updated"] == 1
    assert merged[0].furigana == "話[はな]す"
    assert merged[0].usage_notes == "Curated note"
    assert merged[0].meanings == ["to speak", "to talk"]
    assert merged[0].tags == ["curated", "shirabe"]
