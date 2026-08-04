from japanese_anki.identifiers import stable_record_id


def test_stable_record_id_normalizes_width_and_space() -> None:
    assert stable_record_id(" 話す ", "はなす") == "word:話す:はなす"
    assert stable_record_id("Ａ", "エー") == "word:A:エー"
