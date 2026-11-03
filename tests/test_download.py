import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "download_data", Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
)
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)


def test_truncate_to_last_story(tmp_path):
    p = tmp_path / "t.txt"
    p.write_bytes(b"Story one.\n<|endoftext|>\nStory two.\n<|endoftext|>\nHalf a sto")
    dd.truncate_to_last_story(p)
    assert p.read_bytes() == b"Story one.\n<|endoftext|>\nStory two.\n<|endoftext|>\n"
    assert dd.count_stories(p) == 2
