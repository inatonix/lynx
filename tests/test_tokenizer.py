from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from lynx.tokenizer import UPSTREAM, build, implementation, load_model, main, save_model


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.stdout = redirect_stdout(io.StringIO())
        self.stderr = redirect_stderr(io.StringIO())
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def test_upstream_files_are_unmodified(self):
        manifest = json.loads((UPSTREAM / "SHA256SUMS.json").read_text())
        for name, expected in manifest.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest(), expected)

    def test_unicode_characters_and_bytes_differ(self):
        text = "hello世界😁"
        char, _ = build(1, "", 256)
        byte, _ = build(2, "", 256)
        self.assertEqual(len(char.encode(text)), 8)
        self.assertEqual(len(byte.encode(text)), 15)
        self.assertEqual(byte.decode(byte.encode(text)), text)

    def test_roundtrip_including_unseen_text_and_empty_input(self):
        for section in range(1, 8):
            tokenizer, _ = build(section, "banana banana<|endoftext|>bandana", 270)
            for text in ("", "未知の文字🦊", "banana", "<|endoftext|><|endoftext|>", " a\n\t"):
                with self.subTest(section=section, text=text):
                    self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_special_token_is_one_id_and_not_merged_across_documents(self):
        for section in (5, 6, 7):
            tokenizer, _ = build(section, "a<|endoftext|>b", 270)
            self.assertEqual(tokenizer.encode("a<|endoftext|>b"), [97, tokenizer.end_token_id, 98])

    def test_pretokenization_preserves_boundaries(self):
        module = implementation(6)
        self.assertEqual(module["pretokenize"]("hello world!"), ["hello", " world", "!"])
        tokenizer, _ = build(6, "hello world!", 400)
        self.assertEqual(len(tokenizer.encode("hello world!")), 3)

    def test_tie_breaking_difference_is_preserved(self):
        _, earlier = build(6, "ab cd", 258)
        _, shared = build(7, "ab cd", 258)
        self.assertEqual(list(earlier), [(97, 98)])
        self.assertEqual(list(shared), [(99, 100)])

    def test_saved_models_preserve_ids_for_every_bpe_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            for section in range(3, 8):
                tokenizer, rules = build(section, "banana banana<|endoftext|>bandana", 270)
                save_model(path, section, rules, {})
                loaded = load_model(path)
                self.assertEqual(loaded.encode("banana世界<|endoftext|>"),
                                 tokenizer.encode("banana世界<|endoftext|>"))

    def test_invalid_rules_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"format": "lynx-bpe-v1", "section": 7,
                                        "merges": [[999, 0, 256]]}))
            with self.assertRaisesRegex(ValueError, "merge rule"):
                load_model(path)

    def test_small_tinycodes_training_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.json"
            self.assertEqual(main(["train", "--max-chars", "1000", "--vocab-size", "270",
                                   "--output", str(path)]), 0)
            self.assertEqual(main(["encode", "--model", str(path), "--text", "こんにちは"]), 0)
            self.assertEqual(json.loads(path.read_text())["training"]["characters"], 1000)

    def test_vocab_size_and_character_limit_validation(self):
        with self.assertRaises(ValueError):
            build(6, "test", 256)
        self.assertEqual(main(["train", "--max-chars", "-1"]), 1)


if __name__ == "__main__":
    unittest.main()
