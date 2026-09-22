import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
from urllib.error import URLError

from lynx.cli import main
from lynx.client import LynxError, chat


class ClientTests(unittest.TestCase):
    @patch("lynx.client.urlopen")
    def test_request_and_response(self, urlopen):
        urlopen.return_value.__enter__.return_value = io.BytesIO(
            json.dumps({"message": {"content": "こんにちは"}}).encode()
        )
        messages = [{"role": "user", "content": "hello"}]
        self.assertEqual(chat(messages, "test-model"), "こんにちは")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(json.loads(request.data), {
            "model": "test-model", "messages": messages, "stream": False,
        })

    @patch("lynx.client.urlopen", side_effect=URLError("offline"))
    def test_connection_error_is_actionable(self, _):
        with self.assertRaisesRegex(LynxError, "ollama serve"):
            chat([], "test-model")

    @patch("lynx.client.urlopen")
    def test_malformed_response(self, urlopen):
        for body in (b"not json", b"null", b'{"message": {}}'):
            with self.subTest(body=body):
                urlopen.return_value.__enter__.return_value = io.BytesIO(body)
                with self.assertRaises(LynxError):
                    chat([], "test-model")


class CliTests(unittest.TestCase):
    @patch("lynx.cli.chat", side_effect=LynxError("offline"))
    def test_single_prompt_fails_with_nonzero_exit(self, _):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["hello"]), 1)

    @patch("builtins.input", side_effect=["one", "failed", "two", "/clear", "three", "/exit"])
    @patch("lynx.cli.chat", side_effect=["first", LynxError("offline"), "second", "third"])
    def test_history_failure_recovery_and_clear(self, backend, _):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main([]), 0)
        self.assertEqual(backend.call_args_list[2].args[0], [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
        ])
        self.assertEqual(backend.call_args_list[3].args[0], [
            {"role": "user", "content": "three"},
        ])


if __name__ == "__main__":
    unittest.main()
