import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nacl.public import PrivateKey, SealedBox

SCRIPT = Path(__file__).with_name("encrypt_wifi.py")
spec = importlib.util.spec_from_file_location("encrypt_wifi", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EncryptionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="doorbell-wifi-test-")
        self.root = Path(self.directory.name)
        self.output = self.root / "wifi.json"
        self.private = PrivateKey(bytes([7]) * 32)
        self.public = bytes(self.private.public_key).hex()
        self.mock = self.root / "secretspec"
        self.env = dict(os.environ, PATH=str(self.root) + ":" + os.environ["PATH"])
        self.mock.write_text(
            "#!" + sys.executable + "\nimport os,sys\n"
            'args=sys.argv[sys.argv.index("--")+1:]\n'
            'env=dict(os.environ,WIFI_SSID="unit-test-network",WIFI_PASS="unit-test-password")\n'
            "os.execvpe(args[0],args,env)\n"
        )
        self.mock.chmod(0o700)

    def tearDown(self):
        self.directory.cleanup()

    def invoke(self, public=True):
        args = [sys.executable, str(SCRIPT), "--output", str(self.output)]
        if public:
            args += ["--public-key", self.public]
        return subprocess.run(args, env=self.env, capture_output=True, check=False)

    def assert_no_credentials(self, result):
        self.assertNotIn(b"unit-test-network", result.stdout + result.stderr)
        self.assertNotIn(b"unit-test-password", result.stdout + result.stderr)

    def test_end_to_end_and_reencryption(self):
        first = self.invoke()
        self.assertEqual(first.returncode, 0)
        self.assert_no_credentials(first)
        data = json.loads(self.output.read_bytes())
        self.assertEqual(set(data), {"format", "recipient", "ciphertext"})
        ciphertext = bytes.fromhex(data["ciphertext"])
        plaintext = SealedBox(self.private).decrypt(ciphertext)
        self.assertEqual(len(plaintext), 100)
        self.assertEqual(plaintext[4:21], b"unit-test-network")
        self.assertEqual(plaintext[36:54], b"unit-test-password")
        second = self.invoke(public=False)
        self.assertEqual(second.returncode, 0)
        self.assert_no_credentials(second)
        updated = bytes.fromhex(json.loads(self.output.read_bytes())["ciphertext"])
        self.assertNotEqual(ciphertext, updated)
        self.assertEqual(SealedBox(self.private).decrypt(updated), plaintext)

    def test_provider_failure_is_redacted_and_preserves_file(self):
        self.assertEqual(self.invoke().returncode, 0)
        old = self.output.read_bytes()
        self.mock.write_text(
            "#!" + sys.executable + "\nimport sys\n"
            'print("unit-test-network")\nprint("unit-test-password",file=sys.stderr)\nsys.exit(1)\n'
        )
        result = self.invoke()
        self.assertEqual(result.returncode, 1)
        self.assert_no_credentials(result)
        self.assertEqual(self.output.read_bytes(), old)

    def test_malformed_provider_result_is_redacted(self):
        self.mock.write_text("#!" + sys.executable + '\nprint("unit-test-password")\n')
        result = self.invoke()
        self.assertEqual(result.returncode, 1)
        self.assert_no_credentials(result)
        self.assertFalse(self.output.exists())

    def test_invalid_inputs_rejected_and_utf8_accepted(self):
        for ssid, password in [
            ("", "12345678"),
            ("x" * 33, "12345678"),
            ("ok", "short"),
            ("ok", "z" * 64),
        ]:
            with self.assertRaises(ValueError):
                module.encrypt(self.public, ssid, password)
        data = module.encrypt(self.public, "玄関", "12345678")
        plain = SealedBox(self.private).decrypt(bytes.fromhex(data["ciphertext"]))
        self.assertEqual(plain[1], 6)
        self.assertEqual(plain[4:10], "玄関".encode())


if __name__ == "__main__":
    unittest.main(verbosity=2)
