"""Artifact installation and benchmark selection tests (stdlib only)."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'plotting'))
import _reproduction as reproduction

spec = importlib.util.spec_from_file_location('fetch_data', ROOT / 'scripts/fetch_data.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'install'
        self.root.mkdir()
        self.archive = Path(self.temporary.name) / 'example.tar.gz'

    def make_archive(self, entries):
        with tarfile.open(str(self.archive), 'w:gz') as archive:
            for name, payload, kind in entries:
                info = tarfile.TarInfo(name)
                info.type = kind
                if kind == tarfile.REGTYPE:
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                else:
                    if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                        info.linkname = '/outside'
                    archive.addfile(info)
        return {'filename': self.archive.name, 'bytes': self.archive.stat().st_size,
                'checksum': {'algorithm': 'md5', 'value': hashlib.md5(self.archive.read_bytes()).hexdigest()},
                'archive_root': 'final_checkpoints', 'destination': 'data/final_benchmarks'}

    def test_misnamed_benchmark_root_and_apple_metadata(self):
        artifact = self.make_archive([
            ('./._final_checkpoints', b'apple', tarfile.REGTYPE),
            ('./final_checkpoints/._result.json', b'apple', tarfile.REGTYPE),
            ('./final_checkpoints/arnold/result.json', b'{}', tarfile.REGTYPE)])
        fetch.install_archive(self.archive, artifact, self.root)
        self.assertEqual((self.root / 'data/final_benchmarks/arnold/result.json').read_bytes(), b'{}')
        self.assertFalse((self.root / 'data/final_checkpoints').exists())
        self.assertEqual(len(list((self.root / 'data/final_benchmarks').rglob('*.json'))), 1)

    def test_checksum_checked_before_extraction(self):
        artifact = self.make_archive([('final_checkpoints/result.json', b'{}', tarfile.REGTYPE)])
        artifact['checksum']['value'] = '0' * 32
        with self.assertRaisesRegex(ValueError, 'checksum'):
            fetch.install_archive(self.archive, artifact, self.root)
        self.assertFalse((self.root / 'data').exists())

    def test_size_checked_before_extraction(self):
        artifact = self.make_archive([('final_checkpoints/result.json', b'{}', tarfile.REGTYPE)])
        artifact['bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'size'):
            fetch.install_archive(self.archive, artifact, self.root)
        self.assertFalse((self.root / 'data').exists())

    def test_invalid_members_leave_installation_untouched(self):
        for name, kind in [('../outside', tarfile.REGTYPE), ('/outside', tarfile.REGTYPE),
                           ('wrong_root/data', tarfile.REGTYPE),
                           ('final_checkpoints/link', tarfile.SYMTYPE),
                           ('final_checkpoints/link', tarfile.LNKTYPE)]:
            with self.subTest(name=name, kind=kind):
                artifact = self.make_archive([('final_checkpoints/good.json', b'{}', tarfile.REGTYPE), (name, b'bad', kind)])
                with self.assertRaises(ValueError):
                    fetch.install_archive(self.archive, artifact, self.root)
                self.assertFalse((self.root / 'data/final_benchmarks').exists())
                self.assertEqual(list((self.root / 'data').glob('.artifact-staging-*')), [])

    def test_conflict_preflight_preserves_existing_files(self):
        target = self.root / 'data/final_benchmarks/z_existing.json'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'user result')
        artifact = self.make_archive([('final_checkpoints/a_new.json', b'new', tarfile.REGTYPE),
                                      ('final_checkpoints/z_existing.json', b'different', tarfile.REGTYPE)])
        with self.assertRaises(FileExistsError):
            fetch.install_archive(self.archive, artifact, self.root)
        self.assertEqual(target.read_bytes(), b'user result')
        self.assertFalse((target.parent / 'a_new.json').exists())

    def test_reinstallation_reuses_identical_files(self):
        artifact = self.make_archive([('final_checkpoints/result.json', b'{}', tarfile.REGTYPE)])
        fetch.install_archive(self.archive, artifact, self.root)
        target = self.root / 'data/final_benchmarks/result.json'
        before = target.stat().st_mtime_ns
        fetch.install_archive(self.archive, artifact, self.root)
        self.assertEqual(target.stat().st_mtime_ns, before)

    def test_destination_symlink_cannot_redirect_installation(self):
        outside = Path(self.temporary.name) / 'outside'
        outside.mkdir()
        (self.root / 'data').mkdir()
        (self.root / 'data/final_benchmarks').symlink_to(outside, target_is_directory=True)
        artifact = self.make_archive([('final_checkpoints/result.json', b'{}', tarfile.REGTYPE)])
        with self.assertRaises(ValueError):
            fetch.install_archive(self.archive, artifact, self.root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_download_rejects_bad_payload_without_caching_it(self):
        artifact = self.make_archive([('final_checkpoints/result.json', b'{}', tarfile.REGTYPE)])
        artifact['url'] = 'https://example.invalid/archive'
        cache = Path(self.temporary.name) / 'cache'
        with mock.patch.object(fetch.urllib.request, 'urlopen', return_value=io.BytesIO(b'bad')):
            with self.assertRaises(ValueError):
                fetch.fetch_archive(artifact, cache)
        self.assertEqual(list(cache.iterdir()), [])

    def test_download_installs_only_verified_payload(self):
        artifact = self.make_archive([('final_checkpoints/result.json', b'{}', tarfile.REGTYPE)])
        artifact['url'] = 'https://example.invalid/archive'
        cache = Path(self.temporary.name) / 'cache'
        payload = self.archive.read_bytes()
        with mock.patch.object(fetch.urllib.request, 'urlopen', return_value=io.BytesIO(payload)):
            result = fetch.fetch_archive(artifact, cache)
        self.assertEqual(result.read_bytes(), payload)
        self.assertEqual(list(cache.glob('*.part')), [])


class BenchmarkSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_benchmarks(self, method):
        data = json.loads((ROOT / 'data/reproduction/policies.json').read_text())
        policies = [p for p in data['policies'] if p['method'] == method]
        paths = []
        for policy in policies:
            path = self.root / policy['benchmark']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
            paths.append(path)
        return paths

    def test_task_sv_partial_files_do_not_become_extra_seeds(self):
        canonical = self.make_benchmarks('obc_task_sv')
        for path in canonical:
            (path.parent / 'duplicate_hand_index_reach_results.json').write_text('{"hand_index_reach": {}}')
        selected = reproduction.benchmark_result_paths('obc_task_sv', self.root)
        self.assertEqual(selected, [str(p) for p in canonical])
        self.assertEqual(len(selected), 3)

    def test_method_alias_selects_same_benchmarks(self):
        canonical = self.make_benchmarks('mt_ppo')
        self.assertEqual(reproduction.benchmark_result_paths('mt-ppo', self.root), [str(p) for p in canonical])


if __name__ == '__main__':
    unittest.main()
