import unittest
import os
import subprocess
from pkg_resources import parse_version


class TestPineconeCLI(unittest.TestCase):

    dir_path = os.path.dirname(os.path.realpath(__file__))

    TEST_INDEX = 'test_index'
    cli = f'{dir_path}/../pinecli.py'

    def _run(self, cmd):
        print(cmd)
        return subprocess.run(cmd, capture_output=True, check=True, universal_newlines=True).stdout.strip()

    def _run_exit_code(self, cmd):
        print(cmd)
        return subprocess.run(cmd, check=True)

    def test_version(self):
        cmd = [f'{self.cli}', 'version']
        v = self._run(cmd)
        print(f'Saw version {v}')
        self.assertTrue(parse_version(str(v)) > parse_version('0.1.0'))

    def test_help(self):
        rv = self._run([f'{self.cli}', '--help'])
        self.assertIsNotNone(rv)

    def test_list_indexes(self):
        indexes = self._run([f'{self.cli}', 'list-indexes'])
        length = len(indexes.split('\n'))
        self.assertGreater(length, 0)
        self.assertIsNotNone(indexes)
        
    def test_list_indexes_print(self):
        stats = self._run([f'{self.cli}', 'list-indexes', '--print-table'])
        self.assertIsNotNone(stats)

    def test_describe_index(self):
        stats = self._run([f'{self.cli}', 'describe-index', 'lpfactset'])
        print(stats)
        self.assertIsNotNone(stats)

    def test_describe_index_stats(self):
        stats = self._run([f'{self.cli}', 'describe-index-stats', 'lpfactset'])
        print(stats)
        self.assertIsNotNone(stats)

    def test_query(self):
        stats = self._run([f'{self.cli}', 'query', 'lpfactset', 'random'])
        print(stats)
        self.assertIsNotNone(stats)

    def test_query_print_table(self):
        stats = self._run(
            [f'{self.cli}', 'query', 'lpfactset', 'random', '--print-table'])
        print(stats)
        self.assertIsNotNone(stats)

    def test_head(self):
        stats = self._run([f'{self.cli}', 'head', 'lpfactset'])
        print(stats)
        self.assertIsNotNone(stats)

    def test_head_print(self):
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table'])
        print(stats)
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table', '--include-values=true'])
        print(stats)
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table', '--include-meta=true'])
        print(stats)
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table', '--include-meta=true', '--include-values=true'])
        print(stats)
        self.assertIsNotNone(stats)

    def test_head_random_dims(self):
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--random_dims'])
        print(stats)
        self.assertIsNotNone(stats)

    def test_list_collections(self):
        stats = self._run([f'{self.cli}', 'list-collections'])
        print(stats)
        self.assertIsNotNone(stats)

    def test_desc_collection(self):
        stats = self._run([f'{self.cli}', 'describe-collection', 'testcoll'])
        print(stats)
        self.assertIsNotNone(stats)

    def test_upsert(self):
        stats = self._run([f'{self.cli}', 'upsert', 'upsertfile',
                          "[('vec1', [0.1, 0.2, 0.3], {'genre': 'drama'}), ('vec2', [0.2, 0.3, 0.4], {'genre': 'action'}),]"])
        print(stats)
        self.assertIsNotNone(stats)
        self.assertEqual(stats, 'upserted_count: 2')
        
    def test_upsert_random(self):
        stats = self._run([f'{self.cli}', 'upsert-random', 'upsertfile', '--num_vectors=2', '--num_vector_dims=3'])
        print(stats)
        self.assertIsNotNone(stats)

    """
    def test_upsert_webpage(self):
        openaiapikey = os.getenv('OPENAI_API_KEY')
        stats = self._run([f'{self.cli}', 'upsert-webpage', 'https://www.menlovc.com',
                          'pageuploadtest', f'--openaiapikey={openaiapikey}'])
        print(stats)
        self.assertIsNotNone(stats)
    """

    def test_fetch(self):
        stats = self._run([f'{self.cli}', 'fetch', 'lpfactset',
                           "--vector_ids=\"05b4509ee655aacb10bfbb6ba212c65c\""])
        print(stats)
        self.assertIsNotNone(stats)

    def test_fetch_pretty(self):
        stats = self._run([f'{self.cli}', 'fetch', 'lpfactset',
                          "--vector_ids=\"05b4509ee655aacb10bfbb6ba212c65c\"", '--pretty'])
        print(stats)
        self.assertIsNotNone(stats)
    """
    def test_create_delete_index(self):
        index_name = ''.join(random.choices(string.ascii_lowercase, k=7))
        index_name = f'testindex{index_name}'
        print(index_name)
        cmd = [self.cli, 'create-index', index_name, '--dims=16', '--pods=1', '--shards=1', '--pod-type=p1.x1']
        rv = self._run_exit_code(cmd)
        self.assertEqual(rv.returncode, 0)
        rev_index = index_name[::-1]
        cmd = [self.cli, 'delete-index', index_name]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.communicate(input=rev_index.encode())
        self.assertEqual(p.returncode, 0)
    """

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
