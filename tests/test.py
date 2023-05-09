import unittest
import os
import random
import string
import subprocess
from pkg_resources import parse_version


class TestPineconeCLI(unittest.TestCase):

    dir_path = os.path.dirname(os.path.realpath(__file__))

    TEST_INDEX = 'test_index'
    cli = f'{dir_path}/../pinecli.py'

    def _run(self, cmd, timeout=60):
        print(cmd)
        return subprocess.run(cmd, capture_output=True, check=True, universal_newlines=True, close_fds=True, timeout=timeout).stdout.strip()

    def _run_exit_code(self, cmd):
        print(cmd)
        return subprocess.run(cmd, check=True)

    def __run_returncode(self, cmd):
        print(cmd)
        process = subprocess.Popen(
            cmd, universal_newlines=True, close_fds=True)
        out, err = process.communicate()
        return process.returncode

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
        self.assertIsNotNone(stats)

    def test_describe_index_stats(self):
        stats = self._run([f'{self.cli}', 'describe-index-stats', 'lpfactset'])
        self.assertIsNotNone(stats)

    def test_query(self):
        stats = self._run([f'{self.cli}', 'query', 'upsertfile', 'random'])
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'query', 'upsertfile', '[1.0,2.0,3.0]'])
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'query', 'lpfactset', 'random', '--print-table'])
        self.assertIsNotNone(stats)
        """
        def _failedplot():
            stats = self._run(
                [f'{self.cli}', 'query', 'lpfactset', 'random', '--show-tsne=true', '--perplexity=2'])            
        self.assertRaises(subprocess.CalledProcessError, _failedplot)
        """

    def test_head(self):
        stats = self._run([f'{self.cli}', 'head', 'lpfactset'])
        self.assertIsNotNone(stats)

    def test_fail_connect(self):
        rc = self.__run_returncode(
            [f'{self.cli}', 'head', 'thisindexdoesnotexist'])
        self.assertNotEquals(rc, 0)

    def test_head_print(self):
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table'])
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table', '--include-values=true'])
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table', '--include-meta=true'])
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table', '--include-meta=true', '--include-values=true'])
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table', '--include-meta=false', '--include-values=true'])
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table', '--include-meta=true', '--include-values=false'])
        self.assertIsNotNone(stats)
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--print-table', '--include-meta=false', '--include-values=false'])
        self.assertIsNotNone(stats)

    def test_head_random_dims(self):
        stats = self._run(
            [f'{self.cli}', 'head', 'lpfactset', '--random_dims'])
        self.assertIsNotNone(stats)

    def test_list_collections(self):
        stats = self._run([f'{self.cli}', 'list-collections'])
        self.assertIsNotNone(stats)

    def test_desc_collection(self):
        stats = self._run([f'{self.cli}', 'describe-collection', 'testcoll'])
        self.assertIsNotNone(stats)

    def test_upsert(self):
        stats = self._run([f'{self.cli}', 'upsert', 'upsertfile',
                          "[('vec1', [0.1, 0.2, 0.3], {'genre': 'drama'}), ('vec2', [0.2, 0.3, 0.4], {'genre': 'action'}),]"])
        self.assertIsNotNone(stats)
        self.assertEqual(stats, 'upserted_count: 2')
        stats = self._run([f'{self.cli}', 'upsert', 'upsertfile',
                          "[('vec1', [0.1, 0.2, 0.3], {'genre': 'drama'}), ('vec2', [0.2, 0.3, 0.4], {'genre': 'action'}),]", '--debug'])
        self.assertIsNotNone(stats)

    def test_upsert_random(self):
        stats = self._run([f'{self.cli}', 'upsert-random',
                          'upsertfile', '--num_vectors=2', '--num_vector_dims=3'])
        self.assertIsNotNone(stats)
        stats = self._run([f'{self.cli}', 'upsert-random', 'upsertfile',
                          '--num_vectors=2', '--num_vector_dims=3', '--debug'])
        self.assertIsNotNone(stats)

    def test_minimize_cluster(self):
        rc = self.__run_returncode([f'{self.cli}', 'minimize-cluster',
                                       'upsertfile'])
        # this is going to fail as we cant reduce this pod any furhter than we are at
        self.assertEqual(rc, 1)

    def test_update(self):
        retcode = self.__run_returncode([f'{self.cli}', 'update',
                                         'id-1', 'upsertfile', '[0.1, 0.2, 0.3]'])
        self.assertEqual(retcode, 0)
        retcode = self.__run_returncode([f'{self.cli}', 'update',
                                         'id-1', 'upsertfile', '[0.1, 0.2, 0.3]', '--debug'])
        self.assertEqual(retcode, 0)
        retcode = self.__run_returncode([f'{self.cli}', 'update',
                                         'id-1', 'upsertfile', '[0.1, 0.2, 0.3]', '--metadata={\'foo\':\'asdf\'}'])
        self.assertEqual(retcode, 0)

    def test_upsert_webpage(self):
        openaiapikey = os.environ['OPENAI_API_KEY']
        print(f'KEY IS OF LENGTH: {len(openaiapikey)}')
        print('*'*30)
        print('running upsert on webpage')
        retcode = self.__run_returncode(
            [f'{self.cli}', 'upsert-webpage', 'https://yahoo.com', 'pageuploadtest', f'--openaiapikey={openaiapikey}'])
        self.assertEqual(retcode, 0)
        retcode = self.__run_returncode(
            [f'{self.cli}', 'upsert-webpage', 'https://yahoo.com', 'pageuploadtest', f'--openaiapikey={openaiapikey}', '--debug'])
        self.assertEqual(retcode, 0)
        # should throw ValueError
        retcode = self.__run_returncode(
            [f'{self.cli}', 'upsert-webpage', 'https://yahoo.com', 'pageuploadtest', f'--openaiapikey=', '--debug'])
        self.assertNotEqual(retcode, 0)

        openaiapikey = 'invalidkey'
        retcode = self.__run_returncode(
            [f'{self.cli}', 'upsert-webpage', 'https://yahoo.com', 'pageuploadtest', f'--openaiapikey={openaiapikey}', '--debug'])
        self.assertNotEqual(retcode, 0)

    def test_upsert_file(self):
        x = """
index,ID,Vectors,Metadata
1,abc,"[0.23223, -1.333, 0.2222222]",{'foo':'bar'}
2,ghi,"[0.23223, -1.333, 0.2222222]",{'bar':'baz'}
        """
        fname = './vectorfile.txt'
        with open(fname, "w+") as f:
            f.writelines(x)
        retcode = self.__run_returncode(
            [f'{self.cli}', 'upsert-file', fname, 'upsertfile', "{'id':'ID', 'vectors':'Vectors'}"])
        self.assertEqual(retcode, 0)
        # test bad mapping file
        retcode = self.__run_returncode(
            [f'{self.cli}', 'upsert-file', fname, 'upsertfile', "{'badkey1':'ID', 'badkey2':'Vectors'}"])
        self.assertNotEqual(retcode, 0)

    def test_fetch(self):
        stats = self._run([f'{self.cli}', 'fetch', 'lpfactset',
                           "--vector_ids=\"05b4509ee655aacb10bfbb6ba212c65c\""])
        self.assertIsNotNone(stats)
        stats = self._run([f'{self.cli}', 'fetch', 'lpfactset',
                          "--vector_ids=\"05b4509ee655aacb10bfbb6ba212c65c\"", '--pretty'])
        self.assertIsNotNone(stats)

    def test_delete_all(self):
        retcode = stats = self._run(
            [f'{self.cli}', 'delete-all', 'upsertfile'])
        self.assertNotEqual(retcode, 0)

    def test_create_delete_index(self):
        index_name = ''.join(random.choices(string.ascii_lowercase, k=7))
        index_name = f'testindex{index_name}'
        print(index_name)
        cmd = [self.cli, 'create-index', index_name, '--dims=16',
               '--pods=1', '--shards=1', '--pod-type=p1.x1']
        rv = self._run_exit_code(cmd)
        self.assertEqual(rv.returncode, 0)
        rev_index = index_name[::-1]
        cmd = [self.cli, 'delete-index', index_name]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.communicate(input=rev_index.encode())
        self.assertEqual(p.returncode, 0)


if __name__ == '__main__':
    unittest.main()
