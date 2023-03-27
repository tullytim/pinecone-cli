import unittest
import pinecone
import os
import random
import string
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
        
    def test_head(self):
        stats = self._run([f'{self.cli}', 'head', 'lpfactset'])    
        print(stats)
        self.assertIsNotNone(stats)
        
    def test_upsert(self):
        stats = self._run([f'{self.cli}', 'upsert', 'upsertfile', "[('vec1', [0.1, 0.2, 0.3], {'genre': 'drama'}), ('vec2', [0.2, 0.3, 0.4], {'genre': 'action'}),]"])    
        print(stats)
        self.assertIsNotNone(stats)    
        self.assertEqual(stats, 'upserted_count: 2')
        
    def test_fetch(self):
        stats = self._run([f'{self.cli}', 'fetch', 'lpfactset', "--vector_ids=\"05b4509ee655aacb10bfbb6ba212c65c\""])    
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