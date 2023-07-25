from setuptools import setup
from pathlib import Path
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.build import build
from setuptools.command.egg_info import egg_info

import os
import sys
import time

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='pinecone-cli',
    description='pinecone-cli is a command-line client for interacting with the pinecone vector embedding database.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='1.1',
    url='https://github.com/tullytim/pinecone-cli',
    author='Tim Tully',
    author_email='tim@menlovc.com',
    license='MIT',
    keywords='pinecone vector vectors embeddings database transformers models',
    python_requires='>=3',
    py_modules=['pinecli'],
    install_requires=[
        'click', 'pandas', 'numpy', 'openai', 'pinecone-client', 'pinecone-client[grpc]', 'matplotlib', \
            'scikit-learn', 'beautifulsoup4', 'nltk', 'retry', 'rich', 'python-dotenv', 'openai', 'typing'
    ],
    entry_points={
        'console_scripts': [
            'pinecli = pinecli:cli',
        ],
    }
)
