from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='pinecone-cli',
    description='pinecone-cli is a command-line client for interacting with the pinecone vector embedding database.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.5.1',
    url='https://github.com/tullytim/pinecone-cli',
    author='Tim Tully',
    author_email='tim@menlovc.com',
    license='MIT',
    keywords='pinecone vector vectors embeddings database transformers models',
    python_requires='>=3',
    py_modules=['pinecli'],
    install_requires=[
        'click', 'pandas', 'numpy', 'openai', 'pinecone-client', 'pinecone-client[grpc]', 'matplotlib', \
            'scikit-learn', 'beautifulsoup4', 'nltk', 'sklearn', 'rich', 'python-dotenv', 'openai', 'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'pinecli = pinecli:cli',
        ],
    },
)
