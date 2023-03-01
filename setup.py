from setuptools import setup

setup(
    name='pinecone-cli',
    description='pinecone-cli is a command-line client for interacting with the pinecone vector embedding database.',
    version='0.2.0',
    url='https://github.com/tullytim/pinecone-cli',
    author='Tim Tully',
    author_email='tim@menlovc.com',
    license='MIT',
    keywords='pinecone vector vectors embeddings database transformers models',
    python_requires='>=3',
    py_modules=['pinecli'],
    install_requires=[
        'click', 'pandas', 'numpy', 'openai', 'pinecone-client', 'pinecone-client[grpc]', 'matplotlib', \
            'scikit-learn', 'beautifulsoup4', 'nltk', 'sklearn', 'rich', 'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'pinecli = pinecli:cli',
        ],
    },
)
