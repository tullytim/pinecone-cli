from setuptools import setup

setup(
    name='pinecli',
    version='0.1.0',
    py_modules=['pinecli'],
    install_requires=[
        'Click', 'pandas', 'numpy', 'openai', 'pinecone-client', 'matplotlib', \
            'scikit-learn', 'beautifulsoup4', 'nltk', 'sklearn', 'rich'
    ],
    entry_points={
        'console_scripts': [
            'pinecli = pinecli:cli',
        ],
    },
)