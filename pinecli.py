#!/usr/bin/env python3
# encoding: utf-8
# pylint: disable=line-too-long,too-many-arguments,invalid-name,no-member,missing-function-docstring,missing-module-docstring

import hashlib
import itertools
import os
import requests
import sys
import unittest

from ast import literal_eval
from time import sleep

import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openai
import nltk.data
import pinecone

from bs4 import BeautifulSoup
from bs4.element import Comment, Tag
from dotenv import load_dotenv, find_dotenv
from numpy import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from retry import retry
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm
from typing import List, Optional, TypeVar

DEFAULT_REGION = 'us-west1-gcp'
OPENAI_EMBED_MODEL = "text-embedding-ada-002"
REGION_HELP = 'Pinecone cluster region'

# take environment variables from .env
load_dotenv(find_dotenv(), override=True)

U = TypeVar('U')


def nn(inst: Optional[U]) -> U:
    """Not-none helper to stop mypy errors"""
    assert inst is not None
    return inst


def tag_visible(element: Tag) -> bool:
    """ Strip out undesirables """
    parent: Tag = nn(element.parent)
    if parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def _text_from_html(body: str) -> str:
    """ Obv pull text from doc with tag_visible filters """
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)
    return " ".join(t.strip() for t in visible_texts)


@retry(tries=3, delay=5)
def _get_openai_embedding(apikey: str, data: List[str]) -> openai.Embedding:
    openai.api_key = apikey
    return openai.Embedding.create(input=data, engine=OPENAI_EMBED_MODEL)


@click.group()
def cli() -> None:
    """ A command line interface for working with Pinecone. """
    pass


def _pinecone_init(apikey: str, environment: str, indexname: str = '') -> pinecone.GRPCIndex:
    index = None
    apikey = os.environ.get(
        'PINECONE_API_KEY', apikey) if apikey is None else apikey
    environment = os.environ.get(
        'PINECONE_ENVIRONMENT', environment) if environment is None else environment
    if apikey is None:
        sys.exit(
            "No Pinecone API key set through PINECONE_API_KEY in .env file, environment variable or --apikey\nExample: export PINECONE_API_KEY=1234-abc-9876")
    pinecone.init(api_key=apikey, environment=environment)
    if indexname:
        try:
            # index = pinecone.Index(indexname)
            index = pinecone.GRPCIndex(indexname)
        except:
            sys.exit("Unable to connect.  Caught exception:")
    return index


def _format_values(vals) -> str:
    return ",".join(str(x) for x in vals)[:30]


def _print_table(res, pinecone_index_name, namespace, include_meta, include_values, expand_meta) -> None:
    """ Print a table of results """
    table = Table(
        title=f"🌲 {pinecone_index_name} ns=({namespace}) Index Results")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("NS", justify="left", style="red", no_wrap=True)
    if include_values:
        table.add_column("Values", style="magenta")
    if include_meta:
        table.add_column("Meta", justify="right", style="green")
    table.add_column("Score", justify="right", style="green")

    ns = res['namespace'] if 'namespace' in res else ''
    for row in res.matches:
        metadata = ''
        score = str(row['score'])
        vecid = row['id']
        if include_meta and 'metadata' in row:
            metadata = str(row['metadata'])
            metadata = metadata[:100] if not expand_meta else metadata

        if include_values and include_meta:
            table.add_row(vecid, ns, _format_values(
                row['values']), metadata, score)
        elif include_values and not include_meta:
            table.add_row(vecid, ns, _format_values(
                row['values']), score)
        elif not include_values and include_meta:
            table.add_row(vecid, ns, metadata, score)
        elif not include_values and not include_meta:
            table.add_row(vecid, ns, score)

    console = Console()
    console.print(table)


@click.command(short_help='Prints version number.')
def version() -> None:
    if sys.version_info >= (3, 8):
        from importlib import metadata
    else:
        import importlib_metadata as metadata
    click.echo(metadata.version('pinecone_cli'))


@click.command(short_help='Queries Pinecone with a given vector.')
@click.option('--apikey',  help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.option('--include_values', help='Should we return the vectors', show_default=True, default=True)
@click.option('--topk', '--numrows', 'topk', type=click.INT, show_default=True, default=10, help='Top K number to return')
@click.option('--namespace',  default="", help='Namespace to select results from')
@click.option('--include-meta', help='Whether to include the metadata values', default=False, show_default=True)
@click.option('--expand-meta', help='Whether to fully expand the metadata returned.', is_flag=True, show_default=True, default=False)
@click.option('--meta_filter', help='Filter out metadata w/ the Pinecone filter syntax which is really JSON.  Default is no filter.', default="{}")
@click.option('--print-table', help='Display the output as a pretty table.', is_flag=True, show_default=True, default=False)
@click.option('--num-clusters', help='Number of clusters in TSNE plot if --show-tsne is used.', type=click.INT, show_default=True, default=4)
@click.option('--perplexity', '--perp', help='The perplexity of the TSNE plot, if --show-tsne is used.', type=click.INT, default=15, show_default=True)
@click.option('--tsne-random-state', type=click.INT, default=42, show_default=True)
@click.option('--show-tsne', default=False)
@click.argument('pinecone_index_name')
@click.argument('query_vector')
def query(pinecone_index_name, apikey, query_vector, region=DEFAULT_REGION, topk=10, include_values=True, include_meta=True, expand_meta=False, num_clusters=4,
          perplexity=15, tsne_random_state=True, namespace="", show_tsne=False, meta_filter="{}", print_table=False) -> None:
    """ Queries Pinecone index named <PINECONE_INDEX_NAME> with the given <QUERY_VECTOR> and optional namespace. 

        \b
        Example: 
        % ./pinecli.py query lpfactset  "[0,0]"

        \b
        Example 2:
        % ./pinecli.py query  upsertfile  "[1.2, 1.0, 3.0]" --print-table --include-meta=true  --filter="{'genre':'drama'}"

        \b 
        Example 3 [Query randomly]:
        % ./pinecli.py query lpfactset random 

        For filter syntax see: https://docs.pinecone.io/docs/metadata-filtering
    """
    index = _pinecone_init(apikey, region, pinecone_index_name)

    if query_vector.lower() == "random":
        res = index.describe_index_stats()
        num_vector_dims = res['dimension']
        query_vector = [i for i in range(num_vector_dims)]
    else:
        query_vector = literal_eval(query_vector)

    res = index.query(vector=query_vector, top_k=topk, include_metadata=True,
                      include_values=include_values, namespace=namespace, filter=literal_eval(meta_filter))
    if print_table:
        _print_table(res, pinecone_index_name, namespace,
                     include_meta, include_values, expand_meta)
    else:
        click.echo(res)

    if show_tsne:
        show_tsne_plot(pinecone_index_name, res.matches,
                       num_clusters, perplexity, tsne_random_state)


def show_tsne_plot(pinecone_index_name, results, num_clusters, perplexity, random_state):  # pragma: no cover
    res2 = np.asarray([np.array(v['values']) for v in results])
    df = pd.DataFrame(data=res2)

    kmeans = KMeans(n_clusters=num_clusters, init="k-means++",
                    random_state=random_state, n_init='auto')
    kmeans.fit(res2)
    labels = kmeans.labels_
    df["Cluster"] = labels

    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, init='random', learning_rate=200)
    embeddings2d = tsne.fit_transform(res2)
    x = [x for x, y in embeddings2d]
    y = [y for x, y in embeddings2d]
    (_, ax) = plt.subplots(figsize=(9, 6))  # inches
    plt.style.use('seaborn-whitegrid')
    plt.grid(color='#EAEAEB', linewidth=0.5)
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['left'].set_color('#2B2F30')
    ax.spines['bottom'].set_color('#2B2F30')
    for category, color in enumerate(["purple", "green", "red", "blue", "brown", "gray", "olive", "cyan", "orange",  "pink"]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)
        avg_x = xs.mean()
        avg_y = ys.mean()
        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

    plt.title(
        f"Clustering of Pinecone Index {pinecone_index_name}", fontsize=16, fontweight='bold', pad=20)
    plt.suptitle(
        f't-SNE (perplexity={perplexity} clusters={num_clusters})', y=0.92, fontsize=13)
    plt.legend(loc='best', frameon=True)
    plt.show()


@click.command(short_help='Fetches vectors from Pinecone specified by the vectors\' ids.')
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=DEFAULT_REGION)
@click.option('--vector_ids', help='Vector ids to fetch')
@click.option('--pretty', is_flag=True, help='Pretty print output.')
@click.option('--namespace', help='Namespace within the index to search.', default="")
@click.argument('pinecone_index_name')
def fetch(pinecone_index_name: str, apikey: str, region: str, vector_ids: str, namespace: str = "", pretty: bool = True) -> None:
    """ Fetch queries from Pinecone by vector_ids 

        \b    
        Example:
        % ./pinecli.py fetch lpfactset --vector_ids="05b4509ee655aacb10bfbb6ba212c65c"
    """
    index = _pinecone_init(apikey, region, pinecone_index_name)
    parsed_ids = [x.strip() for x in vector_ids.split(",")]
    fetch_response = index.fetch(ids=parsed_ids, namespace=namespace)
    if pretty:
        click.echo(fetch_response)
    else:
        click.echo(fetch_response)


@click.command(short_help='Extracts text from url arg, vectorizes w/ openai embedding api, and upserts to Pinecone.')
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.option('--namespace', help='Pinecone index namespace', default='', show_default=True)
@click.option("--debug", is_flag=True, show_default=True, default=False, help="Output debug to stdout.")
@click.argument('pinecone_index_name')
@click.argument('vector_literal')
def upsert(pinecone_index_name: str, apikey: str, region: str, vector_literal: str, namespace: str = "", debug: bool = False) -> None:
    """ 
    Upserts vectors into the index <PINECONE_INDEX_NAME> by using the <VECTOR_LITERAL> which is a string representation of a list of tuples.
    Note the literal is quoted.

    \b
    Example:
    % ./pinecli.py upsert upsertfile "[('vec1', [0.1, 0.2, 0.3], {'genre': 'drama'}), ('vec2', [0.2, 0.3, 0.4], {'genre': 'action'}),]" --debug 
    """
    index = _pinecone_init(apikey, region, pinecone_index_name)
    if debug:
        click.echo(f"Will upload vectors as {literal_eval(vector_literal)}")
    resp = index.upsert(vectors=literal_eval(
        vector_literal), namespace=namespace)
    click.echo(resp)


@click.command(short_help='Updates the index based on the given id passed in.')
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.option('--namespace', help='Pinecone index namespace', default='', show_default=True)
@click.option("--debug", is_flag=True, show_default=True, default=False, help="Output debug to stdout.")
@click.option("--metadata", help="The update to the metadata values.", default='', show_default=True)
@click.argument("id")
@click.argument('pinecone_index_name')
@click.argument('vector_literal')
def update(pinecone_index_name: str, apikey: str, region: str, id: str, vector_literal: str, metadata: str, namespace: str, debug: bool = False):
    """ Updates the index <PINECONE_INDEX_NAME> with id <ID> and vector values <VECTOR_LITERAL> """
    index = _pinecone_init(apikey, region, pinecone_index_name)
    if metadata:
        resp = index.update(id=id, values=literal_eval(
            vector_literal), set_metadata=literal_eval(metadata), namespace=namespace)
    else:
        resp = index.update(id=id, values=literal_eval(
            vector_literal), namespace=namespace)
    if debug:
        click.echo(resp)


@click.command(short_help='Extracts text from url arg, vectorizes w/ openai embedding api, and upserts to Pinecone.')
@click.option('--apikey', help='Pinecone API Key')
@click.option('--openaiapikey', required=True, help='OpenAI API Key')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.option("--debug", is_flag=True, show_default=True, default=False)
@click.option("--namespace", help='Namespace to store the generated vectors', default='', show_default=True)
@click.option("--metadata_content_key", help="The key used to store the page content in the metadata with.", show_default=True, default="content")
@click.option("--window", help='Number of sentences to combine in one embedding (vector).', show_default=True, default=10)
@click.option("--other_meta", help="Other meta to merge w/ the metadata_content_key", show_default=True, default="{}")
@click.option("--stride", help='Number of sentences to stride over to create overlap', show_default=True, default=4)
@click.argument('url')
@click.argument('pinecone_index_name')
def upsert_webpage(pinecone_index_name, apikey, namespace, openaiapikey, metadata_content_key, other_meta, region, url, window, stride, debug) -> None:
    """ Upserts vectors into the index <PINECONE_INDEX_NAME> using the openai embeddings api.  You will need your api key for openai and specify it using --openapikey """
    pinecone_index = _pinecone_init(apikey, region, pinecone_index_name)
    if openaiapikey is None or openaiapikey == "":
        raise ValueError(
            "You need to specify an OpenAI API key using --openaiapikey")

    html = requests.get(url).text
    html = _text_from_html(html)
    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(html)
    sentences = list(filter(None, sentences))

    new_data = []

    text = ''
    for i in tqdm(range(0, len(sentences), stride)):
        i_end = min(len(sentences)-1, i+window)
        if sentences[i] == sentences[i_end]:
            continue
        text = ' '.join(sentences[i:i_end]).strip()
        # create the new merged dataset
        if debug:
            click.echo(f"Text is: {text}")
        if text != "":
            new_data.append(text)
    new_data.append(sentences[-1])
    new_data = list(filter(None, new_data))
    if debug:
        print(*new_data, sep="\n")

    batch_size = 10  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(new_data), batch_size)):
        i_end = min(len(new_data), i+batch_size)
        meta_batch = new_data[i:i_end]
        ids_batch = [hashlib.md5(x.encode('utf-8')).hexdigest()
                     for x in meta_batch]
        res = _get_openai_embedding(openaiapikey, meta_batch)
        embeds = [record['embedding'] for record in res['data']]
        new_meta_batch = []
        for x in meta_batch:
            d = {metadata_content_key: x}
            d.update(literal_eval(other_meta))
            new_meta_batch.append(d)
        to_upsert = list(zip(ids_batch, embeds, new_meta_batch))
        rv = pinecone_index.upsert(vectors=to_upsert, namespace=namespace)
        if debug:
            click.echo(rv)


@click.command(short_help='Shows a preview of vectors in the <PINECONE_INDEX_NAME>')
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=DEFAULT_REGION)
@click.option('--topk', '--numrows', 'topk', default=10, help='The number of rows to show')
@click.option('--random_dims', is_flag=True, help='Flag to have query vector dims be random.  Default will be 0.0')
@click.option('--include-values', help='Should we return the vectors', show_default=True, default=True)
@click.option('--namespace',  default="", help='Namespace to select results from')
@click.option('--include-meta', help='Whether to include the metadata values', default=False, show_default=True)
@click.option('--expand-meta', help='Whether to fully expand the metadata returned.', is_flag=True, show_default=True, default=False)
@click.option('--print-table', help='Display the output as a pretty table.', is_flag=True, show_default=True, default=False)
@click.argument('pinecone_index_name')
def head(pinecone_index_name: str, apikey: str, region: str, topk: int = 10, random_dims: int = 0, namespace: str = "",
         include_meta: bool = False, expand_meta: bool = False, include_values: bool = True, print_table: bool = False) -> None:
    """ Shows a preview of vectors in the <PINECONE_INDEX_NAME> with optional numrows (default 10) 
    \b
        Example 1:

        % ./pinecli.py head lpfactset --include-meta=true --include-values=true 
        {'matches': [{'id': 'ae23d7574c19cea0b3479c93858a3ee3',
              'metadata': {'content': 'Oded K. R&D Group Lead          Why '
                                      'Pinecone Fast, fresh, and filtered '
                                      'Python client. Scalable Scale from zero '
                                      'to billions of items, with no downtime '
                                      'and minimal latency impact.'},
              'score': 0.0,
              'sparseValues': {},
              'values': [0.010676071,

    \b
        Example 2: (printing results with a table)

        % tim@yoda pinecone-cli % ./pinecli.py head pageuploadtest --include-values=True  --include-meta=True --namespace=test  --print-table --topk=3
                                                                         🌲 pageuploadtest ns=(test) Index Results                                                                         
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
        ┃                               ID ┃ NS   ┃ Values                         ┃                                                                                                 Meta ┃ Score ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
        │ 25d4d1444f9b2942440ce22e026c2a06 │ test │ -0.00443430152,-0.0156992543,0 │ {'content': 'Sign Up for Free  or contact us Use Cases What can you do with vector search ? Once you │   0.0 │
        │ d0bdfbee942fadf531b1feea5b909217 │ test │ 0.0214423928,-0.0283056349,0.0 │  {'content': "Easy to use Get started on the free plan with an easy-to-use API or the Python client. │   0.0 │
        │ ae23d7574c19cea0b3479c93858a3ee3 │ test │ 0.0106426,-0.000842177891,-0.0 │ {'content': "Oded K. R&D Group Lead          Why Pinecone Fast, fresh, and filtered vector search. F │   0.0 │
        └──────────────────────────────────┴──────┴────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┴───────┘
    """
    pinecone_index = _pinecone_init(apikey, region, pinecone_index_name)
    res = pinecone_index.describe_index_stats()
    dims = res['dimension']
    if random_dims:
        dims = [random.random() for _ in range(dims)]
    else:
        dims = [0.0 for _ in range(dims)]
    resp = pinecone_index.query(vector=dims, top_k=topk, namespace=namespace,
                                include_metadata=include_meta, include_values=include_values)
    if print_table:
        _print_table(resp, pinecone_index_name, namespace,
                     include_meta, include_values, expand_meta)
    else:
        click.echo(resp)


@click.command(short_help='Creates a Pinecone Index.')
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=DEFAULT_REGION)
@click.option('--dims', help='Number of dimensions for this index', type=click.INT, required=True)
@click.option('--metric', help='Distance metric to use.', required=True, default='cosine')
@click.option('--pods', help='Number of pods', default=1, show_default=True, type=click.INT)
@click.option('--replicas', help='Number of replicas', default=1, show_default=True, type=click.INT)
@click.option('--shards', help='Number of shards', default=1, show_default=True, type=click.INT)
@click.option('--pod-type', help='Type of pods to create.', required=True)
@click.option('--metadata_config', help='Metadata config to use.', default="", show_default=True)
@click.option('--source_collection', help='Source collection to create index from', default="")
@click.argument('pinecone_index_name')
def create_index(pinecone_index_name, apikey, region, dims, metric, pods, replicas, shards, pod_type, metadata_config, source_collection):
    """ Creates the Pinecone index named <PINECONE_INDEX_NAME> """
    if len(pinecone_index_name) > 45 or len(pinecone_index_name) < 1:
        click.echo(
            "Pinecone index name must be between 1 and 45 characters.  You entered: " + pinecone_index_name)
        sys.exit(-1)
    index = _pinecone_init(apikey, region, pinecone_index_name)
    m_config = literal_eval(metadata_config) if metadata_config else {}

    resp = pinecone.create_index(pinecone_index_name, dimension=dims, metric=metric,
                                 pods=pods, replicas=replicas, shards=shards, pod_type=pod_type, metadata_config=m_config, source_collection=source_collection)
    click.echo(resp)


def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


@click.command(short_help='Upserts a vector(s) with random dimensions into the specified vector.')
@click.option('--apikey',  help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=DEFAULT_REGION)
@click.argument('pinecone_index_name')
@click.option('--num_vectors', '--num_rows', '--numrows', type=click.INT)
@click.option('--debug', is_flag=True, default=False, show_default=True)
@click.option('--num_vector_dims', type=click.INT)
def upsert_random(pinecone_index_name, apikey, region, num_vectors, num_vector_dims, debug) -> None:
    """ Upserts random vectors and dimension values using num_vectors (rows) and num_vector_dims (number of dims per vector). IDs for the example vectors will be of the form \'id-{rownum}\' """
    index = _pinecone_init(apikey, region, pinecone_index_name)

    # Example generator that generates many (id, vector) pairs
    example_data_generator = map(lambda i: (
        f'id-{i}', [random.random() for _ in range(num_vector_dims)]), range(num_vectors))

    # Upsert data with 100 vectors per upsert request
    batch_size = 100
    for ids_vectors_chunk in tqdm(chunks(example_data_generator, batch_size=batch_size), total=num_vectors/batch_size):
        rv = index.upsert(vectors=ids_vectors_chunk)
    if debug:
        click.echo(rv)


@click.command(short_help='Upserts a file (csv) into the specified index.')
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=DEFAULT_REGION)
@click.option('--batch_size', help='Number vectors to upload per batch', default=100, type=click.INT)
@click.option('--namespace', default='')
@click.option("--debug", is_flag=True, show_default=True, default=False, help="Output debug to stdout.")
@click.argument('vector-file', )
@click.argument('pinecone_index_name')
@click.argument('colmap')
def upsert_file(pinecone_index_name, apikey, region, vector_file, batch_size, colmap, namespace, debug):
    colmap = literal_eval(colmap)
    if (('id' not in colmap) or ('vectors' not in colmap)):
        click.echo(
            "Missing 'id' or 'vectors' keys in mapping of CSV file. Check header definitions.")
        sys.exit(-1)

    reverse_col_map = dict(reversed(list(colmap.items())))
    index = _pinecone_init(apikey, region, pinecone_index_name)

    def convert(item):
        item = item.strip()  # remove spaces at the end
        item = item[1:-1]    # remove `[ ]`
        return list(map(float, item.split(',')))

    usecols = [reverse_col_map['id'], reverse_col_map['vectors']]
    converters = {reverse_col_map['vectors']: convert}

    if 'metadata' in colmap:
        csv_meta_col_name = reverse_col_map['metadata']
        usecols.append(reverse_col_map['metadata'])
        converters[csv_meta_col_name] = literal_eval

    for chunk in pd.read_csv(vector_file, chunksize=batch_size, index_col=False, usecols=usecols, converters=converters):
        v = chunk.to_records(index=False).tolist()
        rv = index.upsert(vectors=v, namespace=namespace)
        if debug:
            click.echo(rv)


@click.command(short_help='Lists the indexes for your api key.')
@click.option('--apikey', help='API Key')
@click.option('--print-table', is_flag=True, default=False)
@click.argument('region', default=DEFAULT_REGION)
def list_indexes(apikey: str, region: str, print_table: bool = False) -> None:
    """ List all Pinecone indexes for the given api key. """
    index = _pinecone_init(apikey, region)
    res = pinecone.list_indexes()
    if not print_table:
        click.echo('\n'.join(res))
    else:
        table = Table(
            title=f"🌲 Indexes")
        table.add_column("Index Name", justify="right",
                         style="cyan", no_wrap=True)
        table.add_column("Dimensions", justify="right",
                         style="cyan", no_wrap=True)
        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Replicas", justify="right",
                         style="cyan", no_wrap=True)
        table.add_column("Pods", justify="right", style="cyan", no_wrap=True)
        table.add_column("Pod Type", justify="right",
                         style="cyan", no_wrap=True)
        table.add_column("Shards", justify="right", style="cyan", no_wrap=True)
        table.add_column("Ready", justify="right", style="cyan", no_wrap=True)
        table.add_column("State", justify="right", style="cyan", no_wrap=True)

        for index in res:
            desc = pinecone.describe_index(index)
            state = str(desc.status['state'])
            ready = str(desc.status['ready'])
            ready_formatted = f"[bold green]{ready}[/bold green]" if ready == "True" else f"[bold yellow]{ready}[/bold yellow]"
            table.add_row(index, f"{int(desc.dimension)}", desc.metric, str(desc.replicas), str(
                desc.pods), desc.pod_type, str(desc.shards), ready_formatted, desc.status['state'])

        console = Console()
        console.print(table)


@click.command(short_help='Describes an index.')
@click.option('--apikey')
@click.argument('pinecone_index_name', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
def describe_index(apikey: str, pinecone_index_name: str, region: str = DEFAULT_REGION) -> None:
    """ Describe a Pinecone index with given index_name. """
    index = _pinecone_init(apikey, region, pinecone_index_name)
    desc = pinecone.describe_index(pinecone_index_name)
    click.echo("\n".join([f"Name: {desc.name}", f"Dimensions: {int(desc.dimension)}",
                          f"Metric: {desc.metric}", f"Pods: {desc.pods}", f"PodType: {desc.pod_type}", f"Shards: {desc.shards}",
                          f"Replicas: {desc.replicas}", f"Ready: {desc.status['ready']}", f"State: {desc.status['state']}"
                          f"Metaconfig: {desc.metadata_config}", f"Sourcecollection: {desc.source_collection}"]))


@click.command(short_help='Configures the given index to have a pod type.')
@click.option('--apikey', help='Pinecone API Key')
@click.argument('pinecone_index_name', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.option('--pod_type', required=True, help='Type of pod to use')
def configure_index_pod_type(apikey, pinecone_index_name, region, pod_type):
    """ Configure the pod type for a given index_name. """
    _pinecone_init(apikey, region)
    pinecone.configure_index(pinecone_index_name, pod_type=pod_type)


@click.command(short_help='Configures the number of replicas for a given index.')
@click.option('--apikey')
@click.argument('pinecone_index_name', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.option('--num_replicas', required=True, help='Number of replicas to use.')
def configure_index_replicas(apikey, pinecone_index_name, region, num_replicas):
    """ Configure the number of replicas for an index. """
    _pinecone_init(apikey, region)
    pinecone.configure_index(pinecone_index_name, replicas=num_replicas)


@click.command(short_help='Minimizes everything for a cluster to lowest settings.')
@click.option('--apikey')
@click.argument('pinecone_index_name', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
def minimize_cluster(apikey, pinecone_index_name, region):
    """ Minimizes a cluster to lowest settings index. """
    _pinecone_init(apikey, region)
    pinecone.configure_index(pinecone_index_name, replicas=1, pod_type='s1.x1')


@click.command(short_help='Creates a Pinecone collection from the argument \'source_index\'')
@click.option('--apikey')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.option('--collection_name', help='The name of the collection to create.', required=True)
@click.option('--source_index', help='The name index to create collection from.', required=True)
def create_collection(apikey, region, collection_name, source_index):
    """ Create a Pinecone collection with the given collection_name and source_index. """
    _pinecone_init(apikey, region)
    pinecone.create_collection(collection_name, source_index)


@click.command(short_help='Prints out index stats to stdout.')
@click.option('--apikey')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.argument('pinecone_index_name', required=True)
def describe_index_stats(apikey, region, pinecone_index_name):
    """ Show the stats for index with name <PINECONE_INDEX_NAME>. Note that if the index has several namespaces, those will be broken out. 

    \b
    Example:
    % ./pinecli.py describe-index-stats lpfactset
    """
    index = _pinecone_init(apikey, region, pinecone_index_name)
    res = index.describe_index_stats()
    console = Console()
    click.echo(f"Dimensions: {res['dimension']}")
    click.echo(f"Vectors: {res['total_vector_count']}")
    click.echo(f"Index_Fullness: {res['index_fullness']}")
    ns_data = res['namespaces']
    console.print("Namespace data:", style="b")
    for ns in ns_data.keys():
        click.echo(f"\t{ns}: {ns_data[ns]['vector_count']}")


@click.command(short_help='Lists collections for the given apikey.')
@click.option('--apikey')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
def list_collections(apikey, region):
    """ List Pinecone collections with the given api key """
    _pinecone_init(apikey, region)
    res = pinecone.list_collections()
    print(*res, sep='\n')


@click.command(short_help='Describes a collection.')
@click.option('--apikey')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.argument('collection_name', required=True)
def describe_collection(apikey, region, collection_name):
    """ Describe the collection described by <COLLECTION_NAME> """
    _pinecone_init(apikey, region)
    desc = pinecone.describe_collection(collection_name)
    click.echo("\n".join([f"Name: {desc.name}", f"Dimensions: {int(desc.dimension)}",
                          f"Vectors: {int(desc.vector_count)}", f"Status: {desc.status}", f"Size: {desc.size}"]))


@click.command(short_help="Delete all vectors (note separate command [delete-index] can completely delete an index)")
@click.option('--apikey')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.option('--namespace', default='')
@click.argument('pinecone_index_name')
def delete_all(pinecone_index_name, apikey, region, namespace):
    """ Delete all vectors from the index with optional namespace, but doesnt delete the index itself, will simply have 0 vectors.
        If you want to delete the entire index from existence, use 'delete-index'
    """
    index = _pinecone_init(apikey, region, pinecone_index_name)
    delete_response = index.delete(delete_all=True, namespace=namespace)
    click.echo(delete_response)


@click.command(short_help="Deletes a collection.")
@click.option('--apikey')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.option('--collection_name', help='The name of the collection to create.', required=True)
def delete_collection(apikey, region, collection_name):
    """ Delete a collection with the given collection_name """
    _pinecone_init(apikey, region)
    pinecone.delete_collection(collection_name)


@click.command(short_help='Deletes an index.  You will be prompted to confirm.')
@click.option('--apikey')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=DEFAULT_REGION)
@click.option('--disable-safety', is_flag=True, default=False, show_default=True, help='Disables the prompt check to see if you really want to delete')
@click.argument('pinecone_index', required=True)
def delete_index(apikey, region, pinecone_index, disable_safety):
    """ Delete an index with the given pinecone_index name """
    _pinecone_init(apikey, region)
    if not disable_safety:
        value = click.prompt('Type name of index backwards to confirm: ')
        if value != pinecone_index[::-1]:
            click.echo("Index not deleted: reversed index name does not match.")

    pinecone.delete_index(pinecone_index)


[cli.add_command(c) for c in [query, upsert, upsert_file, upsert_random, update, list_indexes, delete_index, create_index, describe_index, upsert_webpage, configure_index_pod_type,
                              configure_index_replicas, create_collection, list_collections, describe_collection, delete_collection, describe_index_stats, fetch, head, version, minimize_cluster, delete_all]]

if __name__ == "__main__":
    cli()