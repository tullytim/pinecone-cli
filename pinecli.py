#!/usr/bin/env python3
# encoding: utf-8

import click
import hashlib
import itertools
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openai
import nltk.data
import pinecone
import urllib.request

from numpy import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm
from pprint import pp
from bs4 import BeautifulSoup
from bs4.element import Comment
from time import sleep

default_region = 'us-west1-gcp'
openai_embed_model = "text-embedding-ada-002"

REGION_HELP = 'Pinecone cluster region'

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

def get_openai_embedding(apikey, data, batch=True):
    openai.api_key = apikey
    try:
        res = openai.Embedding.create(input=data, engine=openai_embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=data, engine=openai_embed_model)
                done = True
            except Exception as e: 
                print(e)
                pass
    return res

@click.group()
def cli():
    pass

@click.command()
@click.option('--apikey', required=True, help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
@click.option('--include_values', help='Should we return the vectors', show_default=True, default=True)
@click.option('--topk', type=click.INT, show_default=True, default=10, help='Top K number to return')
@click.option('--namespace',  default="", help='Namespace to select results from')
@click.option('--expand-meta', help='Whether to fully expand the metadata returned.', is_flag=True, show_default=True, default=False)
@click.option('--show_tsne', default=False)
@click.argument('pinecone_index_name')
@click.argument('query_vector')
def query(pinecone_index_name, apikey, query_vector, region, topk, include_values, expand_meta, namespace, show_tsne):
    click.echo(f'Query the database {apikey} {query_vector}')
    pinecone.init(api_key=apikey, environment=region)
    pinecone_index = pinecone.Index(pinecone_index_name)
    query_vector = [random.random() for i in range(1536)]
    res = pinecone_index.query(query_vector, top_k=topk, include_metadata=True, include_values=include_values, namespace=namespace)
    table = Table(title=f"🌲 {pinecone_index_name} ns=({namespace}) Index Results")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Values", style="magenta")
    table.add_column("Meta", justify="right", style="green")

    for row in res.matches:
        metadata = str(row['metadata'])
        metadata = metadata[:200] if not expand_meta else metadata
        table.add_row(row['id'], metadata, str(row['score']))
        #print(row['values'])
        #table.add_row(row['id'], "".join(x for x in row['values']), str(row['score']))

    console = Console()
    console.print(table)
    if(show_tsne):
        show_tsne_plot(res.matches)
        
def show_tsne_plot(results):    
    res2 = [np.array(v['values']) for v in results]
    print(len(res2))
    df = pd.DataFrame({'embeds':res2})
    matrix = np.vstack(df.embeds)
    
    n_clusters = 4
    kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    df['Cluster'] = kmeans.labels_
    tsne = TSNE(n_components=2, perplexity=2, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]    
    for category, color in enumerate(["purple", "green", "red", "blue"]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        avg_x = xs.mean()
        avg_y = ys.mean()

    plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    plt.title("Clusters identified visualized in language 2d using t-SNE")
    plt.show()


@click.command()
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=default_region)
@click.argument('vector-file', )
@click.argument('pinecone_index_name')
def upsert_file(pinecone_index_name, apikey, region, vector_file):
    click.echo('Upsert the database')
    
@click.command()
@click.option('--apikey', required=True, help='Pinecone API Key')
@click.option('--openaiapikey', required=True, help='OpenAI API Key')
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
@click.option("--debug", is_flag=True, show_default=True, default=False, help="Output debug to stdout.")
@click.argument('url')
@click.argument('pinecone_index_name')
def upsert_webpage(pinecone_index_name, apikey, openaiapikey, region, url, debug):
    html = urllib.request.urlopen(url).read()
    html = text_from_html(html)
    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(html)
    sentences = list(filter(None, sentences))
    """
    new_data = []
    window = 20  # number of sentences to combine
    stride = 4  # number of sentences to 'stride' over, used to create overlap
    for i in tqdm(range(0, len(sentences), stride)):
        i_end = min(len(sentences)-1, i+window)
        if sentences[i] == sentences[i_end]:
            continue
        text = ' '.join(sentences[i:i_end]).strip()
        # create the new merged dataset
        if(text != ""):
            new_data.append(text)
    """
    new_data = []
    window = 10  # number of sentences to combine
    stride = 4  # number of sentences to 'stride' over, used to create overlap
    print(f"Have {len(sentences)}")
    text = ''
    for i in tqdm(range(0, len(sentences), stride)):
        i_end = min(len(sentences)-1, i+window)
        if sentences[i] == sentences[i_end]:
            continue
        text = ' '.join(sentences[i:i_end]).strip()
        # create the new merged dataset
        print(f"Text is: {text}")
        if(text != ""):
            new_data.append(text)
    new_data.append(sentences[-1])     
    new_data = list(filter(None, new_data))
    print(new_data)
    if debug:
        print(*new_data, sep="\n")
    
    embeddings, ids, metadata = [], [], []
    batch_size = 10  # how many embeddings we create and insert at once
    pinecone.init(api_key=apikey, environment=region)
    pinecone_index = pinecone.Index(pinecone_index_name)
    
    for i in tqdm(range(0, len(new_data), batch_size)):
        i_end = min(len(new_data), i+batch_size)
        meta_batch = new_data[i:i_end]
        ids_batch = [hashlib.md5(x.encode('utf-8')).hexdigest() for x in meta_batch]
        res = get_openai_embedding(openaiapikey, meta_batch)
        embeds = [record['embedding'] for record in res['data']]
        meta_batch = [{'content': x} for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        rv = pinecone_index.upsert(vectors=to_upsert)
        print(rv)
      
      
@click.command()
@click.option('--apikey', required=True, help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=default_region, required=True)
@click.option('--dims', help='Number of dimensions for this index', type=click.INT, required=True)
@click.option('--metric', help='Distance metric to use.', required=True)
@click.option('--pods', help='Number of pods', default=1, show_default=True, type=click.INT)
@click.option('--replicas', help='Number of replicas', default=1, show_default=True, type=click.INT)
@click.option('--shards', help='Number of shards', default=1, show_default=True, type=click.INT)
@click.option('--pod-type', help='Type of pods to create.', required=True)
@click.argument('pinecone_index_name')  
def create_index(pinecone_index_name, apikey, region, dims, metric, pods, replicas, shards, pod_type):
    pinecone.init(api_key=apikey, environment=region)


def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))
        
@click.command()
@click.option('--apikey', required=True, help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=default_region)
@click.argument('pinecone_index_name')
@click.option('--num_vector', type=click.INT)
@click.option('--num_vector_dims', type=click.INT)
def upsert_random(pinecone_index_name, apikey, region, num_vector, num_vector_dims):
    pinecone.init(api_key=apikey, environment=region)
    index = pinecone.Index(pinecone_index_name)

    # Example generator that generates many (id, vector) pairs
    example_data_generator = map(lambda i: (f'id-{i}', [random.random() for _ in range(num_vector_dims)]), range(num_vector))

    # Upsert data with 100 vectors per upsert request
    batch_size=100
    for ids_vectors_chunk in tqdm(chunks(example_data_generator, batch_size=batch_size), total=num_vector/batch_size):
        index.upsert(vectors=ids_vectors_chunk)  # Assuming `index` defined elsewhere
    
    
@click.command()
@click.argument('apikey', required=True)
@click.argument('region', default=default_region)
def list_indexes(apikey, region):
    pinecone.init(api_key=apikey, environment=region)
    res = pinecone.list_indexes()
    print('\n'.join(res))
    
@click.command()
@click.option('--apikey', required=True)
@click.argument('index_name', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
def describe_index(apikey, index_name, region):
    pinecone.init(api_key=apikey, environment=region)
    desc = pinecone.describe_index(index_name)
    print(f"Name: {desc.name}")
    print(f"Dimensions: {int(desc.dimension)}")
    print(f"Metric: {desc.metric}")
    print(f"Pods: {desc.pods}")
    print(f"PodType: {desc.pod_type}")
    print(f"Shards: {desc.shards}")
    print(f"Replicas: {desc.replicas}")
    print(f"Ready: {desc.status['ready']}")
    print(f"State: {desc.status['state']}")
    print(f"Metaconfig: {desc.metadata_config}")
    print(f"Sourcecollection: {desc.source_collection}")
    
@click.command()
@click.option('--apikey', required=True)
@click.argument('index_name', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
@click.option('--pod_type', required=True, help='Type of pod to use')
def configure_index_pod_type(apikey, index_name, region, pod_type):
    pinecone.init(api_key=apikey, environment=region)    
    pinecone.configure_index(index_name, pod_type=pod_type)
    
@click.command()
@click.option('--apikey', required=True)
@click.argument('index_name', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
@click.option('--num_replicas', required=True, help='Number of replicas to use.')
def configure_index_replicas(apikey, index_name, region, num_replicas):
    pinecone.init(api_key=apikey, environment=region)    
    pinecone.configure_index(index_name, replicas=num_replicas)

@click.command()
@click.option('--apikey', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
@click.option('--collection_name', help='The name of the collection to create.', required=True)
@click.option('--source_index', help='The name index to create collection from.', required=True)
def create_collection(apikey,region, collection_name, source_index):
    pinecone.init(api_key=apikey, environment=region)    
    pinecone.create_collection(collection_name, source_index)
    
@click.command()
@click.option('--apikey', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
@click.argument('index_name', required=True)
def describe_index_stats(apikey, region, index_name):
    pinecone.init(api_key=apikey, environment=region)    
    index = pinecone.Index(index_name)
    res = index.describe_index_stats()
    console = Console()
    print(f"Dimensions: {res['dimension']}")
    print(f"Vectors: {res['total_vector_count']}")
    print(f"Index_Fullness: {res['index_fullness']}")
    ns_data = res['namespaces']
    console.print("Namespace data:", style="b")
    for ns in ns_data.keys():
        print(f"\t{ns}: {ns_data[ns]['vector_count']}")
    
@click.command()
@click.option('--apikey', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
def list_collections(apikey,region):
    pinecone.init(api_key=apikey, environment=region)    
    res = pinecone.list_collections()
    print(*res, sep='\n')
    
@click.command()
@click.option('--apikey', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
@click.option('--collection_name', help='The name of the collection to create.', required=True)
def describe_collection(apikey,region, collection_name):
    pinecone.init(api_key=apikey, environment=region)    
    desc = pinecone.describe_collection(collection_name)
    print(f"Name: {desc.name}")
    print(f"Dimensions: {int(desc.dimension)}")
    print(f"Vectors: {int(desc.vector_count)}")
    print(f"Status: {desc.status}")
    print(f"Size: {desc.size}")
    
@click.command()
@click.option('--apikey', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
@click.option('--collection_name', help='The name of the collection to create.', required=True)
def delete_collection(apikey, region, collection_name):
    pinecone.init(api_key=apikey, environment=region)    
    desc = pinecone.delete_collection(collection_name)
    
@click.command()
@click.option('--apikey', required=True)
@click.option('--region', help='Pinecone Index Region', show_default=True, default=default_region)
@click.argument('pinecone_index', required=True)
def delete_index(apikey, region, pinecone_index):
    pinecone.init(api_key=apikey, environment=region)    
    value = click.prompt('Type name of index backwards to confirm: ')
    if value == pinecone_index[::-1]:
        pinecone.delete_index(pinecone_index)
    else:
        print("Index not deleted: reversed index name does not match.")
    
cli.add_command(query)
cli.add_command(upsert_file)
cli.add_command(upsert_random)
cli.add_command(list_indexes)
cli.add_command(delete_index)
cli.add_command(create_index)
cli.add_command(describe_index)
cli.add_command(upsert_webpage)
cli.add_command(configure_index_pod_type)
cli.add_command(configure_index_replicas)
cli.add_command(create_collection)
cli.add_command(list_collections)
cli.add_command(describe_collection)
cli.add_command(delete_collection)
cli.add_command(describe_index_stats)

if __name__ == "__main__":
    cli()