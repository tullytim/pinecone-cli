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
@click.argument('url')
@click.argument('pinecone_index_name')
def upsert_webpage(pinecone_index_name, apikey, openaiapikey, region, url):
    html = urllib.request.urlopen(url).read()
    html = text_from_html(html)
    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(html)
    sentences = list(filter(None, sentences))
    
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
            
    for s in new_data:
        print(f'****{s}****')
    
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
        
        
"""
**** UPSERT Random ***
"""
@click.command()
@click.option('--apikey', required=True, help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=default_region)
@click.argument('pinecone_index_name')
@click.argument('number_random_rows')
def upsert_random(pinecone_index_name, apikey, region, number_random_rows):
    click.echo('NOT WORKING YET')
    
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
def list_collections(apikey,region):
    pinecone.init(api_key=apikey, environment=region)    
    res = pinecone.list_collections()
    print('\n'.join(res))
    
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

    
cli.add_command(query)
cli.add_command(upsert_file)
cli.add_command(upsert_random)
cli.add_command(list_indexes)
cli.add_command(describe_index)
cli.add_command(upsert_webpage)
cli.add_command(configure_index_pod_type)
cli.add_command(configure_index_replicas)
cli.add_command(create_collection)
cli.add_command(list_collections)
cli.add_command(describe_collection)

if __name__ == "__main__":
    cli()