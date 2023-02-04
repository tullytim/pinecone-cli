#!/usr/bin/env python3
# encoding: utf-8

import click
import hashlib
import itertools
import json
import openai
import nltk.data
import pinecone
import urllib.request

from numpy import random
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm
from pprint import pp
from bs4 import BeautifulSoup
from bs4.element import Comment


default_region = 'us-west1-gcp'

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

@click.group()
def cli():
    pass

@click.command()
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=default_region)
@click.option('--include_values', help='Should we return the vectors', default=True)
@click.option('--topk', type=click.INT, default=10, help='Top K number to return')
@click.option('--namespace',  default="", help='Namespace to select results from')
@click.option('--expand-meta', help='Whether to fully expand the metadata returned.', default=False)
@click.argument('pinecone_index_name')
@click.argument('query_vector')
def query(pinecone_index_name, apikey, query_vector, region, topk, include_values, expand_meta, namespace):
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


@click.command()
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=default_region)
@click.argument('vector-file', )
@click.argument('pinecone_index_name')
def upsert_file(pinecone_index_name, apikey, region, vector_file):
    click.echo('Upsert the database')
    
    
    
    
    
@click.command()
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=default_region)
@click.argument('url')
@click.argument('pinecone_index_name')
def upsert_webpage(pinecone_index_name, apikey, region, url):
    click.echo('Upsert the database')
    html = urllib.request.urlopen(url).read()
    html = text_from_html(html)
    print(html)
    
    
    
    
    
"""
**** UPSERT Random ***
"""
@click.command()
@click.option('--apikey', help='Pinecone API Key')
@click.option('--region', help='Pinecone Index Region', default=default_region)
@click.argument('pinecone_index_name')
@click.argument('number_random_rows')
def upsert_random(pinecone_index_name, apikey, region, number_random_rows):
    click.echo('Upsert the database')
    
@click.command()
@click.argument('apikey')
@click.argument('region', default=default_region)
def list_indexes(apikey, region):
    pinecone.init(api_key=apikey, environment=region)
    res = pinecone.list_indexes()
    print('\n'.join(res))
    
@click.command()
@click.argument('apikey')
@click.argument('index_name')
@click.argument('region', default=default_region)
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

cli.add_command(query)
cli.add_command(upsert_file)
cli.add_command(upsert_random)
cli.add_command(list_indexes)
cli.add_command(describe_index)
cli.add_command(upsert_webpage)

if __name__ == "__main__":
    cli()