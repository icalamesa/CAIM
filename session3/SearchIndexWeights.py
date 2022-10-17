"""
.. module:: SearchIndexWeight

SearchIndex
*************

:Description: SearchIndexWeight

    Performs a AND query for a list of words (--query) in the documents of an index (--index)
    You can use word^number to change the importance of a word in the match

    --nhits changes the number of documents to retrieve

:Authors: bejar
    

:Version: 

:Created on: 04/07/2017 10:56 

"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

import argparse

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q
import numpy as np
from elasticsearch.exceptions import NotFoundError
from elasticsearch_dsl import Search
from elasticsearch.client import CatClient
from elasticsearch_dsl.query import Q
from elasticsearch import Elasticsearch

import math
import functools
import operator
import argparse
__author__ = 'bejar'


beta = 1
alpha = 2
R = 4
nrows = 10

def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])

def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())

def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file:
    :return:
    """

    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)

    tfidfw = {}
    for (t, w),(_, df) in zip(file_tv, file_df):
        #
        idfi = np.log2((dcount/df))
        tfdi = w/max_freq
        tfidfw[t] = tfdi * idfi
        # Something happens here
        #

    return normalize(tfidfw)

def normalize(d):
    s = sum(d.values())
    r = np.sqrt(s)
    norm = {t: d.get(t, 0)/r for t in set(d)}
    return norm

def search_file_by_path(client, index, path):
    """
    Search for a file using its path

    :param path:
    :return:
    """
    s = Search(using=client, index=index)
    q = Q('match', path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id

#
def queryToDict(query):
    dictQuery = {}
    for elem in query:
        if '^' in elem:
            key, value = elem.split('^')
            value = float(value)
        else:
            key = elem
            value = 1.0
        dictQuery[key] = value
        
    return normalize(dictQuery)

def dictToquery(di):
    query = []
    for elem in di:
        q = elem + '^' + str(di[elem])
        query.append(q)
    return query

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, help='Index to search')
    parser.add_argument('--nhits', default=10, type=int, help='Number of hits to return')
    parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')

    args = parser.parse_args()

    index = args.index
    query = args.query
    print(query)
    nhits = args.nhits

    try:
        client = Elasticsearch()
        s = Search(using=client, index=index)

        if query is not None:
            for j in range(0, nrows):
                q = Q('query_string',query=query[0])
                for i in range(1, len(query)):
                    q &= Q('query_string',query=query[i])

                print(query)
                s = s.query(q)
                response = s[0:nhits].execute()

                dictQuery = queryToDict(query)
                docSum = {}

                for r in response:  # only returns a specific number of results
                    file_tw = toTFIDF(client, index, r.meta.id) # tf-idf
                    docSum = {t: docSum.get(t, 0) + file_tw.get(t, 0) for t in set(docSum) | set(file_tw)} # sumem els valors de cada document
                    print(f'ID= {r.meta.id} SCORE={r.meta.score}')
                    print(f'PATH= {r.path}')
                    print(f'TEXT: {r.text[:50]}')
                    print('-----------------------------------------------------------------')
                docSum = {t: docSum.get(t,0)*beta/nhits for t in set(docSum)} # Beta * vector de documents / K
                oldQuery = {t: dictQuery.get(t,0)*alpha for t in set(dictQuery)} # Alpha * query
                query2 = {}
                query2 = {t: docSum.get(t, 0) + oldQuery.get(t, 0) for t in set(docSum) | set(oldQuery)} # alpha * query + beta * vector documents / K
                query2 = sorted(query2.items(), key=operator.itemgetter(1), reverse = True) # ordenem per valor, es converteix en tuples
                query2 = query2[:R] #agafem els R mes relevants
                dictQuery = dict((t, val) for (t, val) in query2) #ho tornem a transformar en un diccionari
                query = dictToquery(normalize(dictQuery))
                print (f"{response.hits.total['value']} Documents")

        else:
            print('No query parameters passed')


    except NotFoundError:
        print(f'Index {index} does not exists')

