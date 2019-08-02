#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import argparse
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_OUTPUT = 'output.txt'
DEFAULT_INTERVAL = 5.0  # interval between requests (seconds)
DEFAULT_ARTICLES_LIMIT = 1  # total number articles to be extrated
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'

visited_urls = set()  # all urls already visited, to not visit twice
pending_urls = []  # queue


def scrap(base_url, article, output_file):
    """Represents one request per article"""

    full_url = base_url + article
    try:
        r = requests.get(full_url, headers={'User-Agent': USER_AGENT})
    except requests.exceptions.ConnectionError:
        print("Check your Internet connection")
        return
    if r.status_code not in (200, 404):
        print("Failed to request page (code {})".format(r.status_code))
        return

    soup = BeautifulSoup(r.text, 'html.parser')
    content = soup.find('div', {'id':'mw-content-text'})

    # add new related articles to queue
    # check if are actual articles URL
    for a in content.find_all('a'):
        href = a.get('href')
        if not href:
            continue
        if href[0:6] != '/wiki/':  # allow only article pages
            continue
        elif ':' in href:  # ignore special articles e.g. 'Special:'
            continue
        elif href[-4:] in ".png .jpg .jpeg .svg":  # ignore image files inside articles
            continue
        elif base_url + href in visited_urls:  # already visited
            continue
        if href in pending_urls:  # already added to queue
            continue
        pending_urls.append(href)

    # skip if already added text from this article, as continuing session
    if full_url in visited_urls:
        return
    visited_urls.add(full_url)

    parenthesis_regex = re.compile('\(.+?\)')  # to remove parenthesis content
    citations_regex = re.compile('\[.+?\]')  # to remove citations, e.g. [1]

    # get plain text from each <p>
    p_list = content.find_all('p')
    with open(output_file, 'w') as fout:
        for p in p_list:
            text = p.get_text().strip()
            text = parenthesis_regex.sub('', text)
            text = citations_regex.sub('', text)
            if text:
                fout.write(text + '\n\n')  # extra line between paragraphs


def crawler(initial_url, articles_limit=DEFAULT_ARTICLES_LIMIT, output_file=DEFAULT_OUTPUT):
    """ Main loop, single thread """

    base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(initial_url))
    initial_url = initial_url[len(base_url):]
    article_format = initial_url.replace('/wiki/', '')[:35]
    scrap(base_url, initial_url, output_file)