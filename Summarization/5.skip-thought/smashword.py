"""Book scraping script for smashwords.com.

Usage: python smashwords.py [scrape_link] [output_dir (defaults to data/books)]
"""

import os
import re
import sys

from bs4 import BeautifulSoup
import requests


def browse(url):
  """Retrieve the server response contents of the given URL."""
  # A cookie is required to allow books with adult content to be served.
  return requests.get(url, cookies={"adultOff": "no"}).text


def to_filename(s):
  """Convert the given string to a valid filename."""
  s = str(s).strip().replace(' ', '_')
  return re.sub(r'(?u)[^-\w.]', '', s)


if __name__ == '__main__':

  write_dir = 'books'
  if len(sys.argv) > 2:
    write_dir = sys.argv[2]

  count = 0
  num_downloaded = 0
  while True:
    res = browse((sys.argv[1] + '/{}').format(count))
    soup = BeautifulSoup(res, 'html.parser')
    for div in soup.find_all('div', {'class': 'library-book'}):
      
      # Detect language
      language_html = div.find('div', {'class': 'subnote'})
      language_html = language_html.find_all('span', {'class': 'text-nowrap'})
      language_html = ''.join(map(lambda tag: tag.get_text(), language_html))
      
      if 'english' in language_html.lower():
      
        # Get title and download link
        link_html = div.find('a', {'class': 'library-title'})
        title = link_html.get_text()
        link = link_html.get('href').split('/')
        link[-2] = 'download'
        link.append('6')  # text file format
        
        download = browse('/'.join(link))
        if not download.startswith('<!DOCTYPE html>'):
          num_downloaded += 1
          print(num_downloaded, title, sep='\t')
          with open(os.path.join(write_dir, to_filename(title)), 'w') as f:
            f.write(download)
    
    count += 20

