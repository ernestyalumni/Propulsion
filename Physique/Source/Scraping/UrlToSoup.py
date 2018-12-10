"""
@name UrlsToSoup.py
@file UrlsToSoup.py
@author Ernest Yeung
@email ernestyalumni@gmail.com
@brief From constant, given URL strings, to BeautifulSoup objects, and
optionally to tables
@ref
@details
@copyright If you find this code useful, feel free to donate directly and easily
at this direct PayPal link: 

https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 

which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.
Otherwise, I receive emails and messages on how all my (free) material on
physics, math, and engineering have helped students with their studies, and I
know what it's like to not have money as a student, but love physics (or math,
sciences, etc.), so I am committed to keeping all my material open-source and
free, whether or not sufficiently crowdfunded, under the open-source MIT
license: feel free to copy, edit, paste, make your own versions, share, use as
you wish.    

Peace out, never give up! -EY
"""

import decimal
import re
import requests
from urllib.request import urlopen

from decimal import Decimal
from bs4 import BeautifulSoup

################################################################################
### NIST - National Institute of Standards and Technology
################################################################################

class ScrapedSoup(object):
  """@class ScrapedSoup
  @brief Encapsulates as an object all results from scraping with requests and
  BeautifulSoup

  @example python -i UrlsToSoup.py
  >>> mainNISTcuu = ScrapedSoup(NISTCUU_URL)
  >>> NISTconvBS = ScrapedSoup(NISTCONValpha)
  """
  def __init__(self, url):
    self.url = url
    self.req = requests.get(url)
    self.soup = BeautifulSoup(self.req.content, "lxml")
    self.req.close()

################################################################################
### NIST Reference on Constants, Units, and Uncertainty mainpage
################################################################################

class ScrapedFundamentalPhysicalConstants(ScrapedSoup):

  URL = "http://physics.nist.gov/cuu/Constants/Table/allascii.txt"

  def __init__(self):
    ScrapedSoup.__init__(self, self.URL)

################################################################################
### On looking at mainNISTcuu.soup.find_all("a") and your favorite Web Inspector
### of the NIST cuu webpage, NISTCUU_URL, one realizes that to get to Frequently
### Used Constants or All values, one does a query.  Let's grab all values in
### ascii.
### Do this with FPCasciitbl = ScrapedSoup(FPCascii) if you want to use requests
### Do this with urllib as follows, using urllib.request
################################################################################

def retrieve_file(
  url=ScrapedFundamentalPhysicalConstants.URL,
  subdirectory="../../Data/",
  filename="allFundamentalPhysicalConstants_ascii.txt"):
  """@fn retrieve_file
  """
  with open(subdirectory + filename, 'wb') as target_file:
    u = urlopen(url)
    target_file.write(u.read())
    target_file.close()

  return u

################################################################################
### At this point, running retrieve_file() should put allascii.txt into the
### rawdata subdirectory.
### You can open up allascii.txt and see you have all the Fundamental Physical
### Constants (!!!).
### Time to parse this table:
################################################################################

def line_reading(
  fullfilename='../../Data/rawdata/allFundamentalPhysicalConstants_ascii.txt'):
    opened_file = open(fullfilename, 'rb')
    lines = openedfile.read().splitlines()
    lines = [line for line in lines if line != '']
    return lines

def scraping_ascii_text(fullfilename=
  '../../Data/allFundamentalPhysicalConstants_ascii.txt'):
  """@fn scraping_ascii_text
  """
  lines  = line_reading(fullfilename)
  title  = lines[0].strip()
  src    = lines[1].strip()
  header = lines[2].split()

  # cf. http://stackoverflow.com/questions/12866631/python-split-a-string-with-at-least-2-whitespaces
  # for re

  raw_tbl = []

  for line in lines[4:]:
      raw_tbl.append( re.split(r'\s{2,}', line) )

  tbl = []

  for raw_line in raw_tbl:

      line = []
      line.append(raw_line[0])

      try:
          line.append(Decimal(raw_line[1].replace(" ","")))
      except decimal.InvalidOperation:
#            line.append(Decimal(rawline[1].replace(" ","").replace(".","") ))
          value = raw_line[1].replace(" ","")
          value = ''.join(re.split(r'[.]{2,}',value))

          line.append(Decimal(value))

      try:
          line.append(Decimal(raw_line[2].replace(" ","")))
      except decimal.InvalidOperation:
#            line.append(rawline[2].replace(" ","")
          # EY : 20150823 for SQLAlchemy, instead of "(exact)" use None type
          # None for SQLAlchemy to denote "(exact)" in the Uncertainty column
          line.append(None)

      line.append(raw_line[3])
      tbl.append(line)

  return lines, title, src, header, rawtbl, tbl


################################################################################
### Now you should be able to put this into your favorite database of choice
################################################################################

################################################################################
### NIST Official conversions
### from http://physics.nist.gov/cuu/Reference/unitconversions.html
### Appendix Bof NIST Special Publication 811
### are the official values of conversion factors
################################################################################

################################################################################
### NIST Guide to SI
### B.8 Factors for Units Listed Alphabetically
################################################################################

def parse_table_bodies_from_url(url):
  """@fn parse_table_bodies_from_url
  """

  scraped_soup = ScrapedSoup(url)
  scraped_soup.tables = scraped_soup.soup.find_all("table")
  scraped_soup.tbodies = scraped_soup.soup.find_all("tbody")

  rows = []
  rows_with_split_tds = []

  headers = scraped_soup.tables[0].find_all('tr')[1].find_all('th')
  headers = [element.text.replace(' ', '') for element in headers]

  for tbody in scraped_soup.tbodies:

    # Assume the first two rows are headers
    for row in tbody.find_all('tr')[2:]:

      if row.find_all('td') != [] and row.text != '':
        row_split = row.text.replace("\n", '', 1).split('\n')

        try:
          row_split = [pt.replace(u'\xa0', u' ').strip() for pt in row_split]
        except UnicodeDecodeError as err:
          print(row_split)
          Break
          raise err

        rows.append(row_split)

        if len(row.find_all('td')) == (len(headers) + 1):
          rows_with_split_tds.append(row.find_all('td'))

  cleaned_rows = []

  for row in rows_with_split_tds:
    cleaned_row = []
    cleaned_row.append(row[0].text.strip())
    cleaned_row.append(row[1].text.strip())

    value = \
      (row[2].text + \
        row[3].text).strip().replace( \
          u'\xa0', ' ').replace(u'\n', ' ').replace(' ', '')
      
    cleaned_row.append(Decimal(value))

    cleaned_rows.append(cleaned_row)

  rows_with_split_tds = cleaned_rows

  return headers, rows, rows_with_split_tds


class ParseNISTConversionTable(object):
  """@class ParseNISTConversionTable
  """

  NIST_CONVERSION_ALPHA = \
    "https://www.nist.gov/physical-measurement-laboratory/nist-guide-si-appendix-b8"

  def __call__(self):
    """@brief This construct allows objects to be called as functions in Python
    """

    return parse_table_bodies_from_url(self.NIST_CONVERSION_ALPHA)


################################################################################
### NASA Planetary Fact Sheet (Metric)
###
### Author/Curator:
### Dr. David R. Williams, dave.williams@nasa.gov
### NSSDCA, Mail Code 690.1
### NASA Goddard Space Flight Center
### Greenbelt, MD 20771
### +1-301-286-1258
### NASA Official: Ed Grayzeck, edwin.j.grayzeck@nasa.gov
### Last Updated: 17 July 2015, DRW
################################################################################

def parse_table_from_url(url):
  """@fn parse_table_from_url 
  """
  scraped_soup = ScrapedSoup(url)
  table = scraped_soup.soup.find("table")

  table_data = []

  for row in table.find_all("tr"):
    if row.find_all("td") != []:
      table_data.append([element.text for element in row.find_all("td")])

  table_data.pop()

  return table_data


class MakeNASAPlanetaryFactsTable(object):

  # NASA Planetary Data URL
  NASA_PLANETARY_FACTS_URL = \
    "http://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html"

  def __call__(self):

    return parse_table_from_url(self.NASA_PLANETARY_FACTS_URL)

# Now, go to to_panda_DataFrame.py


################################################################################
### Rocket & Space Technology
### Robert A. Braeunig
### @url http://www.braeunig.us/space/index.htm
################################################################################

def parse_tables_from_url(url):
  """@fn parse_tables_from_url
  """

  scraped_soup = ScrapedSoup(url)
  tables = scraped_soup.soup.find_all("table")

  headers_data = []
  tables_data = []

  for table in tables:

    header_data = []
    table_data = []

    for row in table.find_all('tr'):

      if row.find_all('td') != []:
        table_data.append([element.text for element in row.find_all('td')])
      elif row.find_all('th') != []:
        header_data.append([element.text for element in row.find_all('th')])

    tables_data.append(table_data)
    headers_data.append(header_data)

  return tables, tables_data, headers_data


class MakeBraeunigAtmosphereDict(object):

  Braeunig_ATMOS_URL = "http://www.braeunig.us/space/atmos.htm"

  def __call__(self):

    tables, tables_data, headers_data = \
      parse_tables_from_url(self.Braeunig_ATMOS_URL)

    # Physical Properties of U.S. Standard Atmosphere, 1976 in SI Units
    try:
      standatm_FOOTNOTE = tables_data[1].pop()
    except IndexError as error:
      print("Empty table data", error)

    standatm_DATA  = \
      [[Decimal(ele.replace(',','')) for ele in row] for row in tables_data[1]]
    standatm_TITLE = headers_data[1][0][0]
    standatm_HDR   = headers_data[1][1]
    standatm_dict = \
      {"footnote" : standatm_FOOTNOTE, \
        "data" : standatm_DATA, \
        "title" : standatm_TITLE, \
        "header" : standatm_HDR}

    return standatm_dict

# Now, go to to_panda_DataFrame.py
