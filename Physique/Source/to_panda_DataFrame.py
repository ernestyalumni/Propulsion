"""
@name to_and_from_panda_DataFrame.py
@file to_and_from_panda_DataFrame.py
@author Ernest Yeung
@email ernestyalumni@gmail.com
@brief Using Pandas (Panda dataframe) for NIST Physical Constants, i.e.
NIST National Institute of Standards and Technology 
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
import numpy as np
import pandas as pd
import re

from decimal import Decimal, InvalidOperation

from .Scraping import (MakeBraeunigAtmosphereDict,
  MakeNASAPlanetaryFactsTable,
  ParseNISTConversionTable)


################################################################################
### NIST Official conversions
### @url from http://physics.nist.gov/cuu/Reference/unitconversions.html
### Appendix Bof NIST Special Publication 811
### are the official values of conversion factors
################################################################################

################################################################################
### NIST Guide to SI
### B.8 Factors for Units Listed Alphabetically
################################################################################

def NISTConversionTable_to_pd_DF(
  subdirectory='../Data/',
  filename="NISTConversionTable_pd_df"):
  """@fn NISTConversionTable_to_pd_DF
  
  @brief run NISTConversionTable_to_pd_DF first to put the panda DataFrame
  saved locally
  """
  
  headers, rows, rows_with_split_tds = ParseNISTConversionTable()()

  DF_conversions = pd.DataFrame(rows_with_split_tds, columns=headers)

  DF_conversions.to_pickle(subdirectory + filename) 

  return DF_conversions


################################################################################
### NASA Planetary Data
################################################################################

def NASAPlanetaryFacts_to_pd_DF(
  subdirectory='../Data/',
  filename="NASAPlanetaryFacts_pd_df"):
  """@NASAPlanetaryFacts_to_pd_DF

  @brief Make a pandas (panda? singular or plural?) DataFrame from html table 
  of NASA Planetary Fact Sheet
  """
  tbl = MakeNASAPlanetaryFactsTable()()

  # preprocess, data wrangle planets header row
  tbl[0][0] = 'Planet'

  # clean the unicode for each planet
  # cf. http://stackoverflow.com/questions/15321138/removing-unicode-u2026-like-characters-in-a-string-in-python2-7
  for i in range(1, len(tbl[0])):
      tbl[0][i] = tbl[0][i].encode('ascii', 'ignore')
  
  # take the "transpose" of the planetary data
  # cf. http://stackoverflow.com/questions/6473679/python-list-of-lists-transpose-without-zipm-thing
  tbl = list(map(list, zip(*tbl)))
  
  # strip the footnotes * character
  #tbl = [[entry.replace("*","") for entry in row] for row in tbl]
  
  # detach header from data
  header = tbl[0]
  tbl_data = tbl[1:]

  # preprocess data from strings to specific types (decimals and integers)
  cleaned_tbl = []

  for row in tbl_data:

      row_in_decimals = []

      for j in range(1, 18):

          try:
              row_in_decimals.append(Decimal(row[j].replace(",","")))
          except decimal.InvalidOperation:
              row_in_decimals.append(None)        

      cleaned_tbl.append([row[0],] + row_in_decimals + [int(row[18]),] + row[19:])

  # pandas DataFrame of Planetary Fact Sheet
  DF_planetary_facts = pd.DataFrame(cleaned_tbl, columns=header)
  DF_planetary_facts.to_pickle(subdirectory + filename)

  return DF_planetary_facts


################################################################################
### Rocket & Space Technology
### Robert A. Braeunig
### @url http://www.braeunig.us/space/index.htm
################################################################################

def BraeunigAtmosphere_to_pd_DF(
  subdirectory="../Data/",
  filename="BraeunigAtmosphere_pd_df"):

  standard_atm_dict = MakeBraeunigAtmosphereDict()()

  DF_standard_atm  = \
    pd.DataFrame(
      standard_atm_dict["data"],
      columns=standard_atm_dict["header"])
  
  DF_standard_atm.to_pickle(subdirectory + filename)
  return DF_standard_atm
