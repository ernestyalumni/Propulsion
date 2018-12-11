"""
@name Physique.py
@file Physique.py
@author Ernest Yeung
@email ernestyalumni@gmail.com
@brief Main module file for Physique 
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
import os
import pandas as pd

from .Source import NIST_fundamental_constants_ascii_to_pd_DF

def get_NIST_fundamental_constants_pd_DF():
  """@fn get_NIST_fundamental_constants_pd_DF
  @brief Retrieve from an already saved ASCII text file a pandas DataFrame.
  """
  #current_directory = os.path.abspath(os.getcwd())

  current_directory = os.path.dirname(__file__)

  Data_subdirectory = "Data/"

  filename = "allFundamentalPhysicalConstants_ascii.txt"

  return NIST_fundamental_constants_ascii_to_pd_DF(os.path.join(
    current_directory, Data_subdirectory) + filename)


FundamentalPhysicalConstants = get_NIST_fundamental_constants_pd_DF()


def get_NIST_conversions_pd_DF():
  """@fn get_NIST_conversions_pd_DF
  @brief Retrieve from an already saved, pickled pandas DataFrame"""

  #current_directory = os.path.abspath(os.getcwd())

  current_directory = os.path.dirname(__file__)

  Data_subdirectory = "Data/"

  filename = "NISTConversionTable_pd_df"

  try:
    return pd.read_pickle(
      os.path.join(current_directory, Data_subdirectory) + filename)
  except FileNotFoundError as error:
    print("File not found; make sure to generate the file: " + error)
    return    


Conversions = get_NIST_conversions_pd_DF()


def get_NASA_planetary_facts_pd_DF():
  """@fn get_NASA_planetary_facts_pd_DF
  @brief Retrieve from an already saved, pickled pandas DataFrame"""

  #current_directory = os.path.abspath(os.getcwd())

  current_directory = os.path.dirname(__file__)

  Data_subdirectory = "Data/"

  filename = "NASAPlanetaryFacts_pd_df"

  try:
    return pd.read_pickle(
      os.path.join(current_directory, Data_subdirectory) + filename)
  except FileNotFoundError as error:
    print("File not found; make sure to generate the file: " + error)
    return    


PlanetaryFacts = get_NASA_planetary_facts_pd_DF()


def get_Braeunig_atmosphere_pd_DF():
  """@fn get_Braeunig_atmosphere_pd_DF
  @brief Retrieve from an already saved, pickled pandas DataFrame"""

  #current_directory = os.path.abspath(os.getcwd())

  current_directory = os.path.dirname(__file__)

  Data_subdirectory = "Data/"

  filename = "BraeunigAtmosphere_pd_df"

  try:
    return pd.read_pickle(
      os.path.join(current_directory, Data_subdirectory) + filename)
  except FileNotFoundError as error:
    print("File not found; make sure to generate the file: " + error)
    return    


BraeunigStandardAtmosphere  = get_Braeunig_atmosphere_pd_DF()



