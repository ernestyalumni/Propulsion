"""
@name Refresh.py
@file Refresh.py
@author Ernest Yeung
@email ernestyalumni@gmail.com
@brief Make sure to run this script for the first time to populate ../Data/
subdirectory with freshly downloaded and scraped data.
@ref
@details Run file as executable in the parent directory:
  python3 Scripts/Refresh.py
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

def main():

  Data_subdirectory = "./Data/"

  # Get NIST Fundamental Physical Constants as an ascii text file.  
  retrieve_file(subdirectory=Data_subdirectory)

  NISTConversionTable_to_pd_DF(subdirectory=Data_subdirectory)

  NASAPlanetaryFacts_to_pd_DF(subdirectory=Data_subdirectory)

  BraeunigAtmosphere_to_pd_DF(subdirectory=Data_subdirectory)

if __name__ == '__main__':

  import os, sys

  sys.path.append(os.path.abspath("./"))

  from Source.Scraping import retrieve_file

  from Source import (BraeunigAtmosphere_to_pd_DF,
    NASAPlanetaryFacts_to_pd_DF,
    NISTConversionTable_to_pd_DF)

  main()
