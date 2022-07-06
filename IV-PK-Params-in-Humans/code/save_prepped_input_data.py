"""
========================================================================================================
Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters in Humans for 1352 Drug Compounds
========================================================================================================
"""

from util_local import load_data
filename_data_in   = '../data/IV-PK-Params-in-Humans.csv'
filename_data_out  = '../data/IV-PK-Params-in-Humans.PREPPED.csv'
_ , data , feature = load_data(filename_data_in)
data.to_csv( filename_data_out , columns = feature['B'] , index = False )
