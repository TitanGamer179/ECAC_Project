#1.THE DATASET-> SEPARAR AS MÚSICAS TESTE DAS RESTANTES
import pandas as pd
import os
tests= pd.read_csv(r"C:\Users\titin\Downloads\dataset\build\MERGE_ECAC\merge_ecac\test_songs.csv")
meta= pd.read_csv(r"C:\Users\titin\Downloads\dataset\build\MERGE_ECAC\merge_ecac\metadata.csv")

is_test= meta["Audio_Song"].isin(tests["Audio_Song"]) #separar as músicas que são para testar

meta_test= meta[is_test]
meta_not_test=meta[~is_test]
