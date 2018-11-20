import pandas as pd
from numpy import *
import numpy as np
from scipy import sparse




def create_Matrix():

    data_playlists= pd.read_csv('all/train.csv',low_memory = False)

    #SPLITTAGGIO DEI DATI

    #prendo il 20% delle random playlist da target playlist
    target_playlist= pd.read_csv('all/target_playlists.csv',low_memory = False)
    random_playlist=target_playlist.iloc[5000:10000]
    random_id = random_playlist.playlist_id.values
    random_data=data_playlists.loc[data_playlists['playlist_id'].isin(random_id)]
    #random_data.shape = (114513, 2)
    n=round((114513/100)*20)
    random_subset = random_data.sample(n=n)
    random_subset.head(10)

    #prendo il 20% delle sequential playlist da target playlist
    sequential_playlist=target_playlist.iloc[:5000]
    sequential_id = sequential_playlist.playlist_id.values
    sequential_data=data_playlists.loc[data_playlists['playlist_id'].isin(sequential_id)]
    n1=round((115553/100)*20)
    for pid in sequential_id:
        playlistID=sequential_data.loc[sequential_data['playlist_id']==pid]
        start=sequential_data.index[sequential_data['playlist_id'] == pid][0]
        end=start+round((playlistID.shape[0]*50)/100)

        dropping=list(range(start, end))    
        sequential_data=sequential_data.drop(dropping)


    sequential_subset = sequential_data.sample(n=n1)

    #SETTO URM TEST
    URM_test=zeros((50446,20635))
    #20% from random
    for row in random_subset.itertuples():
        riga = row.playlist_id
        #print(riga)
        colonna= row.track_id
        #print(colonna)
        URM_test[riga,colonna]=1
    #20% from random
    for row in sequential_subset.itertuples():
        riga = row.playlist_id
        #print(riga)
        colonna= row.track_id
        #print(colonna)
        URM_test[riga,colonna]=1

    #SETTO URM TRAIN e ALL
    URM_train=zeros((50446,20635))
    URM_all=zeros((50446,20635))
    #inizializzazione completa
    for row in data_playlists.itertuples():
        riga = row.playlist_id
        #print(riga)
        colonna= row.track_id
        #print(colonna)
        URM_all[riga,colonna]=1
        URM_train[riga,colonna]=1
    #rimozione del 20%from random
    for row in random_subset.itertuples():
        riga = row.playlist_id
        #print(riga)
        colonna= row.track_id
        #print(colonna)
        URM_train[riga,colonna]=0
    #rimozione del 20%from sequential
    for row in sequential_subset.itertuples():
        riga = row.playlist_id
        #print(riga)
        colonna= row.track_id
        #print(colonna)
        URM_train[riga,colonna]=0

    URM_all=sparse.csr_matrix(URM_all)   
    URM_train=sparse.csr_matrix(URM_train)
    URM_test=sparse.csr_matrix(URM_test)
    
    sparse.save_npz('/Matrix/URM_all_matrix.npz', URM_all)
    sparse.save_npz('/Matrix/URM_train_matrix.npz', URM_train)
    sparse.save_npz('/Matrix/URM_test_matrix.npz', URM_test)


create_Matrix()