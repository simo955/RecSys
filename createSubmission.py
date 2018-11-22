import numpy as np
import pandas as pd
import scipy.sparse as sps

def create_submission(recommender):

    data_playlists= pd.read_csv('all/train.csv',low_memory = False)
    CFRec= pd.read_csv('all/sample_submission.csv',low_memory = False)
    RecommenderCSV=recommender

    def recommendationSS (playlistID):
        playListTrack = data_playlists.loc[data_playlists['playlist_id'] == playlistID]
        trackList= playListTrack["track_id"]
    
        reccomandationList = RecommenderCSV.recommend(playlistID)
    
        recommendation = list()
        n=0
        while len(recommendation)<10:
            if ~ trackList.isin([reccomandationList[n]]).any():
                recommendation.append (reccomandationList[n])
            n=n+1
        
        return recommendation
    
    for row in CFRec.itertuples():
        CFRec.at[row.Index, "track_ids"]= recommendationSS(row.playlist_id)
    
    CFRec.to_csv("Submissions/newSubmission1.csv", encoding='utf-8', index=False)

