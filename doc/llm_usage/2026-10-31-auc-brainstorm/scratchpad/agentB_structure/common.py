import numpy as np, pandas as pd
ROOT='/Users/zwl0331/Desktop/Mini_project_1'
AUDIO=['time_signature','speechiness','danceability','duration_ms','energy','mode','instrumentalness','valence','key','tempo','loudness','acousticness','liveness']
def load():
    tr=pd.read_csv(f'{ROOT}/data/train.csv'); te=pd.read_csv(f'{ROOT}/data/test.csv'); sp=pd.read_csv(f'{ROOT}/artifacts/splits/production_v1.csv')
    tr['y']=(tr['Popularity_Type']=='High').astype(int)
    for d in (tr,te):
        s=d['track_album_release_date'].astype(str)
        d['date_str']=s; d['date_len']=s.str.len()
        d['year']=s.str[:4].astype(int)
        d['month']=pd.to_numeric(s.str[5:7],errors='coerce'); d['day']=pd.to_numeric(s.str[8:10],errors='coerce')
        dt=pd.to_datetime(s,errors='coerce',format='%Y-%m-%d'); d['dow']=dt.dt.dayofweek
    return tr,te,sp
def zfeat(tr,te=None):
    """standardized log-transformed audio features (fit on train); returns Ztr, Zte"""
    def f(d):
        X=pd.DataFrame(index=d.index)
        X['speechiness']=np.log(d.speechiness+1e-3); X['danceability']=d.danceability; X['duration']=np.log(d.duration_ms)
        X['energy']=d.energy; X['instrumentalness']=np.log(d.instrumentalness+1e-4); X['valence']=d.valence
        X['tempo']=np.log(d.tempo.clip(lower=30)); X['loudness']=d.loudness; X['acousticness']=np.log(d.acousticness+1e-4); X['liveness']=np.log(d.liveness+1e-3)
        return X
    A=f(tr); mu=A.mean(); sd=A.std()
    Z=(A-mu)/sd
    if te is None: return Z
    return Z,(f(te)-mu)/sd
