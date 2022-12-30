"""
Tarun Nadipalli - 705.603 Creating AI-Enabled Systems Final Project

{recommend.py}
This script contains the full functionality end-to-end to retrieve a Spotify playlist
input from the user and send back a playlist of recommendations using TensorFlow and Keras.
This is the final script that is run by the docker image in order to generate recommendations 
and makes use of the other modules in this project: spotify.py, data.py, models.py.

Methods
----------
main():
    - Retrieves user input, gets the recommended songs, and returns the playlist to the user.
normalize_numerical_features(model, data)
    - Helper function to normalize all numerical embeddings in a tf.keras.Model.
retrieval(user_df, songs_df):
    - Trains FactorizedTopK model on user data and retrieves 1000 approximate 
    song recommendations from full dataset.
ranking(user_df, retrieved_recs_df):
    - Ranks 1000 approximate recommendations using Ranking model and returns best 50 songs.
"""
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# suppress tensorflow unnecessary logging ^

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from data import create_connection, db_file, get_table_df
from models import *
from spotify import Spotify


def main():
    """ Retrieves user input, gets the recommended songs, and returns the playlist to the user.
    (Also prints a banner to terminal for a more clean "front-end" experience for the user)
    
    Utilizes functionality from all modules in this project: spotify.py, data.py, models.py.
    """    
    # printing terminal banner to act as pseudo front-end
    f = open('banner.txt', 'r')
    content = f.read()
    print(content)
    f.close()
    
    # instantiating spotify connection object
    sp = Spotify(client_id='baf04d54648346de81af8a9904349531', 
                client_secret='074087a86045465dbd582802befa6f94',
                scope="playlist-modify-public")
    
    # creating SQLite db connection to access song data
    conn = create_connection(db_file)
    
    user_df = sp.get_spotify_data_from_user()
    songs_df = get_table_df(conn, 'features', '300000').drop(['track_id'], axis=1)
    
    print('\033[93m' + "Retrieved all data! Generating recommendations...")
    retrieved_recs_df = retrieval(user_df, songs_df)
    print('\033[93m' + "Almost there...")
    ranked_recs = ranking(user_df, retrieved_recs_df)
    
    sp.create_playlist(ranked_recs)

def normalize_numerical_features(model, data):
    """ Helper function to normalize all numerical embeddings in our SongModel: tf.keras.Model
    from models.py.
    
    Parameters
    ----------
    model: tf.keras.model
        - The SongModel with numerical features needing to be normalized
    data: tf.data.Dataset
        - the Tensor Dataset with the numerical data needing to be normalized
    """    
    model.danceability_normalized.adapt(
        data.map(lambda x: x["danceability"]).batch(100))

    model.energy_normalized.adapt(
        data.map(lambda x: x["energy"]).batch(100))

    model.loudness_normalized.adapt(
        data.map(lambda x: x["loudness"]).batch(100))

    model.speechiness_normalized.adapt(
        data.map(lambda x: x["speechiness"]).batch(100))

    model.acousticness_normalized.adapt(
        data.map(lambda x: x["acousticness"]).batch(100))

    model.instrumentalness_normalized.adapt(
        data.map(lambda x: x["instrumentalness"]).batch(100))

    model.liveness_normalized.adapt(
        data.map(lambda x: x["liveness"]).batch(100))

    model.valence_normalized.adapt(
        data.map(lambda x: x["valence"]).batch(100))

    model.tempo_normalized.adapt(
        data.map(lambda x: x["tempo"]).batch(100))

    model.duration_normalized.adapt(
        data.map(lambda x: x["duration_ms"]).batch(100))
    
def retrieval(user_df, songs_df):
    """ First step in recommendation process. 
    The retrieval function using the tfrs.metrics.FactorizedTopK trained on 
    the song data from the user parses through the full 300k songs to return 
    1000 approximated recommendations. 
    
    Parameters
    ----------
    user_df: pd.DataFrame
        - Pandas DataFrame containing all of the song data from user playlist
    songs_df: pd.DataFrame
        - Pandas DataFrame containing all song data from 300k songs dataset
        
    Returns
    ----------
    retrieved_recs: pd.DataFrame
        - dataframe of 1000 approximate recommendations for ranking
    """      
    # convert dataframes to tensorflow datasets
    user = tf.data.Dataset.from_tensor_slices(dict(user_df))
    songs = tf.data.Dataset.from_tensor_slices(dict(songs_df))
    
    # iterate through tf datasets to only keep necessary features
    user = user.map(lambda x: {
        "danceability": x["danceability"],
        "energy": x["energy"],
        "key": x["key"],
        "loudness": x["loudness"],
        "mode": x["mode"],
        "speechiness": x["speechiness"],
        "acousticness": x["acousticness"],
        "instrumentalness": x["instrumentalness"],
        "liveness": x["liveness"],
        "valence": x["valence"],
        "tempo": x["tempo"],
        "duration_ms": x["duration_ms"],
        "time_signature": x["time_signature"]
    })

    songs = songs.map(lambda x: {
        "danceability": x["danceability"],
        "energy": x["energy"],
        "key": x["key"],
        "loudness": x["loudness"],
        "mode": x["mode"],
        "speechiness": x["speechiness"],
        "acousticness": x["acousticness"],
        "instrumentalness": x["instrumentalness"],
        "liveness": x["liveness"],
        "valence": x["valence"],
        "tempo": x["tempo"],
        "duration_ms": x["duration_ms"],
        "time_signature": x["time_signature"]
    })
    
    # instantiate SongModels for the user songs and all songs
    user_model = SongModel()
    normalize_numerical_features(user_model, user)
    
    song_model = SongModel()
    normalize_numerical_features(song_model, songs)

    # shuffle training data
    tf.random.set_seed(42)
    shuffled = user.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    
    # retrieval task algorithm to be used in RetrievalModel
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        songs.batch(128).map(song_model)))
    
    # instantiate Retrieval Model using user data model, song data model, and retrieval task
    model = RetrievalModel(user_model, song_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))
    
    fit_data = model.fit(shuffled.batch(4096), epochs=5, verbose=0)

    # retrieve 1000 approximate song recommendations
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(songs.batch(100).map(model.song_model))

    _, retrieved_songs = index({
        "danceability": np.array(user_df['danceability']),
        "energy": np.array(user_df['energy']),
        "key": np.array(user_df['key']),
        "loudness": np.array(user_df['loudness']),
        "mode": np.array(user_df['mode']),
        "speechiness": np.array(user_df['speechiness']),
        "acousticness": np.array(user_df['acousticness']),
        "instrumentalness": np.array(user_df['instrumentalness']),
        "liveness": np.array(user_df['liveness']),
        "valence": np.array(user_df['valence']),
        "tempo": np.array(user_df['tempo']),
        "duration_ms": np.array(user_df['duration_ms']),
        "time_signature": np.array(user_df['time_signature']),
        },k=1000
    )
    
    # make approximate song rec dataframe based on indexes retrieved from above ^
    retrieved_recs = songs_df.iloc[retrieved_songs[0].numpy(), :]
    return retrieved_recs
        
def ranking(user_df, retrieved_recs_df):
    """ Second step in recommendation process. 
    The ranking function calculates the top 50 songs out of the 1000 approximate
    recommendations from the Retrieval stage. We instantiate a rating of 1 on 
    all of the user songs and let the Ranking function predict a rating for all of
    our approximate recommendations. The songs with the highest predicted rating
    are returned to be created into a playlist and sent back to the user.
    
    Parameters
    ----------
    user_df: pd.DataFrame
        - Pandas DataFrame containing all of the song data from user playlist
    retrieved_recs_df: pd.DataFrame
        - Pandas DataFrame containing all song data from 1000 approximate recommendations
    
    ranked_recs: list
        - list of 50 best recommendations with song id's to create final playlist with
    """   
    # instantiate rating of 1 for each user song (shows that they like it)
    user_df['ratings'] = 1
    
    # instantiate tf.Dataset from input dataframes
    user = tf.data.Dataset.from_tensor_slices(dict(user_df))
    retrieved_recs = tf.data.Dataset.from_tensor_slices(dict(retrieved_recs_df))

    user = user.map(lambda x: {
        "danceability": x["danceability"],
        "energy": x["energy"],
        "key": x["key"],
        "loudness": x["loudness"],
        "mode": x["mode"],
        "speechiness": x["speechiness"],
        "acousticness": x["acousticness"],
        "instrumentalness": x["instrumentalness"],
        "liveness": x["liveness"],
        "valence": x["valence"],
        "tempo": x["tempo"],
        "duration_ms": x["duration_ms"],
        "time_signature": x["time_signature"],
        "ratings": x["ratings"]
    })

    retrieved_recs = retrieved_recs.map(lambda x: {
        "danceability": x["danceability"],
        "energy": x["energy"],
        "key": x["key"],
        "loudness": x["loudness"],
        "mode": x["mode"],
        "speechiness": x["speechiness"],
        "acousticness": x["acousticness"],
        "instrumentalness": x["instrumentalness"],
        "liveness": x["liveness"],
        "valence": x["valence"],
        "tempo": x["tempo"],
        "duration_ms": x["duration_ms"],
        "time_signature": x["time_signature"],
        # "ratings": x["ratings"]
    })
    
    user_model = SongModel()
    normalize_numerical_features(user_model, user)

    retrieved_recs_model = SongModel()
    normalize_numerical_features(retrieved_recs_model, retrieved_recs)
  
    # instantiate ranking model with both user and recommendations datasets
    model = FullRankingModel(RankingModel(user_model, retrieved_recs_model))
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    # instantiate training dataset with batching and shuffling
    tf.random.set_seed(42)
    shuffled = user.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    cached_train = train.shuffle(100_000).batch(8192).cache()
    
    fit_data = model.fit(cached_train, epochs=3, verbose=0)

    # predict ratings for all 1000 potential recommendations
    predicted_ratings = {}
    predicted_ratings = model({
        "danceability": np.array(retrieved_recs_df['danceability']),
        "energy": np.array(retrieved_recs_df['energy']),
        "key": np.array(retrieved_recs_df['key']),
        "loudness": np.array(retrieved_recs_df['loudness']),
        "mode": np.array(retrieved_recs_df['mode']),
        "speechiness": np.array(retrieved_recs_df['speechiness']),
        "acousticness": np.array(retrieved_recs_df['acousticness']),
        "instrumentalness": np.array(retrieved_recs_df['instrumentalness']),
        "liveness": np.array(retrieved_recs_df['liveness']),
        "valence": np.array(retrieved_recs_df['valence']),
        "tempo": np.array(retrieved_recs_df['tempo']),
        "duration_ms": np.array(retrieved_recs_df['duration_ms']),
        "time_signature": np.array(retrieved_recs_df['time_signature'])
    })

    # find top 50 rated songs to return to user
    retrieved_recs_df['ratings'] = predicted_ratings[:].numpy()
    retrieved_recs_df = retrieved_recs_df.sort_values(by=['ratings'], ascending=False)[:50]
    ranked_recs = retrieved_recs_df['track_uri'].values
    return ranked_recs

if __name__ == "__main__":
    main()