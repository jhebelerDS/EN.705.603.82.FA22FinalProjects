"""
Tarun Nadipalli - 705.603 Creating AI-Enabled Systems Final Project

{data.py}
A script used to collect Spotify song features data from the 
Spotify Million Playlist Dataset as found here:
https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

The Spotify Million Playlist Dataset is a .zip file that contains 
1000 json files that contain 1 million user-created Spotify playlists. To get 
a large selection of songs for our recommendation algorithm to choose from,
we iterate through these playlists and obtain the unique songs.

This script is also an extension/adaptation of what was written by nsanka
to process the Spotify Million Playlist Dataset. Original code here:
https://github.com/nsanka/RecSys/blob/main/code/read_spotify_million_playlists.py

Please note that for this script to work, two files must be exist within this project
folder. The first is the spotify_million_playlist_dataset.zip downloaded from the link
above. The second is the spotify_data.db file that contains the SQLite3 database
to store our processed data. Update the pathing in the global variables below if necessary.

This file can also be imported as a module and contains the following methods:

Methods
----------
create_connection(db_file: str):
    - Creates connection to SQLite3 database file that contains our processed data. 
create_table(conn, create_table_sql, table_name):
    - Creates a table in the specified db file.
create_all_tables(conn):
    - Creates the tracks and features schemas and tables in the db_file.
get_table_df(conn, table_name, limit):
    - Exports data in the tables as a dataframe for the recommendation algorithm.
get_max_track_id(conn, table_name):
    - Finds the max index to accurately populate the db table with song data.
create_audio_features(conn, cnt_uris, max_songs):
    - Obtains audio features for all of the songs within the tracks table.
extract_mpd_dataset(zip_file, db_file, num_files):
    - Extracts all of the Spotify Million Playlist Dataset files and reads json data from them.
process_json_data(json_data, db_file):
    - Combines json data from all of the .json files in the zip and keeps unique song id's.
"""

import fnmatch
import json
import logging
import sqlite3
import time
from sqlite3 import Error
from zipfile import ZipFile

import pandas as pd
from spotify import Spotify

zip_file = 'data/spotify_million_playlist_dataset.zip'
db_file = 'data/spotify_data.db'

logging.getLogger().setLevel(logging.ERROR)

def create_connection(db_file: str):
    """ Creates connection to SQLite3 database file that contains our processed data.

    Parameters
    ----------
    db_file: str
        - The string path of the db file for SQLite
        
    Raises
    ------
    Exception
        - If db connection cannot be established
        
    Returns
    ----------
    conn: obj
        - SQLite connection obj that gives us access to the db
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f'Connection to {db_file} is successful!')
    except Error as e:
        logging.error(f"Error connecting to {db_file} due to: {e}")
    return conn

def create_table(conn, create_table_sql: str, table_name: str):
    """ Creates a table in the specified db file
    
    Parameters
    ----------
    conn: obj
        - SQLite connection obj that gives us access to the db
    create_table_sql: str
        - string SQL query that creates the table and schema
    table_name: str
        - string name of table to be made in SQL db file
        
    Raises
    ------
    Exception
        - If table cannot be created from the query
    """
    try:
        cur = conn.cursor()
        cur.execute(create_table_sql)
        logging.info(f'Created table {table_name} successfully!')
    except Error as e:
        logging.error(f"Error creating table {table_name} due to: {e}")

def create_all_tables(conn):
    """ Creates the tracks and features schemas and tables in the db_file.

    Parameters
    ----------
    conn: obj
        - SQLite connection obj that gives us access to the db

    Raises
    ------
    Exception
        - If db connection cannot be established
        
    """
    sql_create_tracks_table = """ CREATE TABLE IF NOT EXISTS tracks (
                                    track_uri text NOT NULL,
                                    track_id integer NOT NULL
                                    ); """

    sql_create_features_table = """ CREATE TABLE IF NOT EXISTS features (
                                    track_id integer,
                                    track_uri text NOT NULL,
                                    danceability real,
                                    energy real,
                                    key real,
                                    loudness real,
                                    mode real,
                                    speechiness real,
                                    acousticness real,
                                    instrumentalness real,
                                    liveness real,
                                    valence real,
                                    tempo real,
                                    duration_ms integer,
                                    time_signature integer
                                    ); """

    # create tables
    if conn is not None:
        # create tracks table
        create_table(conn, sql_create_tracks_table, 'tracks')

        # create features table
        create_table(conn, sql_create_features_table, 'features')

    else:
        logging.error(f"Could not create tables in {db_file}!")

def get_table_df(conn, table_name: str, limit: str):
    """ Exports data in the tables as a dataframe for the recommendation algorithm.
    
    Parameters
    ----------
    conn: obj
        - SQLite connection obj that gives us access to the db
    table_name: str
        - string name of table to be made in SQL db file
    limit: str
        - limit the number of rows returned in the DataFrame
        
    Returns
    ----------
    table_df: obj
        - DataFrame containing all of the table data
    """
    
    logging.info(f'Reading table {table_name} from database.')
    if limit is None:
        query = 'select * from ' + table_name + ';'
    else:
        query = 'select * from ' + table_name + ' limit ' + limit + ';'
    table_df = pd.read_sql(query, conn)
    return table_df

def get_max_track_id(conn, table_name: str):
    """ Finds the max index to accurately populate the db table with song data.
    
    Parameters
    ----------
    conn: obj
        - SQLite connection obj that gives us access to the db
    table_name: str
        - string name of table to be made in SQL db file
    
    Returns
    ----------
    max_track_id: int
        - Integer containing max track index so more data can be inserted at that index and after
    """
    cur = conn.cursor()
    cur.execute("select max(track_id) from " + table_name)
    rows = cur.fetchall()
    max_track_id = rows[0][0]
    if max_track_id is None:
        max_track_id = 0
    return max_track_id

def extract_mpd_dataset(zip_file, db_file, num_files=0):
    """ Extracts all of the Spotify Million Playlist Dataset files and processes json data from them.
    
    Parameters
    ----------
    zip_file: obj
        - Path to the million playlist dataset zip file
    db_file: str
        - The string path of the db file for SQLite
    num_files: int
        - number of json files from Spotify MPD.zip to iterate through
    """
    with ZipFile(zip_file) as zipfiles:
        file_list = zipfiles.namelist()
        
        json_files = fnmatch.filter(file_list, "*.json")
        json_files = [f for i,f in sorted([(int(filename.split('.')[2].split('-')[0]), filename) for filename in json_files])]
        logging.info('Obtained all .json files from MPD.zip!')
        cnt = 0
        for filename in json_files:
            cnt += 1
            with zipfiles.open(filename) as json_file:
                json_data = json.loads(json_file.read())
                process_json_data(json_data, db_file)

            if (cnt == num_files) and (num_files > 0):
                break
        
def process_json_data(json_data, db_file):
    """ Combines json data from all of the .json files in the zip and keeps unique song id's
    in the tracks table.
    
    Parameters
    ----------
    db_file: str
        - The string path of the db file for SQLite
    json_data: list
        - JSON data obtained from each slice of the MPD.zip file
    """  
    conn = create_connection(db_file)
    # track_uri is the uri of the song
    # track_id is the id of the row within the database.
    max_track_id = get_max_track_id(conn, 'tracks')

    # get all the tracks in the file
    tracks_df = pd.json_normalize(json_data['playlists'], record_path=['tracks'])
    tracks_df['track_uri'] = tracks_df['track_uri'].apply(lambda uri: uri.split(':')[2])

    all_tracks_df = pd.read_sql('select track_id, track_uri from tracks', conn)
    tracks_df = tracks_df.merge(all_tracks_df, how='left', on='track_uri').fillna(0)
    num_existing_tracks = len(tracks_df[tracks_df['track_id'] != 0]['track_uri'].unique())
    logging.info(f'Number of tracks that already exist: {num_existing_tracks}')

    # temp column to identify existing/duplicate songs to be removed
    tracks_df['track_id1'] = tracks_df[tracks_df["track_id"] == 0][['track_uri']].groupby('track_uri').ngroup()+max_track_id+1
    tracks_df['track_id'] = tracks_df['track_id'] + tracks_df['track_id1'].fillna(0)
    tracks_df['track_id'] = tracks_df['track_id'].astype('int64')

    # save unique tracks to the database
    tracks_df = tracks_df[tracks_df['track_id1'].notna()]
    tracks_df.drop(['pos', 'duration_ms', 'track_id1'], axis=1, inplace=True)
    tracks_df = tracks_df[['track_id', 'track_uri']]
    tracks_df = tracks_df.drop_duplicates(subset='track_uri', keep="first")
    logging.info(f'Total unique tracks: {len(tracks_df)}')
    logging.info(f"Adding tracks to database: {max_track_id+1} to {tracks_df['track_id'].max()}")
    tracks_df.to_sql(name='tracks', con=conn, if_exists='append', index=False)

    if conn:
        conn.close()

def create_audio_features(conn, cnt_uris=100, max_songs=1_000):
    """ Obtains audio features for all of the songs within the tracks table.
    
    Parameters
    ----------
    conn: obj
        - SQLite connection obj that gives us access to the db
    cnt_uris: int
        - integer amount of song id's to iteratively obtain 
        information on (100 songs is the limit for Spotify API calls)
    """
    sp = Spotify(client_id='baf04d54648346de81af8a9904349531', 
                 client_secret='074087a86045465dbd582802befa6f94',
                 scope="playlist-modify-public")

    max_track_id = get_max_track_id(conn, 'tracks')
    min_track_id = get_max_track_id(conn, 'features')

    logging.info(f'Minimum Track ID in Features: {min_track_id} | Max Track ID in Tracks: {max_track_id}')
    for idx in range(min_track_id, max_songs, cnt_uris):
        logging.info(f"Getting audio features for Track ID: {idx+1} to {idx+cnt_uris}")
        cur = conn.cursor()
        cur.execute('''select track_id, track_uri from tracks where (track_id > ?) and (track_id <= ?)''', (idx, idx+cnt_uris))
        rows = cur.fetchall()
        
        uris = [row[1] for row in rows]
        for _ in range(10):
            try:
                feats_list = sp.get_audio_features(uris)
                time.sleep(1)
            except Exception as e: 
                logging.info(f'Error retrieveing audio features due to {e}')
            else:
                break
        else:
            logging.info(f'All 10 attempts to get audio features failed, try again later!')
            break
        
        # Remove all objects that are null before adding to database
        null_features = [i for i in range(len(feats_list)) if feats_list[i] == None]
        for ind in null_features: uris.pop(ind)
        
        track_id_list = range(idx+1, idx+cnt_uris+1)
        track_id_list = [track_id_list[feats_list.index(item)] for item in feats_list if item]
        
        feats_list = [item for item in feats_list if item]
        
        feats_df = pd.DataFrame(feats_list)
        columns = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
        feats_df = feats_df[columns]
        feats_df.insert(loc=0, column='track_id', value=track_id_list)
        feats_df.insert(loc=0, column='track_uri', value=uris)
        feats_df.to_sql(name='features', con=conn, if_exists='append', index=False)
    if conn:
        conn.close()

        
if __name__ == '__main__':
    # Creating db connection to 'spotify_data.db'
    conn = create_connection(db_file)
    
    # Creating tracks, features tables in database
    # Tracks contains the song id's, features has audio features of all songs
    create_all_tables(conn)
    
    # Add tracks each json file in zipfile
    extract_mpd_dataset(zip_file, db_file, 0)
    
    # get audio features for all tracks
    create_audio_features(conn, 100, '300000')
    
    # Get DataFrame of all song features
    # get_table_df(conn, 'features', '10000')