import logging
import re

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

class Spotify():
    """
    Tarun Nadipalli - 705.603 Creating AI-Enabled Systems Final Project
    
    {spotify.py}
    A class used to query playlist data from a user and from the Spotify API,
    extending functionality of the 'spotipy' module for this use case.

    Attributes
    ----------
    client : obj
        - the Spotify client object that utilizes the credentials needed to use the Spotify API

    Methods
    ----------
    get_input(self):
        - Asks the user for the Spotify Playlist link they want to base their recommendations on
    read_playlist_tracks(self, playlist_id: str):
        - Retrieves list of song id's from input playlist
    get_audio_features(self, song_ids: list):
        - Retrieves all audio features for song id's from input playlist
    get_spotify_data_from_user(self):
        - Runs all methods to get input playlist and return DataFrame of user song data
    create_playlist(self, song_ids):
        - Creates playlist from list of song id's returned from recommendation algorithm
    """
    def __init__(self, client_id: str, client_secret: str, scope:str):
        """
        Parameters
        ----------
        client_id: str
            - The client id for the Spotify Developer Account App (needed for OAuth authentication)
        client_secret: str
            - The client id for the Spotify Developer Account App (needed for OAuth authentication)
        scope: str
            - The authorization tags needed to create playlists from a Spotify user account
        """
        self.client = spotipy.Spotify(oauth_manager=SpotifyOAuth(client_id=client_id,
                                                           client_secret=client_secret,
                                                           scope=scope,
                                                           redirect_uri="https://127.0.0.1:8080"))
        
        # Using print to communicate to the user, logging for debugging
        logging.getLogger().setLevel(logging.ERROR)
    
    def get_input(self):
        """ Asks the user for the Spotify Playlist link they want to base their recommendations on.
            
        Returns
        ----------
        playlist_id: str
            - playlist string id for reading all songs from that playlist
        """
        # implement Regex matching to validate link has format: "https://open.spotify.com/playlist/***"
        while True:
            playlist = input('\033[95m' + "Please enter your Spotify playlist link here (ensure it is a PUBLIC playlist): ").strip()
            if not re.match("^(spotify:|https://open+\.spotify\.com/playlist/)", playlist):
                print("That's not quite right. Check the link and enter again.")
            else:
                print('\033[93m' + "Got it! The playlist you've submitted is: " + playlist)
                break

        # initializing substrings that appear on either end of "id"
        sub1 = "playlist/"
        sub2 = "?"

        # getting index of substrings
        idx1 = playlist.index(sub1)
        idx2 = playlist.index(sub2)

        # getting "id" in between substrings
        playlist_id = playlist[idx1 + len(sub1):idx2]
        return playlist_id
    
    def read_playlist_tracks(self, playlist_id: str) -> list:
        """ Retrieves list of song id's from input playlist.

        Parameters
        ----------
        playlist_id: str
            - The string id for the user input playlist
            
        Raises
        ------
        Exception
            - If Spotify API read playlist tracks call malfunctions or runs into API limit error
            
        Returns
        ----------
        songs: list
            - list of song id's from input playlist
        """
        # playlist_tracks API call limited to 100 id's at a time, offset allows to iterate through
        offset = 0
        songs = []
        while True:
            try:
                # returns track id for each song in playlist and total tracks in playlist
                response = self.client.playlist_tracks(playlist_id,
                                            offset=offset,
                                            fields='items.track.id,total',
                                            additional_types=['track'])
            except Exception as e:
                logging.error(f"Error retrieving tracks from playlist due to: {e}")
                return None
                
            if len(response['items']) == 0:
                break
            
            songs = songs + [song['track']['id'] for song in response['items']]
            offset = offset + len(response['items'])
            
        logging.info("Retrieved Playlist track ID's!")        
        return songs
    
    def get_audio_features(self, song_ids: list) -> list:
        """ Retrieves all audio features for song id's from input playlist.

        Parameters
        ----------
        song_ids: list
            - The list of all song id's needed to query for song audio features
            
        Raises
        ------
        Exception
            - If Spotify API read audio features call malfunctions or runs into API limit error
            
        Returns
        ----------
        song_features: list
            - list of json objects of song features data
        """
        # limit of 100 on audio_features API call, must iterate through calls manually
        # unless playlist has <= 100 songs
        if len(song_ids) <= 100:
            try:
                response = self.client.audio_features(song_ids)
                logging.info("Retrieved track audio features!")
                return response
            except Exception as e:
                logging.error(f"Error retrieving <=100 audio features due to: {e}")
                return None
        else:
            offset = 0
            song_features = []
            
            # runs until all songs are queried for their audio features, 100 at a time
            while offset < len(song_ids):
                try:
                    response = self.client.audio_features(song_ids[offset:offset+100])
                except Exception as e:
                    logging.error(f"Error retrieving audio features for tracks '{offset}' to '{offset+100}' due to: {e}")
                    return None
                        
                if len(response) == 0:
                    break
                
                song_features = song_features + response
                offset = offset + len(response)
            
            logging.info("Retrieved track audio features!")            
            return song_features
    
    def get_spotify_data_from_user(self):
        """ Runs all methods to get input playlist and return DataFrame of user song data
            
        Returns
        ----------
        song_data: DataFrame
            - DataFrame containing all song data/features from user input playlist
        """
        playlist_id = self.get_input()
        # playlist_id = '37i9dQZF1DX0XUsuxWHRQd'
        songs = self.read_playlist_tracks(playlist_id)
        song_data = self.get_audio_features(songs)
        song_data = pd.DataFrame(song_data).drop(['type', 'uri','track_href', 'analysis_url'], axis=1)
        song_data.columns = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness',
              'liveness','valence','tempo','track_uri','duration_ms','time_signature']
        return song_data
    
    def create_playlist(self, song_ids: list):
        """ Creates playlist from list of song id's returned from recommendation algorithm
        
        Parameters
        ----------
        song_ids: list
            - The list of song id's returned from recommendation algorithm to be added to new playlist
            
        Returns
        ----------
        playlist_link: str
            - The final playlist link to be returned to the user with their recommendations
        """
        # Note: Not possible to create playlists without a user account
        # Thus, user_id is hard-coded here from Tarun Nadipalli's personal Spotify account
        user_id = "1234520113"
        name = "Tarun's Recommendations!"
        desc = "Playlist of recommendations made by Tarun's JHU 705.603 Creating AI-Enabled Systems class project!"
        try:
            resp = self.client.user_playlist_create(user=user_id, name=name,
                                                description=desc, public=True)
        except Exception as e:
            logging.error(f"Error creating playlist due to: {e}. Try again.")
            return None
        
        playlist = resp['id']
        try:
            self.client.playlist_add_items(playlist, song_ids)
        except Exception as e:
            logging.error(f"Error adding songs to the playlist due to: {e}. Try again.")
            return None

        playlist_link = "https://open.spotify.com/playlist/" + playlist
        print('\033[92m' + "Done! Here are your recommendations: " + playlist_link)

        return playlist_link


if __name__ == "__main__":
    # calling main returns a link to a copy of the input playlist
    # for local testing only
    sp = Spotify(client_id='baf04d54648346de81af8a9904349531', 
                 client_secret='074087a86045465dbd582802befa6f94',
                 scope="playlist-modify-public")
    link = sp.get_input()
    songs = sp.read_playlist_tracks(link)
    song_data = sp.get_audio_features(songs)
    playlist_link = sp.create_playlist(songs)
    print("Your Recommendation Playlist Link!: ", playlist_link)
    # https://open.spotify.com/playlist/37i9dQZF1DX0XUsuxWHRQd?si=850449279b794ac4
    # RapCaviar playlist ^ used for testing
    