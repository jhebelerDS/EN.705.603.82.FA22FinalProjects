""" 
Tarun Nadipalli - 705.603 Creating AI-Enabled Systems Final Project
    
{models.py}
File containing four different TensorFlow Model classes for the retrieval and ranking
steps for creating Spotify song recommendations.

Classes
----------
SongModel():
    - The SongModel contains the generalized structure to create a model with the input or
    full song dataset data
RetrievalModel():
    - The RetrievalModel takes in two SongModels (one query, one candidate model) to create
    approximated recommendations with the FactorizedTopK algorithm
RankingModel():
    - The RankingModel takes in two SongModels (one query, one candidate model) to create 
    embeddings necessary for predicting song ratings
FullRankingModel():
    - The FullRankingModel takes the RankingModel and generates predicted ratings for all
    of the approximate recommendations in order to find the best songs for the user
"""

from typing import Dict, Text

import tensorflow as tf
import tensorflow_recommenders as tfrs


class SongModel(tf.keras.Model):
    """ The SongModel contains the generalized structure to create a model with the input or
    full song dataset data.

    Attributes
    ----------
    key_embedding: tf.keras.layers
        - embedding for the key feature of a song
    time_embedding: tf.keras.layers
        - embedding for the time feature of a song
    danceability_normalized: tf.keras.layers
        - embedding for the danceability feature of a song
    energy_normalized: tf.keras.layers
        - embedding for the energy feature of a song
    loudness_normalized: tf.keras.layers
        - embedding for the loudness feature of a song
    speechiness_normalized: tf.keras.layers
        - embedding for the speechiness feature of a song
    acousticness_normalized: tf.keras.layers
        - embedding for the acousticness feature of a song
    instrumentalness_normalized: tf.keras.layers
        - embedding for the instrumentalness feature of a song
    liveness_normalized: tf.keras.layers
        - embedding for the liveness feature of a song
    valence_normalized: tf.keras.layers
        - embedding for the valence feature of a song
    tempo_normalized: tf.keras.layers
        - embedding for the tempo feature of a song
    duration_normalized: tf.keras.layers
        - embedding for the duration feature of a song
    mode_embedding: tf.keras.layers
        - embedding for the mode feature of a song
    
    Methods
    ----------
    call(features):
        - Takes in the input dictionary of features and passes it through each embedding layer
        and concatenates all the embeddings for use by the model
    """
    def __init__(self):
        super().__init__()
        
        # hard coding all possible values of categorical features
        key_vocab = [-1,0,1,2,3,4,5,6,7,8,9,10,11]
        time_vocab = [3,4,5,6,7]
        mode_vocab = [0,1]

        # categorical feature embeddings
        self.key_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(
                vocabulary=key_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(key_vocab) + 1, 32)
        ])
        
        self.time_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(
                vocabulary=time_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(time_vocab) + 1, 32)
        ])
        
        # numerical feature embeddings
        self.danceability_normalized = tf.keras.layers.Normalization(axis=None)
        self.energy_normalized = tf.keras.layers.Normalization(axis=None)
        self.loudness_normalized = tf.keras.layers.Normalization(axis=None)
        self.speechiness_normalized = tf.keras.layers.Normalization(axis=None)
        self.acousticness_normalized = tf.keras.layers.Normalization(axis=None)
        self.instrumentalness_normalized = tf.keras.layers.Normalization(axis=None)
        self.liveness_normalized = tf.keras.layers.Normalization(axis=None)
        self.valence_normalized = tf.keras.layers.Normalization(axis=None)
        self.tempo_normalized = tf.keras.layers.Normalization(axis=None)
        self.duration_normalized = tf.keras.layers.Normalization(axis=None)
        
        # binary feature embeddings
        self.mode_embedding = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(
                vocabulary=mode_vocab, mask_token=None),
            tf.keras.layers.Embedding(len(mode_vocab) + 1, 32)
        ])

    def call(self, features):
        # takes input dictionary, passes through each input layer and concats the result.
        return tf.concat([
            # categorical variables
            self.key_embedding(features["key"]),
            self.time_embedding(features["time_signature"]),
            
            # numerical variables
            tf.reshape(self.danceability_normalized(features["danceability"]), (-1, 1)),
            tf.reshape(self.energy_normalized(features["energy"]), (-1, 1)),
            tf.reshape(self.loudness_normalized(features["loudness"]), (-1, 1)),
            tf.reshape(self.speechiness_normalized(features["speechiness"]), (-1, 1)),
            tf.reshape(self.acousticness_normalized(features["acousticness"]), (-1, 1)),
            tf.reshape(self.instrumentalness_normalized(features["instrumentalness"]), (-1, 1)),
            tf.reshape(self.liveness_normalized(features["liveness"]), (-1, 1)),
            tf.reshape(self.valence_normalized(features["valence"]), (-1, 1)),
            tf.reshape(self.tempo_normalized(features["tempo"]), (-1, 1)),
            tf.reshape(self.duration_normalized(features["duration_ms"]), (-1, 1)),
            
            # binary variables
            self.mode_embedding(features["mode"])
        ], axis=-1)

class RetrievalModel(tfrs.Model):
    """ The RetrievalModel takes in two SongModels (one query, one candidate model) to create
    approximated recommendations with the FactorizedTopK algorithm.

    Attributes
    ----------
    user_model: tf.keras.Model
        - The SongModel containing embeddings associated with the input playlist data from the user
    song_model: tf.keras.Model
        - The SongModel containing embeddings associated with the song data from 300k songs
    task: tfrs.tasks.Retrieval
        - The RetrievalTask Model (FactorizedTopK) that creates 1000 approximate recommendations for us
        
    Methods
    ----------
    compute_loss(features, training):
        - Passes in the features for creating embeddings for each SongModel in order to calculate 
        the loss for Retrieval task
    """
    def __init__(
        self,
        user_model: tf.keras.Model,
        song_model: tf.keras.Model,
        task: tfrs.tasks.Retrieval):
        super().__init__()
    
        self.user_model = user_model
        self.song_model = song_model
    
        # sets up retrieval task for what is passed in in 'recommend.py'
        self.task = task
    
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model({
            "key": features["key"],
            "time_signature": features["time_signature"],
            "danceability": features["danceability"],
            "energy": features["energy"],
            "loudness": features["loudness"],
            "speechiness": features["speechiness"],
            "acousticness": features["acousticness"],
            "instrumentalness": features["instrumentalness"],
            "liveness": features["liveness"],
            "valence": features["valence"],
            "tempo": features["tempo"],
            "duration_ms": features["duration_ms"],
            "mode": features["mode"],
        })
        
        song_embeddings = self.song_model({
            "key": features["key"],
            "time_signature": features["time_signature"],
            "danceability": features["danceability"],
            "energy": features["energy"],
            "loudness": features["loudness"],
            "speechiness": features["speechiness"],
            "acousticness": features["acousticness"],
            "instrumentalness": features["instrumentalness"],
            "liveness": features["liveness"],
            "valence": features["valence"],
            "tempo": features["tempo"],
            "duration_ms": features["duration_ms"],
            "mode": features["mode"],
        })
        
        return self.task(user_embeddings, song_embeddings)
    
class RankingModel(tf.keras.Model):
    """ The RankingModel takes in two SongModels (one query, one candidate model) to create 
    embeddings necessary for predicting song ratings.

    Attributes
    ----------
    user_model: tf.keras.Model
        - The SongModel containing embeddings associated with the input playlist data from the user
    song_model: tf.keras.Model
        - The SongModel containing embeddings associated with the song data from 1000 retrieved songs
        from the retrieval stage
        
    Methods
    ----------
    call(features):
        - Takes in the input dictionary of features and passes it through each embedding layer
        and concatenates all the embeddings for use by the model
    """
    def __init__(
        self,
        user_model: tf.keras.Model,
        song_model: tf.keras.Model):
        super().__init__()
        
        self.user_model = user_model
        self.song_model = song_model
        
        # computes rating predictions
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
            # final layer makes rating predictions
        ])

    def call(self, features):
        user_embeddings = self.user_model({
            "key": features["key"],
            "time_signature": features["time_signature"],
            "danceability": features["danceability"],
            "energy": features["energy"],
            "loudness": features["loudness"],
            "speechiness": features["speechiness"],
            "acousticness": features["acousticness"],
            "instrumentalness": features["instrumentalness"],
            "liveness": features["liveness"],
            "valence": features["valence"],
            "tempo": features["tempo"],
            "duration_ms": features["duration_ms"],
            "mode": features["mode"]
        })
        
        song_embeddings = self.song_model({
            "key": features["key"],
            "time_signature": features["time_signature"],
            "danceability": features["danceability"],
            "energy": features["energy"],
            "loudness": features["loudness"],
            "speechiness": features["speechiness"],
            "acousticness": features["acousticness"],
            "instrumentalness": features["instrumentalness"],
            "liveness": features["liveness"],
            "valence": features["valence"],
            "tempo": features["tempo"],
            "duration_ms": features["duration_ms"],
            "mode": features["mode"]
        })

        return self.ratings(tf.concat([user_embeddings, song_embeddings], axis=1))
    
class FullRankingModel(tfrs.models.Model):
    """ The FullRankingModel takes the RankingModel and generates predicted ratings for all
    of the approximate recommendations in order to find the best songs for the user.

    Attributes
    ----------
    ranking_model: tf.keras.Model
        - The RankingModel takes in two SongModels (one query, one candidate model) to create 
    embeddings necessary for predicting song ratings.
    task: tf.keras.layers.Layer
        - The Ranking Task(MeanSquaredError, RootMeanSquaredError) for predicting the ratings of
        each of the potential song recommendations
        
    Methods
    ----------
    call(features):
        - Takes in the input dictionary of features and passes it through the input layers to return
        embeddings for all of the songs
    compute_loss(features):
        - Takes in the input dictionary of features and passes it through the task layer to return
        the predicted ratings for all of the potential song recommendations
    """
    def __init__(self, ranking_model: tf.keras.Model):
        super().__init__()
        self.ranking_model: tf.keras.Model = ranking_model
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model({
            "key": features["key"],
            "time_signature": features["time_signature"],
            "danceability": features["danceability"],
            "energy": features["energy"],
            "loudness": features["loudness"],
            "speechiness": features["speechiness"],
            "acousticness": features["acousticness"],
            "instrumentalness": features["instrumentalness"],
            "liveness": features["liveness"],
            "valence": features["valence"],
            "tempo": features["tempo"],
            "duration_ms": features["duration_ms"],
            "mode": features["mode"]
        })

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("ratings")

        rating_predictions = self(features)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)