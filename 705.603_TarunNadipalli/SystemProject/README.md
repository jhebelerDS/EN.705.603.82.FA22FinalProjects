#### Tarun Nadipalli - 705.603 Creating AI-Enabled Systems Final Project
## Spotify Recommender System

For my final project, I built my own content-based song recommendation system using TensorFlow and the Spotify API. With my application, users can paste a link to a Spotify playlist and receive a link to a newly created Spotify playlist with songs they might like! 

With the Spotify API, we can query a variety of different features for each song (e.g. tempo, danceability, key) to understand the characteristics of a song. Using this information, we employ TensorFlow models to build a profile of song preferences, retrieve similar songs and return them to the user. That being said, my application differs from Spotify's recommendation system in one important way: 

**I find recommendations without using data on the genre or artist of a song!** 

For more information on the full codebase, you can read through the following notebooks / python scripts in this repository:

1. Spotify API Utilization (`spotify.py`, `spotify_api_guide.ipynb`)
2. Data Collection and Analysis (`data.py`, `data_collection_guide.ipynb`, `data_analysis_guide.ipynb`)
3. TensorFlow Recommenders Implementation (`recommend.py`, `models.py`, `recommendation_guide.ipynb`)

### Instructions

_**Note about Data Collection first**_

If you want to run your own data collection with a modification of the `data.py` script, first ensure that there is a `spotify_data.db` file within the `data/` folder in the repo.
In addition, you will want to upload the Spotify Million Playlist Dataset from this [link](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
with the exact naming as follows: **'spotify_million_playlist_dataset.zip'**. Leave this as an uncompressed .zip file! The code will draw the data without uncompression.

Once you have those files in the `data/` folder, everything should work appropriately without changes.

_**Running the Recommender Locally**_

1. Clone the repository using the following command - 

``` bash
git clone https://github.com/tnadipa1/705.603_TarunNadipalli.git
```

2. Setup a Virtual Environment with all dependencies

The virtualenv is important because it gives you an isolated playground to download dependencies and run applications.

Navigate into the folder which contains this cloned respository and install virtualenv.
``` bash
python3 -m pip install --user virtualenv
```

Then initialize a new virtual environment called venv in your project folder called env (you can call it whatever you want, but for me consistency is helpful.

``` bash
python3 -m venv env
```

Next you will activate your virtual environment. Note that you need to activate it every time you want to use the application. 
``` bash
source env/bin/activate
```

Next, you will download all of the dependencies listed in the `requirements.txt` file that are necessary to run the application.
``` bash
pip install -r requirements.txt
```

If you want to leave the virtual environment because you are done working on the project, run the following:
``` bash
deactivate
```

3. Running the Application

At this point, we have created all the setup for the running the application. As such, all you have to do is run:
``` bash
python3 recommend.py
```

Once the application is running you will see the following printed out in your terminal:

<img width="1199" alt="image" src="https://user-images.githubusercontent.com/113858538/206937362-6de60292-3b76-464c-aa49-3e88dada1ccf.png">

Here you can paste a link to a Spotify Playlist filled with songs you like and you will get a link returned to you like this:

**NOTE**: When you run the application for the first time, you may encounter Spotify opening a link in your browser that starts with: '127.0.0.1'. If so, copy that link AS IS and paste it into the application prompt. From there on the application should work fine.

<img width="1282" alt="image" src="https://user-images.githubusercontent.com/113858538/206937533-900d4817-7a7c-4d9e-ab6e-d5a1c71ac38b.png">

Take your new Spotify playlist link and give it a listen! Hope you enjoy!



** Note that there is a docker image you can pull from my DockerHub Repository, but it is currently not working due to the constraints of the Spotify API. The best way to run this application is locally. If you would like to try running this through docker anyway, here is the command.
``` bash
docker pull tnadipa1/spotifyrecommender:1.1
```






