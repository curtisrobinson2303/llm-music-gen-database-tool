import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import time


def get_spotify_credentials(credentials_file: str):
    """
    Reads Spotify API credentials from a JSON file.
    """
    with open(credentials_file, "r") as file:
        credentials = json.load(file)
    return credentials["client_id"], credentials["client_secret"]


def test_audio_features(song_id: str, credentials_file: str):
    """
    Fetches audio features for a single song from Spotify API.
    """
    client_id, client_secret = get_spotify_credentials(credentials_file)

    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="http://localhost:8080",
            scope="user-library-read playlist-read-private",
        )
    )

    # Print access token for debugging
    token_info = sp.auth_manager.get_access_token()
    print(f"Access Token: {token_info}")

    try:
        time.sleep(0.5)  # Avoid rate limits
        attributes = sp.audio_features([song_id])  # Pass as a list
        print(f"Audio features for {song_id}: {json.dumps(attributes, indent=4)}")
    except spotipy.SpotifyException as e:
        print(f"Error fetching audio features for {song_id}: {e}")


if __name__ == "__main__":
    test_song_id = "4uLU6hMCjMI75M1A2tKUQC"  # Test with a known track ID
    credentials_path = "credentials.json"  # Path to credentials file
    test_audio_features(test_song_id, credentials_path)
