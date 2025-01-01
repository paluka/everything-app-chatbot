from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from typing import Tuple, List


# YouTube Data API setup
API_KEY = "AIzaSyCxPFNmQyF2TqokMtSaNQcyYmkBOuB341c"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# CHANNEL_ID = "UC_Bfb6qe1Zi_F-q4s2EvaIA"
# CHANNEL_NAME_QUERY = "Joe Rogan"
NUMBER_OF_VIDEOS = 5  # Specify the number of latest videos to fetch

youtube = build(YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION, developerKey=API_KEY)


def get_channel_id(query: str) -> Tuple[str, str]:
    """
    Get the channel ID for a given search query.
    """

    print(f"\n\nget_channel_id: {query}\n\n")

    # Perform a search query to find channels
    search_response = youtube.search().list(
        part="snippet",
        q=query,
        type="channel",
        maxResults=1  # Return the most relevant channel
    ).execute()

    print(f"\n\nget_channel_id's search_response:\n{search_response}\n\n")

    if "items" in search_response and len(search_response["items"]) > 0:
        channel_id = search_response["items"][0]["id"]["channelId"]
        channel_title = search_response["items"][0]["snippet"]["title"]
        return channel_id, channel_title
    else:
        return None, None


def get_video_info(video_id: str) -> dict:
    """
    Fetch a YouTube video's information.
    """
    try:
        video_response = youtube.videos().list(
            part="snippet,contentDetails,statistics,status",
            id=video_id
        ).execute()

        print(f"\n\get_video_info's video_response:\n{video_response}\n\n")

        if "items" in video_response and len(video_response["items"]) > 0:
            return video_response["items"][0]

    except Exception as error:
        print(error)

    return {}  # Return an empty dictionary if there's an error or no video found


def get_latest_video_ids(channel_id: str, max_results=5) -> List[str]:
    """
    Fetch the latest video IDs from a YouTube channel.
    """

    try:
        print(f"\n\get_latest_video_ids: {channel_id} {max_results}\n\n")

        # Fetch the uploads playlist ID
        channel_response = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        ).execute()

        print(
            f"\n\nget_latest_video_ids' channel_response:\n{channel_response}\n\n")

        if "items" in channel_response and len(channel_response["items"]) > 0:
            uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

            # Fetch the latest video IDs from the uploads playlist
            playlist_response = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=max_results
            ).execute()

            print(
                f"\n\nget_latest_video_ids' playlist_response:\n{playlist_response}\n\n")

            if "items" in playlist_response and len(playlist_response["items"]) > 0:
                video_ids = [item["contentDetails"]["videoId"]
                             for item in playlist_response["items"]]
                return video_ids

    except Exception as error:
        print(error)

    return []


def get_video_transcripts(video_ids: List[str]) -> dict:
    """
    Fetch transcripts for a list of video IDs.
    """
    print(
        f"\n\nget_video_transcripts:\n{video_ids}\n\n")

    transcripts = {}

    for video_id in video_ids:
        try:
            transcript_segments = YouTubeTranscriptApi.get_transcript(
                video_id)
            # combined_transcript = " ".join(
            #     entry["text"] for entry in transcripts[video_id])
            transcripts[video_id] = {"segments": transcript_segments}
        except (TranscriptsDisabled, NoTranscriptFound) as error:
            transcripts[video_id] = f"Transcript not available"
    return transcripts


if __name__ == "__main__":

    channel_name_query = input('Enter a YouTube channel name: ')

    channel_id, channel_title = get_channel_id(channel_name_query)
    print(channel_title)

    video_ids = get_latest_video_ids(
        channel_id, max_results=NUMBER_OF_VIDEOS)

    transcripts = get_video_transcripts(video_ids)

    for video_id, transcript in transcripts.items():
        print(f"Video ID: {video_id}")

        if isinstance(transcript, str):
            print(transcript)  # Prints "Transcript not available"
        else:
            for entry in transcript:
                print(f"{entry['start']} - {entry['text']}")

        print("-" * 80)
