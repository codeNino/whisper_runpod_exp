import requests
import tempfile
import os
import io
import time
import torchaudio
import logging
import pandas as pd
import numpy as np
from fastapi import HTTPException
from pydub import AudioSegment
from typing import List, Tuple

from data_body import PayLoadData, TranscriptionOutputBody

# Fix for numpy compatibility with pyannote-audio
# This ensures backward compatibility for pyannote-audio library
if not hasattr(np, 'NAN'):
    np.NAN = np.nan

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(asctime)s - %(message)s"
)


def get_logs():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(asctime)s - %(message)s"
    )
    return logging


logger = get_logs()


def get_audio_data(audio_url: str) -> str:
    try:
        logger.info("Downloading Audio...")
        # audio_bytes = requests.get(audio_url).content
        response = requests.post(
            url="http://18.194.85.55:3001/voip-call-record/bite-data",
            json={"fileUrl": audio_url},
        ).json()
        try:
            audio_data = response["data"]
            audio_bytes = bytes(audio_data)
        except KeyError:
            raise HTTPException(
                status_code=400, detail="'data' key not found in response"
            )

    except requests.exceptions.ConnectionError:
        logger.error("connection error in downloading audio, check network connection of audio download server.")
        return {"error": "connection error in downloading audio, check network connection of audio download server."}
    except requests.exceptions.Timeout:
        logger.error("request timeout in downloading audio, check network connection of audio download server.")
        return {"error": "request timeout in downloading audio, check network connection of audio download server."}
    except requests.exceptions.HTTPError:
        logger.error("HTTP Error from audio_url")
        return {"error": "HTTP Error from audio_url"}
    except requests.exceptions.RequestException:
        logger.error("An error occurred, check audio url")
        return {"error": "An error occurred, check audio url"}

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        with open(temp_file.name, "wb") as f:
            f.write(audio_bytes)
            if os.path.exists(temp_file.name):
                with open(temp_file.name, "rb") as f:
                    file_reader = io.BufferedReader(f)
                    audio_data = temp_file.name
                    temp_file.close()

    return audio_data


def process_audio(audio_url, target_dBFS=-15.0):
    """
    This function processes an audio file from a given URL, 
    normalizes its volume, and converts it into a TorchAudio tensor.

    Parameters:
    - audio_url (str): The URL of the audio file to be processed.
    - target_dBFS (float, optional): The target decibel-relative to 
            full scale (dBFS) for the audio volume. Default is -15.0.

    Returns:
    - tuple: A tuple containing the audio data file path and a 
             dictionary with the processed waveform and sample rate.
    """
    
    
    audio_data = get_audio_data(audio_url=audio_url)
    if isinstance(audio_data, dict):
        error_message = audio_data.get("error", "Unknown error")
        raise ValueError(error_message)
    audio = AudioSegment.from_file(audio_data)

    # Compute gain to normalize audio
    change_in_dBFS = target_dBFS - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)

    temp_wav = io.BytesIO()
    normalized_audio.export(temp_wav, format="wav")
    temp_wav.seek(0)

    # Load into TorchAudio tensor
    waveform, sample_rate = torchaudio.load(temp_wav)
        # Resample if needed
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
        
    waveform_sample_rate = {"waveform": waveform, "sample_rate": sample_rate}

    return audio_data, waveform_sample_rate


def convert_faster_whisper_segments_to_openai_segment(segments):
    openai_segments = []
    for segment in segments:
        id, _, start, end, text, _, _, _, _, _, words = segment
        # create a new segment from each word stamp
        # for index, word in enumerate(words):
        #     word_start, word_end, word_text, prob = word

        openai_segments.append({"id": id - 1, "start": start, "end": end, "text": text})

    return openai_segments


def combine_whisper_and_pyannote(text_df, speaker_df):

    text_df = text_df.loc[:, ["id", "start", "end", "text"]]
    speaker_df = speaker_df.loc[:, ["index", "start", "end", "speaker"]]

    overlap_list = []
    for idx, pyannote_row in speaker_df.iterrows():
        pyannote_start = pyannote_row["start"]
        pyannote_end = pyannote_row["end"]
        pyannote_speaker = pyannote_row["speaker"]

        xx_inds = ~(
            (text_df["end"] < pyannote_start) | (text_df["start"] > pyannote_end)
        )
        this_overlap_texts = text_df.loc[xx_inds, :]

        this_overlap_texts["speaker_start"] = pyannote_start
        this_overlap_texts["speaker_end"] = pyannote_end
        this_overlap_texts["speaker"] = pyannote_speaker

        overlap_list.append(this_overlap_texts)

    all_overlaps = pd.concat(overlap_list)
    all_overlaps = all_overlaps.reset_index(drop=True)

    all_overlaps["max_start"] = np.maximum(
        all_overlaps["start"], all_overlaps["speaker_start"]
    )
    all_overlaps["min_end"] = np.minimum(
        all_overlaps["end"], all_overlaps["speaker_end"]
    )
    all_overlaps["overlap_duration"] = (
        all_overlaps["min_end"] - all_overlaps["max_start"]
    )


    max_overlap_indices = all_overlaps.groupby("id")["overlap_duration"].idxmax()
    text_speaker_df = all_overlaps.loc[max_overlap_indices, :]

    return text_speaker_df


def convert_whisper_output(whisper_output, audio_duration):
    """Converts Whisper output to a dictionary of segments with start and end timestamps."""

    segments = []
    for i, item in enumerate(whisper_output):

        start = item["timestamp"][0]
        end = (
            audio_duration
            if item["timestamp"][1] is None and i == len(whisper_output) - 1
            else item["timestamp"][1]
        )
        text = item["text"]

        segments.append({"id": i, "start": start, "end": end, "text": text})

    print(f"the last value for i is :{i}")
    return segments


def segment_to_dataframe(transcription_result):
    """
    Converts the segments into a pandas DataFrame with columns for
    segment start, end, and text.
    """
    df = pd.DataFrame(transcription_result, columns=["id", "start", "end", "text"])
    df["start"] = df["start"].apply(lambda x: round(x, 2))
    df["end"] = df["end"].apply(lambda x: round(x, 2))
    return df


def combine_consecutive_speakers(text_speaker_df_raw):

    text_speaker_df = text_speaker_df_raw.copy()

    n_iter = text_speaker_df.shape[0]

    for counter in range(1, n_iter):

        is_same_speaker = (
            text_speaker_df["speaker"].iloc[counter]
            == text_speaker_df["speaker"].iloc[counter - 1]
        )

        if is_same_speaker:

            new_start = text_speaker_df["start"].iloc[counter - 1]
            previous_text = text_speaker_df["text"].iloc[counter - 1]
            new_text = previous_text + " " + text_speaker_df["text"].iloc[counter]

            text_speaker_df["start"].iloc[counter] = new_start
            text_speaker_df["text"].iloc[counter] = new_text
            text_speaker_df["start"].iloc[counter - 1] = np.nan
            text_speaker_df["end"].iloc[counter - 1] = np.nan

    text_speaker_df = text_speaker_df.dropna().loc[
        :, ["start", "end", "text", "speaker"]
    ]
    text_speaker_df = text_speaker_df.reset_index(drop=True)
    text_speaker_df = text_speaker_df.sort_values("start")
    return text_speaker_df


def text_speaker_df_to_text(text_speaker_df):

    output_str = ""

    for idx, this_row in text_speaker_df.iterrows():

        this_start =  time.strftime("%H:%M:%S", time.gmtime(np.round(this_row['start'], 2)))
        this_end = time.strftime("%H:%M:%S", time.gmtime(np.round(this_row['end'], 2)))
        this_speaker = this_row["speaker"]
        this_text = this_row["text"]

        output_str += f'{this_speaker}: [{this_start} - {this_end}]--'
        # output_str += f"{this_speaker}: "
        output_str += f"{this_text}\n"

    return output_str


def insert_timestamp(segments):
    transcript = ""
    for i, segment in enumerate(segments):
        transcript += (
            "\n"
            + "["
            + time.strftime("%H:%M:%S", time.gmtime(segment["start"]))
            + "]"
            + " "
        )
        transcript += segment["text"][1:] + " "

    return transcript

def transform_to_string(call_transcript: List[str]) -> Tuple[str, str]:
    """
    Converts a list of strings to a single string with new line between each item.
    """
    diarized_transcript = "\n".join(call_transcript)
    transcript = " ".join(call_transcript)
    return diarized_transcript, transcript


def process_ml_agent(payload_data: PayLoadData) -> bool:
    """
    Process the ml agent call transcript data and send it to the dispatcher
    """
    try:
        is_ml_agent = payload_data.is_ml_agent
        language : str = payload_data.extra_data.get("languages", {}).get("defaultLanguage", "EN")
        dispatcher_endpoint : str = f"{payload_data.dispatcher_endpoint}/transcribtion/data"
        logger.info(f"Dispatcher endpoint: {dispatcher_endpoint}")

        extra_data = dict(payload_data.extra_data)
        extra_data["is_ml_agent"] = is_ml_agent
        diarized_transcript, transcript = transform_to_string(payload_data.transcript)
        transcription_data = TranscriptionOutputBody(
            text=transcript,
            diarized_transcript=diarized_transcript,
            translation="",
            diarized_translation="",
            duration=0,
            language=language,
            extra_data=extra_data
        ).model_dump()
        payload = {"data": transcription_data}
        requests.post(url=dispatcher_endpoint, json=payload)
        logger.info(f"Transcription data sent to dispatcher: {payload}")
        return True
    except Exception as e:
        logger.error(f"Error in processing ml agent: {e}")
        return False