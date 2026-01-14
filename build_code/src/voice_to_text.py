import torch
import math
import requests
import time
import json
import numpy as np

# Fix for numpy compatibility with pyannote-audio
# In NumPy 2.0+, np.NAN was removed and replaced with np.nan
# This ensures backward compatibility for pyannote-audio library
if not hasattr(np, 'NAN'):
    np.NAN = np.nan

from typing import Union, Tuple
from pyannote.audio import Pipeline
import pandas as pd
from helpers import (
    logger,
    combine_consecutive_speakers,
    combine_whisper_and_pyannote,
    segment_to_dataframe,
    text_speaker_df_to_text,
    convert_faster_whisper_segments_to_openai_segment,
    process_audio,
)

from data_body import TranscriptionOutputBody
from models import whisper_models
from config import settings
from faster_whisper.vad import VadOptions

from diarizers import SegmentationModel

hf_token = settings.HF_TOKEN
NUM_SPEAKERS = 2
COMPUTE_RATE = settings.COMPUTE_RATE_PER_SECOND

# set custom vad options for audio trimming
vad_options = VadOptions(
    threshold=0.25,
    min_speech_duration_ms=50,
    min_silence_duration_ms=500,
    speech_pad_ms=1000,
)


def get_transcripts(
    model,
    audio_file: Union[torch.Tensor, bytes, str],
    task: str = "transcribe",
    language: str = None,
) -> Tuple[str, Tuple, str, str]:
    """
    Transcribes the given audio file using the Whisper model and
    extracts segments and the full text.
    """
    logger.info("Start generating the transcription...")
    torch.cuda.empty_cache()
    # Set transcription parameters
    options_dict = {"task": task}
    options_dict["word_timestamps"] = True
    options_dict["beam_size"] = 1
    options_dict["vad_filter"] = True
    options_dict["vad_parameters"] = vad_options
    options_dict["compression_ratio_threshold"] = 3.0
    options_dict["language_detection_threshold"] = 0.5
    options_dict["language_detection_segments"] = 5

    if language is not None:
        options_dict["language"] = language

    segment_generator, info = model.transcribe(audio_file, **options_dict)

    segments = []
    text = ""
    for segment in segment_generator:
        segments.append(segment)
        text = text + segment.text
        result = {
            "language": info.language,
            "duration": info.duration,
            "segments": segments,
            "text": text,
        }
    torch.cuda.empty_cache()
    logger.info("Audio transcription done.")
    return result["text"], result["segments"], result["language"], result["duration"]


def get_diarization(token, diarization_model, waveform_sample_rate):
    """
    Diarizes the given audio file using the provided pyannote diarization model.
    """
    logger.info("Start generating the diarization...")
    # Fine-tunned segmentation model
    segmentation_model = SegmentationModel().from_pretrained(
        "diarizers-community/speaker-segmentation-fine-tuned-callhome-eng"
    )
    model = segmentation_model.to_pyannote_model()

    torch.cuda.empty_cache()
    pipeline = Pipeline.from_pretrained(diarization_model, use_auth_token=token)

    # segmentation, embedding and clustering hyperparameters
    pipeline._segmentation.model = model
    pipeline.segmentation.threshold = 0.7
    pipeline.segmentation.min_duration_off = 0.2
    pipeline.segmentation.min_duration_on = 0.1
    pipeline.segmentation.offset = 0.5
    pipeline.segmentation.onset = 0.75
    pipeline.embedding_exclusive_overlap = True
    pipeline.clustering.threshold = 0.7
    pipeline.clustering.method = "centroid"
    pipeline.segmentation_step = 0.05

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    diarization_result = pipeline(waveform_sample_rate, num_speakers=NUM_SPEAKERS)
    torch.cuda.empty_cache()

    seg_info_list = []
    for speech_turn, track, speaker in diarization_result.itertracks(yield_label=True):
        if speaker == "SPEAKER_00":
            speaker = "SPEAKER 1"
        else:
            speaker = "SPEAKER 2"
        segment_info = {
            "start": np.round(speech_turn.start, 2),
            "end": np.round(speech_turn.end, 2),
            "speaker": speaker,
        }
        segment_info_df = pd.DataFrame.from_dict({track: segment_info}, orient="index")

        seg_info_list.append(segment_info_df)
    torch.cuda.empty_cache()
    logger.info("Diarization done.")
    seg_info_df = pd.concat(seg_info_list, axis=0)

    return seg_info_df.reset_index()


def diarization_pipeline(audio_file, model, diarization_df):
    text, segments, language, duration = get_transcripts(
        model=model, audio_file=audio_file
    )
    transcripts = convert_faster_whisper_segments_to_openai_segment(segments=segments)

    whisper_df = segment_to_dataframe(transcripts)

    full_df = combine_whisper_and_pyannote(whisper_df, diarization_df)

    combine_text = combine_consecutive_speakers(full_df)
    diarized_output = text_speaker_df_to_text(combine_text)

    return diarized_output, text, language, duration


def translation_diarization_pipeline(audio_file, model, diarization_df, lang):

    text, segments, language, duration = get_transcripts(
        model=model, audio_file=audio_file, task="translate", language=lang
    )
    translated_text = convert_faster_whisper_segments_to_openai_segment(
        segments=segments
    )
    whisper_df = segment_to_dataframe(translated_text)
    full_df = combine_whisper_and_pyannote(whisper_df, diarization_df)
    combine_text = combine_consecutive_speakers(full_df)
    diarized_output = text_speaker_df_to_text(combine_text)
    return diarized_output, text


def transcription_and_diarization(
    audio_str_path,
    audio_raw_data,
    whisper_model,
    transcription_data_body: TranscriptionOutputBody,
):

    diarization_df = get_diarization(
        token=hf_token,
        diarization_model="pyannote/speaker-diarization-3.1",
        waveform_sample_rate=audio_raw_data,
    )

    (
        transcription_data_body.diarized_transcript,
        transcription_data_body.text,
        language,
        duration,
    ) = diarization_pipeline(
        audio_file=audio_str_path,
        model=whisper_model,
        diarization_df=diarization_df,
    )
    transcription_data_body.duration = round(duration, 2)
    if language != "en":

        (
            transcription_data_body.diarized_translation,
            transcription_data_body.translation,
        ) = translation_diarization_pipeline(
            audio_file=audio_str_path,
            model=whisper_model,
            diarization_df=diarization_df,
            lang=language,
        )

    transcription_data_body.language = language.upper()
    logger.info("Completed !.")
    return transcription_data_body


def process_audio_request(audio_url, extra_data, response_url):
    """
    Transform the input data and generate a transcription result
    """
    transcription_data = TranscriptionOutputBody()
    transcription_data.extra_data = extra_data
    dispatcher_response_url = f"{response_url}/transcribtion/data"
    try:
        start = time.perf_counter()
        model = whisper_models()
        logger.info("Whisper Model loaded !")
        audio_data, waveform_sample_rate = process_audio(audio_url)

        output = transcription_and_diarization(
            audio_str_path=audio_data,
            audio_raw_data=waveform_sample_rate,
            whisper_model=model,
            transcription_data_body=transcription_data,
        )


        duration = math.ceil(time.perf_counter() - start)
        cost = duration * COMPUTE_RATE

        logistics = {"taskDuration" : duration, "taskCost" : cost}

        output.extra_data["billing"] = logistics

        payload = {"data": output.dict()}
        
        logger.info(f"Elapsed time: {duration} seconds")

        print("sending to dispatcher......", dispatcher_response_url, payload)
        # requests.post(url=dispatcher_response_url, json=payload)
        # return True
        return payload

    except Exception as e:
        logger.info(f"Error: {e}")
        transcription_data = TranscriptionOutputBody().dict()
        transcription_data["extra_data"] = extra_data
        transcription_data["extra_data"]["model_error"] = str(e)
        payload = {"data": transcription_data}
        requests.post(url=dispatcher_response_url, json=payload)
        return payload
        # return True



