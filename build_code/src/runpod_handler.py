"""
RunPod Serverless Handler for Whisper Transcription Service

This handler is designed to work with RunPod's serverless infrastructure.
It processes audio transcription requests and ML agent requests.
"""

import json
import warnings
from typing import Dict, Any
from voice_to_text import process_audio_request
from data_body import PayLoadData
from helpers import process_ml_agent

warnings.filterwarnings("ignore")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.
    
    Expected input format:
    {
        "input": {
            "audio_url": "https://example.com/audio.mp3",  # Optional
            "extra_data": {},  # Optional
            "transcript": [],  # Optional, for ML agent
            "is_ml_agent": false,  # Optional, defaults to False
            "dispatcher_endpoint": "https://example.com/api"  # Optional
        }
    }
    
    Returns:
    {
            "data": {
                "text": "...",
                "diarized_transcript": "...",
                "translation": "...",
                "diarized_translation": "...",
                "duration": 123.45,
                "language": "en",
                "extra_data": {}
            }
        }
    }
    """
    try:
        input_data = event.get("input", {})
        
        if not input_data.get("audio_url") and not input_data.get("transcript"):
            return {
                "error": "Either 'audio_url' or 'transcript' must be provided in the input"
            }

        payload = PayLoadData(
            audio_url=input_data.get("audio_url"),
            extra_data=input_data.get("extra_data", {}),
            transcript=input_data.get("transcript"),
            is_ml_agent=input_data.get("is_ml_agent", False),
            dispatcher_endpoint=input_data.get("dispatcher_endpoint")
        )

        if payload.is_ml_agent:
            result = process_ml_agent(payload)
            if result:
                return {
                    "output": {
                        "status": "success",
                        "message": "ML agent request processed successfully"
                    }
                }
            else:
                return {
                    "error": "Failed to process ML agent request"
                }
        else:
            if not payload.audio_url:
                return {
                    "error": "audio_url is required for audio transcription requests"
                }
            dispatcher = payload.dispatcher_endpoint or "http://localhost"
            
            result = process_audio_request(
                payload.audio_url,
                payload.extra_data,
                dispatcher
            )

            if isinstance(result, dict) and "data" in result:
                return result
            else:
                return {"data": result}
                
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__
        }


# For local testing
if __name__ == "__main__":
    # Example test event
    test_event = {
        "input": {
            "audio_url": "https://f616738f-backend.dataconect.com/api/v1/call-record-ext/documents/download/72681419-cb14-4a24-a301-6f947d5e50aa/CallRecord_1764594250722.mp3",
            "extra_data": {},
            "is_ml_agent": False,
            "dispatcher_endpoint": "https://example.com/api"
        }
    }
    
    result = handler(test_event)
    print(json.dumps(result, indent=2))

