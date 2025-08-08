Improved AI Video Generation Backend
A simple FastAPI backend for generating videos from text prompts or user-uploaded images, using OpenAI, ElevenLabs, and Runway AI.
Features

Generate stories from user or AI-generated prompts (OpenAI).
Use user-uploaded images or generate images from story-based prompts (Runway AI).
Create audio from stories (ElevenLabs).
Combine images and audio into videos (MoviePy).
Basic logging and error handling.

Setup

Install dependencies:
pip install -r requirements.txt


Set API keys:
export OPENAI_API_KEY="your_openai_api_key"
export ELEVENLABS_API_KEY="your_elevenlabs_api_key"
export RUNWAYML_API_KEY="your_runwayml_api_key"


Run the server:
uvicorn main:app --reload



API Usage

Endpoint: POST /generate-video
Request Body (JSON):{
  "story_prompt": "A robot in a cyberpunk city...",
  "use_ai_prompt": false
}


Optional: Upload images via multipart/form-data with the images field (PNG/JPEG only).
Response (202 Accepted):{
  "status": "Video generation started",
  "output_filename": "videos/output_<hash>.mp4"
}



Notes

Replace YOUR_RUNWAYML_TEXT_TO_IMAGE_API_ENDPOINT_HERE in main.py with the actual Runway AI endpoint.
Ensure API keys are set as environment variables.
Files are saved in videos/, images/, audio/, and uploads/.
Image prompts are generated from the story for narrative sequence.

