import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import save
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="AI Video Generator", version="1.2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool
executor = ThreadPoolExecutor(max_workers=3)

# Pydantic models
class VideoPrompt(BaseModel):
    story_prompt: str
    use_ai_prompt: bool = False

class CustomStoryPrompt(BaseModel):
    custom_prompt: str

class ImagePromptRequest(BaseModel):
    story_text: str

class ImageGenerationRequest(BaseModel):
    image_prompts: list[str]

# Validate API keys
def validate_api_keys():
    required_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
        "RUNWAYML_API_KEY": os.getenv("RUNWAYML_API_KEY")
    }
    missing_keys = [key for key, value in required_keys.items() if not value]
    if missing_keys:
        raise ValueError(f"Missing API keys: {', '.join(missing_keys)}")

# Core pipeline functions
def _generate_story_prompt(custom_prompt: str = None):
    logger.info("Generating AI story prompt")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_message = custom_prompt or "Generate a concise sci-fi story prompt for a video."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Create a prompt for a sci-fi narrative."}
        ],
        max_tokens=100,
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()

def _generate_story(prompt: str):
    logger.info(f"Generating story for prompt: {prompt}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write a short, vivid sci-fi story based on the prompt, suitable for a video narrative."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()

def _extract_image_prompts(story_text: str):
    logger.info("Extracting image prompts from story")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract 5 key visual scenes from the provided story text in sequence, formatted as concise image prompts for a sci-fi video."},
            {"role": "user", "content": f"Story: {story_text}\n\nExtract 5 sequential visual scene prompts for image generation."}
        ],
        max_tokens=300,
        temperature=0.7,
    )
    prompts = response.choices[0].message.content.strip().split("\n")
    prompts = [p.strip("- ").strip() for p in prompts if p.strip()]
    logger.debug(f"Extracted prompts: {prompts}")
    return prompts[:5]

def _generate_images(image_prompts: list[str]):
    logger.info("Generating images with Runway AI")
    runwayml_api_key = os.getenv("RUNWAYML_API_KEY")
    if not runwayml_api_key:
        raise ValueError("RUNWAYML_API_KEY not set")

    api_url = "YOUR_RUNWAYML_TEXT_TO_IMAGE_API_ENDPOINT_HERE"  # Replace with actual endpoint
    image_paths = []
    
    for i, prompt in enumerate(image_prompts):
        try:
            logger.debug(f"Generating image {i+1}: {prompt}")
            headers = {"Authorization": f"Bearer {runwayml_api_key}", "Content-Type": "application/json"}
            payload = {"prompt": prompt, "resolution": "720p"}
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            image_url = response.json().get("image_url")
            
            if image_url:
                image_response = requests.get(image_url)
                image_response.raise_for_status()
                file_path = f"images/image_{i+1}.png"
                os.makedirs("images", exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(image_response.content)
                image_paths.append(file_path)
                logger.debug(f"Image saved: {file_path}")
            else:
                logger.warning(f"No image URL for prompt: {prompt}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating image {i+1}: {e}")
    
    if not image_paths:
        raise ValueError("No images generated")
    return image_paths

def _generate_audio(story_text: str):
    logger.info("Generating audio with ElevenLabs")
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    audio_stream = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",
        model_id="eleven_multilingual_v2",
        text=story_text,
        output_format="mp3_44100_128",
    )
    audio_path = "audio/story_audio.mp3"
    os.makedirs("audio", exist_ok=True)
    save(audio_stream, audio_path)
    logger.debug(f"Audio saved: {audio_path}")
    return audio_path

def _create_video(image_paths: list[str], audio_path: str, output_filename: str):
    logger.info(f"Creating video with {len(image_paths)} images")
    if not image_paths:
        raise ValueError("No images available")
    
    audio_clip = AudioFileClip(audio_path)
    duration_per_image = audio_clip.duration / len(image_paths)
    image_clips = [
        ImageClip(path, duration=duration_per_image)
        .set_position("center")
        .resize(width=1280, height=720)
        for path in image_paths
    ]
    final_video = concatenate_videoclips(image_clips, method="compose").set_audio(audio_clip)
    
    os.makedirs("videos", exist_ok=True)
    final_video.write_videofile(output_filename, fps=24, codec="libx264", audio_codec="aac")
    logger.info(f"Video created: {output_filename}")

async def video_generation_pipeline(story_prompt: str, use_ai_prompt: bool, user_image_paths: list[str], output_filename: str, custom_prompt: str = None):
    try:
        validate_api_keys()
        
        if use_ai_prompt:
            story_prompt = await asyncio.get_event_loop().run_in_executor(executor, _generate_story_prompt, custom_prompt)
            logger.info(f"AI-generated prompt: {story_prompt}")
        
        story_text = await asyncio.get_event_loop().run_in_executor(executor, _generate_story, story_prompt)
        
        image_paths = user_image_paths
        if not image_paths:
            image_prompts = await asyncio.get_event_loop().run_in_executor(executor, _extract_image_prompts, story_text)
            image_paths = await asyncio.get_event_loop().run_in_executor(executor, _generate_images, image_prompts)
        
        audio_path = await asyncio.get_event_loop().run_in_executor(executor, _generate_audio, story_text)
        
        await asyncio.get_event_loop().run_in_executor(executor, _create_video, image_paths, audio_path, output_filename)
        
        return story_text
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise

@app.post("/generate-video", status_code=202)
async def generate_video(prompt: VideoPrompt, background_tasks: BackgroundTasks, images: list[UploadFile] = File(None)):
    try:
        uploaded_image_paths = []
        if images:
            os.makedirs("uploads", exist_ok=True)
            for i, image in enumerate(images):
                if not image.content_type.startswith("image/"):
                    raise HTTPException(status_code=400, detail=f"File {image.filename} is not an image")
                image_path = f"uploads/image_{i+1}_{image.filename}"
                with open(image_path, "wb") as f:
                    f.write(await image.read())
                uploaded_image_paths.append(image_path)
                logger.debug(f"Uploaded image: {image_path}")
        
        output_filename = f"videos/output_{hash(prompt.story_prompt) & 0xFFFFFFF}.mp4"
        story_text = await video_generation_pipeline(prompt.story_prompt, prompt.use_ai_prompt, uploaded_image_paths, output_filename)
        
        return {
            "status": "Video generation started",
            "output_filename": output_filename,
            "story_text": story_text
        }
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-story-prompt", status_code=200)
async def generate_custom_story_prompt(prompt: CustomStoryPrompt):
    try:
        if not prompt.custom_prompt.strip():
            raise HTTPException(status_code=400, detail="Custom prompt cannot be empty")
        story_prompt = await asyncio.get_event_loop().run_in_executor(executor, _generate_story_prompt, prompt.custom_prompt)
        return {"story_prompt": story_prompt}
    except Exception as e:
        logger.error(f"Custom story prompt error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-image-prompts", status_code=200)
async def get_image_prompts(request: ImagePromptRequest):
    try:
        image_prompts = await asyncio.get_event_loop().run_in_executor(executor, _extract_image_prompts, request.story_text)
        return {"image_prompts": image_prompts}
    except Exception as e:
        logger.error(f"Image prompts error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-images", status_code=200)
async def get_images(request: ImageGenerationRequest):
    try:
        image_paths = await asyncio.get_event_loop().run_in_executor(executor, _generate_images, request.image_prompts)
        return {"image_urls": [f"file://{path}" for path in image_paths]}
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-audio", status_code=200)
async def get_audio():
    audio_path = "audio/story_audio.mp3"
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/mpeg", filename="story_audio.mp3")

@app.get("/get-video", status_code=200)
async def get_video(filename: str):
    video_path = filename
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(video_path, media_type="video/mp4", filename=os.path.basename(video_path))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)