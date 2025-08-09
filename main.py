# app.py
import os
import uuid
import shutil
import logging
import asyncio
import tempfile
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles
import requests
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import save as eleven_save
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ai-video-generator")

# ---------- Config from env ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY")
RUNWAYML_API_URL = os.getenv("RUNWAYML_API_URL")  # e.g. "https://api.runwayml.example/v1/text-to-image"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))
TEMP_ROOT = os.getenv("TEMP_ROOT", "jobs")  # folder to store job assets
os.makedirs(TEMP_ROOT, exist_ok=True)

# ---------- Basic validation on startup ----------
def validate_api_keys_at_startup(raise_on_missing: bool = False):
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if not RUNWAYML_API_KEY or not RUNWAYML_API_URL:
        # allow runway to be missing if user will upload images manually; but warn
        logger.warning("RUNWAYML_API_KEY or RUNWAYML_API_URL not set — image generation via Runway will fail unless provided.")
    if missing and raise_on_missing:
        raise RuntimeError(f"Missing required keys: {', '.join(missing)}")

validate_api_keys_at_startup()

# ---------- FastAPI app ----------
app = FastAPI(title="AI Video Generator", version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Thread executor for CPU/blocking tasks (moviepy, sync HTTP calls)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ---------- Job store (in-memory). For production replace with Redis or DB ----------
jobs: Dict[str, Dict[str, Any]] = {}

# ---------- Pydantic models ----------
class VideoPrompt(BaseModel):
    story_prompt: Optional[str] = None
    use_ai_prompt: bool = False  # whether to use AI to craft the story prompt
    min_images: Optional[int] = None  # optional override for minimum images (default = scenes count)

class GenerateResponse(BaseModel):
    job_id: str
    message: str

class StoryPromptRequest(BaseModel):
    custom_prompt: str

class ImagePromptRequest(BaseModel):
    story_text: str

class ImageGenerationRequest(BaseModel):
    image_prompts: List[str]

# ---------- Helper utilities ----------
def make_job_folder(job_id: str) -> str:
    path = os.path.join(TEMP_ROOT, job_id)
    os.makedirs(path, exist_ok=True)
    return path

def cleanup_job_folder(job_id: str):
    path = os.path.join(TEMP_ROOT, job_id)
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            logger.info(f"Cleaned up job folder {path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {path}: {e}")

def save_bytes_to_file(path: str, content: bytes):
    with open(path, "wb") as f:
        f.write(content)

# ---------- OpenAI wrapper functions ----------
def _openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")
    return OpenAI(api_key=OPENAI_API_KEY)

def generate_story_prompt_ai(custom_prompt: Optional[str] = None) -> str:
    logger.info("Generating story prompt via OpenAI")
    client = _openai_client()
    system_message = custom_prompt or "Generate a concise sci-fi story prompt for a short video (3-6 scenes)."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Create a prompt for a concise sci-fi narrative suitable for a short video (3-6 scenes)."}
        ],
        max_tokens=120,
        temperature=0.9,
    )
    return resp.choices[0].message.content.strip()

def generate_story_from_prompt(prompt: str) -> str:
    logger.info("Generating story text via OpenAI")
    client = _openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write a short, vivid sci-fi story divided into sequential scenes. Keep it suitable for a 30-90 second video and include scene breaks."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()

def extract_scene_prompts_from_story(story_text: str, scenes_target: int = 5) -> List[str]:
    logger.info("Extracting scene prompts via OpenAI")
    client = _openai_client()
    system = "Extract sequential visual scene descriptions from the story. Output each scene as a concise prompt (one per line)."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Story:\n{story_text}\n\nExtract up to {scenes_target} sequential scene prompts (one per line)."}
        ],
        max_tokens=300,
        temperature=0.6,
    )
    text = resp.choices[0].message.content.strip()
    lines = [l.strip("- ").strip() for l in text.splitlines() if l.strip()]
    # ensure we return at most scenes_target prompts
    return lines[:scenes_target] if lines else []

# ---------- Runway (text-to-image) ----------
def generate_image_via_runway(prompt: str, out_path: str, retry: int = 1) -> str:
    """
    Sends prompt to RUNWAYML_API_URL and saves returned image bytes to out_path.
    Expects JSON response with 'image_url' or binary image body depending on API.
    """
    if not RUNWAYML_API_KEY or not RUNWAYML_API_URL:
        raise RuntimeError("Runway endpoint/key not configured")
    headers = {"Authorization": f"Bearer {RUNWAYML_API_KEY}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "resolution": "720p"}
    for attempt in range(retry + 1):
        try:
            r = requests.post(RUNWAYML_API_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            image_url = data.get("image_url")
            if image_url:
                ir = requests.get(image_url, timeout=60)
                ir.raise_for_status()
                save_bytes_to_file(out_path, ir.content)
                return out_path
            # Some APIs return the image directly:
            if 'image' in data:
                # if it's base64, decode, etc. (left generic)
                # For now check for content-type header:
                if r.headers.get("content-type", "").startswith("image/"):
                    save_bytes_to_file(out_path, r.content)
                    return out_path
            raise RuntimeError("Runway returned no image_url")
        except Exception as e:
            logger.warning(f"Runway attempt {attempt} failed: {e}")
            if attempt == retry:
                raise

# ---------- ElevenLabs audio ----------
def generate_audio_via_elevenlabs(story_text: str, out_path: str) -> str:
    logger.info("Generating audio via ElevenLabs")
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not configured")
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    # voice and model should be adjusted to your account's available voices
    stream = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",
        model_id="eleven_multilingual_v2",
        text=story_text,
        output_format="mp3_44100_128",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    eleven_save(stream, out_path)
    return out_path

# ---------- Video creation with moviepy (blocking) ----------
def create_video_from_assets(image_paths: List[str], audio_path: str, output_path: str) -> str:
    if not image_paths:
        raise RuntimeError("No images provided to create video.")
    audio_clip = AudioFileClip(audio_path)
    duration_per_image = max(1.0, audio_clip.duration / len(image_paths))
    image_clips = []
    for p in image_paths:
        ic = ImageClip(p, duration=duration_per_image).set_position("center").resize(height=720)
        image_clips.append(ic)
    final = concatenate_videoclips(image_clips, method="compose").set_audio(audio_clip)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    # close clips
    final.close()
    audio_clip.close()
    for ic in image_clips:
        try:
            ic.close()
        except Exception:
            pass
    return output_path

# ---------- Orchestration (background job) ----------
async def run_generation_job(job_id: str, prompt_text: Optional[str], use_ai_prompt: bool,
                             uploaded_images: List[str], uploaded_audio: Optional[str],
                             override_min_images: Optional[int]) -> None:
    job = jobs[job_id]
    job["status"] = "running"
    folder = job["folder"]
    progress = job["progress"]
    try:
        # 1) Determine story prompt
        if use_ai_prompt or not prompt_text:
            progress.append("generating_story_prompt")
            loop = asyncio.get_event_loop()
            prompt_text = await loop.run_in_executor(executor, generate_story_prompt_ai, prompt_text)
            job["meta"]["story_prompt_generated"] = True
        job["meta"]["story_prompt"] = prompt_text

        # 2) Generate story text via OpenAI
        progress.append("generating_story_text")
        loop = asyncio.get_event_loop()
        story_text = await loop.run_in_executor(executor, generate_story_from_prompt, prompt_text)
        job["meta"]["story_text"] = story_text

        # 3) Extract scene prompts
        progress.append("extracting_scene_prompts")
        scene_prompts = await loop.run_in_executor(executor, extract_scene_prompts_from_story, story_text, 6)
        if not scene_prompts:
            # fallback: make simple splits of story into 4 scenes
            scene_prompts = [f"Scene {i+1}: key visual from story" for i in range(4)]
        job["meta"]["scene_prompts"] = scene_prompts
        required_images = max(len(scene_prompts), (override_min_images or 0))

        # 4) Prepare images: use uploaded + generate rest (if allowed)
        progress.append("preparing_images")
        image_paths: List[str] = []
        # use uploaded images first
        for p in uploaded_images:
            image_paths.append(p)
        # if fewer than required, attempt to generate more via Runway
        to_generate = required_images - len(image_paths)
        if to_generate > 0:
            if not RUNWAYML_API_KEY or not RUNWAYML_API_URL:
                raise RuntimeError(f"Need {to_generate} images but Runway config not provided; either upload images >= {required_images} or set RUNWAYML_API_URL & RUNWAYML_API_KEY.")
            for i in range(to_generate):
                prompt = scene_prompts[len(image_paths) % len(scene_prompts)]
                outp = os.path.join(folder, f"image_{len(image_paths)+1}.png")
                # blocking call; run in thread pool
                await loop.run_in_executor(executor, generate_image_via_runway, prompt, outp, 1)
                image_paths.append(outp)

        # 5) Audio: if uploaded use it, else generate via ElevenLabs
        progress.append("preparing_audio")
        if uploaded_audio:
            audio_path = uploaded_audio
        else:
            # generate audio file
            audio_path = os.path.join(folder, "story_audio.mp3")
            await loop.run_in_executor(executor, generate_audio_via_elevenlabs, story_text, audio_path)

        # 6) Create video (blocking)
        progress.append("creating_video")
        output_video = os.path.join(folder, "output.mp4")
        await loop.run_in_executor(executor, create_video_from_assets, image_paths, audio_path, output_video)

        # success
        job["status"] = "completed"
        job["result"] = {"video_path": output_video, "story_text": story_text, "scene_prompts": scene_prompts, "images": image_paths, "audio": audio_path}
        progress.append("done")
        logger.info(f"Job {job_id} finished successfully.")
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.exception(f"Job {job_id} failed: {e}")

# ---------- API Endpoints ----------
@app.post("/generate-video", response_model=GenerateResponse, status_code=202)
async def generate_video(
    background_tasks: BackgroundTasks,
    story_prompt: Optional[str] = Form(None),
    use_ai_prompt: bool = Form(False),
    min_images: Optional[int] = Form(None),
    images: Optional[List[UploadFile]] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    """
    Start a video generation job.
    - Provide either story_prompt (text) or set use_ai_prompt=true to auto-create story prompt via OpenAI.
    - Optionally upload images (one or more).
    - Optionally upload audio file (mp3).
    - If uploaded images < required scenes, Runway will be used to generate the remainder (requires RUNWAYML_API_URL & RUNWAYML_API_KEY).
    """
    job_id = str(uuid.uuid4())
    folder = make_job_folder(job_id)
    jobs[job_id] = {"status": "queued", "progress": [], "meta": {}, "folder": folder}

    # save uploaded images
    uploaded_image_paths: List[str] = []
    if images:
        for i, up in enumerate(images):
            if not up.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"{up.filename} is not an image")
            ext = os.path.splitext(up.filename)[1] or ".png"
            fname = os.path.join(folder, f"uploaded_image_{i+1}{ext}")
            async with aiofiles.open(fname, "wb") as f:
                content = await up.read()
                await f.write(content)
            uploaded_image_paths.append(fname)
            logger.info(f"Saved uploaded image {fname}")

    # save uploaded audio
    uploaded_audio_path: Optional[str] = None
    if audio:
        if not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail=f"{audio.filename} is not an audio file")
        ext = os.path.splitext(audio.filename)[1] or ".mp3"
        fname = os.path.join(folder, f"uploaded_audio{ext}")
        async with aiofiles.open(fname, "wb") as f:
            content = await audio.read()
            await f.write(content)
        uploaded_audio_path = fname
        logger.info(f"Saved uploaded audio {fname}")

    # register job meta
    jobs[job_id]["meta"].update({"requested_prompt": story_prompt, "use_ai_prompt": use_ai_prompt, "min_images_override": min_images})
    # start background task
    background_tasks.add_task(asyncio.create_task, run_generation_job(job_id, story_prompt, use_ai_prompt, uploaded_image_paths, uploaded_audio_path, min_images))
    jobs[job_id]["status"] = "started"
    return GenerateResponse(job_id=job_id, message="Job started. Poll /jobs/{job_id} for status.")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", []),
        "meta": job.get("meta", {}),
        "error": job.get("error"),
        "result": {k: (v if k != "video_path" else os.path.basename(v)) for k, v in job.get("result", {}).items()} if job.get("result") else None
    }

@app.get("/jobs/{job_id}/download")
async def download_job_video(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed (current status: {job['status']})")
    video_path = job["result"].get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(video_path, media_type="video/mp4", filename=f"{job_id}.mp4")

@app.post("/generate-story-prompt")
async def generate_custom_story_prompt(request: StoryPromptRequest):
    if not request.custom_prompt.strip():
        raise HTTPException(status_code=400, detail="Custom prompt cannot be empty")
    try:
        loop = asyncio.get_event_loop()
        prompt = await loop.run_in_executor(executor, generate_story_prompt_ai, request.custom_prompt)
        return {"story_prompt": prompt}
    except Exception as e:
        logger.exception("Error generating custom story prompt")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-image-prompts")
async def get_image_prompts(request: ImagePromptRequest):
    try:
        loop = asyncio.get_event_loop()
        prompts = await loop.run_in_executor(executor, extract_scene_prompts_from_story, request.story_text, 6)
        return {"image_prompts": prompts}
    except Exception as e:
        logger.exception("Error extracting image prompts")
        raise HTTPException(status_code=500, detail=str(e))

# Simple health and cleanup endpoints
@app.get("/health")
async def health():
    return {"status": "ok", "jobs_active": len(jobs)}

@app.post("/jobs/{job_id}/cleanup")
async def cleanup_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    cleanup_job_folder(job_id)
    jobs.pop(job_id, None)
    return {"status": "deleted"}

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
