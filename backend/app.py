"""
Indian Bird Identification API

Model:
johnlian/WingCLIP-0.3

Features:
1. Bird image upload
2. Image validation and preprocessing
3. Image quality assessment
4. Bird species identification
5. Brief bird information retrieval

WingCLIP similarity scores are not probabilities.
"""



from __future__ import annotations
import asyncio
import time
import uuid
import io
import logging
import os
import threading

from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from huggingface_hub import hf_hub_download

from PIL import (
    Image,
    ImageFilter,
    ImageOps,
    ImageStat,
    UnidentifiedImageError
)

from starlette.concurrency import run_in_threadpool

from timm.data import (
    create_transform,
    resolve_data_config
)

from bird_database import (
    build_bird_information,
    load_species_database
)

from llm_service import (
    GEMINI_API_KEY,
    GEMINI_ENABLE_SEARCH,
    GEMINI_MODEL,
    research_bird
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_REPO = os.getenv(
    "WINGCLIP_REPO",
    "johnlian/WingCLIP-0.3"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

MAX_IMAGE_PIXELS = 40_000_000

SUPPORTED_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP"
}

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("indian-bird-api")

inference_lock = threading.Lock()


# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

model: WingCLIP | None = None

preprocessor: Any = None

classifier_embeddings: torch.Tensor | None = None

species_labels: list[list[str]] = []

species_database: dict[str, dict[str, Any]] = {}


# ============================================================
# WINGCLIP ARCHITECTURE
# ============================================================

class WingCLIP(torch.nn.Module):

    """
    Official WingCLIP-0.3 visual encoder.

    Architecture:
    TinyCLIP ViT visual encoder
    Linear projection: 512 -> 768
    L2 normalized image embedding
    """

    def __init__(self):

        super().__init__()

        self.visual = timm.create_model(
            "vit_medium_patch16_clip_224.tinyclip_yfcc15m",
            pretrained=False,
            num_classes=0
        )

        self.proj = torch.nn.Linear(
            512,
            768
        )

    def forward(self, images):

        return F.normalize(
            self.proj(self.visual(images)),
            dim=-1
        )


# ============================================================
# DOWNLOAD MODEL FILES
# ============================================================

def download_model_file(filename: str):

    """
    Download a public Hugging Face repository file.

    token=False avoids using the expired saved token.
    """

    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=filename,
        token=False
    )


# ============================================================
# LOAD WINGCLIP
# ============================================================

def load_wingclip():

    logger.info("Loading model: %s", MODEL_REPO)

    logger.info("Using device: %s", DEVICE)

    weights_path = download_model_file(
        "wingclip-0.3.pt"
    )

    embeddings_path = download_model_file(
        "text_classifier_fp32.npy"
    )

    labels_path = download_model_file(
        "labels.json"
    )

    # Build the official model architecture.

    wingclip = WingCLIP()

    # The official checkpoint stores weights under "model".
    # Use only the checkpoint from the trusted official model repository.

    checkpoint = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:

        raise RuntimeError(
            "Unexpected WingCLIP checkpoint format."
        )

    wingclip.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    wingclip.eval()

    wingclip.to(DEVICE)

    # Load model-specific image preprocessing.

    data_config = resolve_data_config(
        {},
        model=wingclip.visual
    )

    image_transform = create_transform(
        **data_config,
        is_training=False
    )

    # Load species text embeddings.

    embedding_array = np.load(
        embeddings_path,
        allow_pickle=False
    )

    if embedding_array.ndim != 2 or embedding_array.shape[1] != 768:

        raise RuntimeError(
            f"Unexpected embedding shape: {embedding_array.shape}"
        )

    embedding_tensor = torch.from_numpy(
        np.asarray(
            embedding_array,
            dtype=np.float32
        )
    )

    embedding_tensor = F.normalize(
        embedding_tensor,
        dim=-1
    ).to(DEVICE)

    # Load species labels.

    import json

    with open(
        labels_path,
        "r",
        encoding="utf-8"
    ) as label_file:

        raw_labels = json.load(label_file)

    if (
        not isinstance(raw_labels, list)
        or len(raw_labels) != embedding_tensor.shape[0]
    ):

        raise RuntimeError(
            "Species labels and embedding catalogue have different lengths."
        )

    cleaned_labels = []

    for index, row in enumerate(raw_labels):

        if not isinstance(row, (list, tuple)) or len(row) < 2:

            raise RuntimeError(
                f"Invalid species label at row {index}."
            )

        cleaned_labels.append([
            str(row[0]),
            str(row[1]),
            str(row[2]) if len(row) > 2 and row[2] else ""
        ])

    logger.info(
        "WingCLIP loaded with %d species.",
        len(cleaned_labels)
    )

    return (
        wingclip,
        image_transform,
        embedding_tensor,
        cleaned_labels
    )


# ============================================================
# IMAGE VALIDATION AND DECODING
# ============================================================

def decode_image(image_bytes: bytes) -> Image.Image:

    """
    Decode uploaded image safely.

    Processing:
    1. Verify actual file format.
    2. Validate dimensions.
    3. Apply EXIF orientation.
    4. Convert to RGB.
    """

    try:

        with Image.open(
            io.BytesIO(image_bytes)
        ) as opened:

            actual_format = (
                opened.format or ""
            ).upper()

            if actual_format not in SUPPORTED_FORMATS:

                raise HTTPException(
                    status_code=415,
                    detail=(
                        "Unsupported image. Upload JPEG, PNG, WEBP or BMP."
                    )
                )

            width, height = opened.size

            if (
                width < 1
                or height < 1
                or width * height > MAX_IMAGE_PIXELS
            ):

                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are invalid or too large."
                )

            opened.seek(0)

            opened.load()

            corrected = ImageOps.exif_transpose(
                opened
            )

            return corrected.convert("RGB")

    except HTTPException:

        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid readable image."
        ) from exc


# ============================================================
# IMAGE QUALITY ASSESSMENT
# ============================================================

def assess_image_quality(image: Image.Image):

    """
    Basic image quality checks.

    These checks provide suggestions only.
    They do not guarantee classification accuracy.
    """

    small = image.convert("L")

    small.thumbnail((256, 256))

    stats = ImageStat.Stat(small)

    brightness = float(stats.mean[0])

    contrast = float(stats.stddev[0])

    edges = small.filter(
        ImageFilter.FIND_EDGES
    )

    edge_strength = float(
        ImageStat.Stat(edges).stddev[0]
    )

    warnings = []

    if min(image.size) < 224:

        warnings.append(
            "The image is small. A higher-resolution photograph may improve identification."
        )

    if brightness < 45:

        warnings.append(
            "The photograph looks quite dark. Try brighter natural lighting."
        )

    elif brightness > 215:

        warnings.append(
            "The photograph may be overexposed."
        )

    if contrast < 22:

        warnings.append(
            "The photograph has low contrast. Try a clearer image."
        )

    if edge_strength < 9:

        warnings.append(
            "The photograph may be soft or out of focus."
        )

    return {

        "status": (
            "attention" if warnings else "ok"
        ),

        "warnings": warnings,

        "width": image.width,

        "height": image.height,

        "brightness_mean": round(
            brightness, 1
        ),

        "contrast_stddev": round(
            contrast, 1
        ),

        "edge_strength_heuristic": round(
            edge_strength, 1
        ),

        "note": (
            "These are basic image quality heuristics, "
            "not a guarantee of identification accuracy."
        )
    }

# ============================================================
# BACKGROUND RESEARCH JOBS
# ============================================================

research_jobs: dict[str, dict[str, Any]] = {}

research_tasks: set[asyncio.Task] = set()

RESEARCH_JOB_TTL = 30 * 60


def cleanup_research_jobs():
    """Remove old completed research jobs."""

    now = time.time()

    expired = [
        job_id
        for job_id, job in research_jobs.items()
        if (
            job.get("status") in {"completed", "failed"}
            and now - job.get("created_at", now) > RESEARCH_JOB_TTL
        )
    ]

    for job_id in expired:
        research_jobs.pop(job_id, None)


async def run_research_job(
    research_id: str,
    common_name: str,
    scientific_name: str,
    ebird_code: str | None
):
    """Research a species without delaying the prediction response."""

    job = research_jobs.get(research_id)

    if job is None:
        return

    job["status"] = "running"
    job["updated_at"] = time.time()

    try:
        result = await research_bird(
            common_name=common_name,
            scientific_name=scientific_name,
            ebird_code=ebird_code
        )

        job["result"] = result
        job["status"] = "completed"

    except Exception:
        logger.exception(
            "Background research failed for %s",
            scientific_name
        )

        job["status"] = "failed"
        job["result"] = {
            "available": False,
            "research_status": "error",
            "message": (
                "Species research could not be completed. "
                "The bird identification is still available."
            ),
            "sources": []
        }

    finally:
        job["updated_at"] = time.time()

# ============================================================
# BIRD IDENTIFICATION
# ============================================================

def identify_species(image: Image.Image):

    """
    Extract image embedding and compare it against the
    WingCLIP species embedding catalogue.

    Returns one highest-ranked species.
    """

    if (
        model is None
        or preprocessor is None
        or classifier_embeddings is None
        or not species_labels
    ):

        raise RuntimeError(
            "WingCLIP model is not loaded."
        )

    # Apply the official WingCLIP image transformation.

    image_tensor = preprocessor(
        image
    ).unsqueeze(0).to(DEVICE)

    # Run inference.

    with inference_lock, torch.inference_mode():

        image_embedding = model(
            image_tensor
        )

        similarities = (
            image_embedding
            @ classifier_embeddings.T
        )[0]

        best_index = int(
            torch.argmax(similarities).item()
        )

        best_score = float(
            similarities[best_index].item()
        )

    common_name, scientific_name, ebird_code = (
        species_labels[best_index]
    )

    return {

        "common_name": common_name,

        "scientific_name": scientific_name,

        "ebird_code": ebird_code or None,

        "similarity_score": round(
            best_score, 6
        )
    }


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(application: FastAPI):

    global model
    global preprocessor
    global classifier_embeddings
    global species_labels
    global species_database

    # Load the model once during application startup.

    (
        model,
        preprocessor,
        classifier_embeddings,
        species_labels
    ) = load_wingclip()

    # Load the local bird information database.

    species_database = load_species_database()

    logger.info(
        "Loaded %d local bird profiles.",
        len(species_database)
    )

    try:

        yield

    finally:

        logger.info(
            "Releasing model resources."
        )

        model = None

        preprocessor = None

        classifier_embeddings = None

        species_labels = []

        species_database = {}

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Indian Bird Identification API",

    description=(
        "Identify a bird from an uploaded photograph "
        "and retrieve basic species information."
    ),

    version="3.0.0",

    lifespan=lifespan
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://127.0.0.1:5500",

        "http://localhost:5500",

        "http://127.0.0.1:5501",

        "http://localhost:5501"

    ],

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],

    allow_headers=["*"]

)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "message": "Indian Bird Identification API",

        "docs": "/docs",

        "health": "/health"

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    loaded = (

        model is not None

        and classifier_embeddings is not None

        and bool(species_labels)

    )

    return {

        "status": (
            "healthy" if loaded else "not_ready"
        ),

        "gemini_configured": bool(GEMINI_API_KEY),
        
        "gemini_model": GEMINI_MODEL,

        "google_search_grounding_enabled": GEMINI_ENABLE_SEARCH,

        "model_loaded": loaded,

        "model": MODEL_REPO,

        "device": str(DEVICE),

        "species_in_catalogue": (
            len(species_labels) if loaded else 0
        ),

        "local_species_profiles": len(
            species_database
        ),

        "score_type": (
            "cosine_similarity_not_probability"
        )

    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict_bird(
    file: UploadFile = File(...)
):

    if (
        model is None
        or preprocessor is None
        or classifier_embeddings is None
    ):
        raise HTTPException(
            status_code=503,
            detail="The bird identification model is still loading."
        )

    # Read uploaded image.
    try:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty."
            )

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB."
            )

        image = decode_image(image_bytes)

    finally:
        await file.close()

    try:
        quality = assess_image_quality(image)

        # Stage 1: WingCLIP identification.
        identification = await run_in_threadpool(
            identify_species,
            image
        )

        # Stage 2: Get local/offline profile immediately.
        information = build_bird_information(
            identification["common_name"],
            identification["scientific_name"],
            identification["ebird_code"],
            species_database
        )

        local_profile_available = bool(
            information.get("available")
        )

        # Stage 3: Schedule Gemini research in the background.
        cleanup_research_jobs()

        research_id = uuid.uuid4().hex

        research_jobs[research_id] = {
            "status": "pending",
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None
        }

        task = asyncio.create_task(
            run_research_job(
                research_id,
                identification["common_name"],
                identification["scientific_name"],
                identification.get("ebird_code")
            )
        )

        research_tasks.add(task)
        task.add_done_callback(research_tasks.discard)

        # Immediate response. Do not await Gemini here.
        information.update({
            "research_available": False,
            "research_status": "pending",
            "research_message": (
                "Bird identified successfully. "
                "Gemini is preparing the species profile."
            ),
            "research_id": research_id,
            "research_model": GEMINI_MODEL,
            "search_grounding_enabled": GEMINI_ENABLE_SEARCH,
            "local_profile_available": local_profile_available,
            "sources": information.get("sources", [])
        })

        return {
            "filename": file.filename or "uploaded-image",
            "model": MODEL_REPO,
            "classification_mode": "whole_image_single_best_match",
            "identification": identification,
            "bird_information": information,
            "image_quality": quality,
            "research_id": research_id,
            "score_type": "cosine_similarity_not_probability",
            "important_note": (
                "This is the model's closest catalogue match, "
                "not a confirmed identification. Indian field "
                "accuracy has not yet been independently measured."
            )
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Identification failed: %s", exc)

        raise HTTPException(
            status_code=500,
            detail=(
                "Bird identification failed. "
                "Check the backend terminal for details."
            )
        ) from exc

@app.get("/research/{research_id}")
def get_research_status(research_id: str):

    job = research_jobs.get(research_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Research job not found or expired."
        )

    return {
        "research_id": research_id,
        "status": job["status"],
        "result": job.get("result")
    }