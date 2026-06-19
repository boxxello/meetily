from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .torch_compat import ensure_torch_amp_compatibility
from .config import load_config
from .diarization import DiarizationService, SpeakerTurn
from .embeddings import EmbeddingService, ProfileEmbedding, group_turns_by_cluster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ensure_torch_amp_compatibility()

config = load_config()
diarization_service = DiarizationService(
    pipeline_name=config.diarization_pipeline,
    hf_token=config.hf_token,
    device=config.device,
)
embedding_service = EmbeddingService(
    model_name=config.embedding_model,
    device=config.device,
)

app = FastAPI(
    title="Meetily Speaker Recognition Service",
    version="0.1.0",
)


class TurnPayload(BaseModel):
    cluster_label: str
    start: float
    end: float


class DiarizeRequest(BaseModel):
    audio_path: str
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class DiarizeResponse(BaseModel):
    model: str
    turns: list[TurnPayload]


class EmbedClusterRequest(BaseModel):
    audio_path: str
    turns: list[TurnPayload]
    max_turns: int = 8
    min_turn_duration: float = 1.5


class EmbedClusterResponse(BaseModel):
    embedding_model: str
    embedding_dim: int
    embedding: list[float]
    sample_count: int
    total_duration: float


class ProfilePayload(BaseModel):
    profile_id: str
    display_name: str
    embedding_model: str
    embedding: list[float] = Field(default_factory=list)


class IdentifyRequest(BaseModel):
    audio_path: str
    turns: list[TurnPayload]
    profiles: list[ProfilePayload] = Field(default_factory=list)
    threshold: float | None = None
    ambiguity_margin: float | None = None


class ClusterMatchPayload(BaseModel):
    cluster_label: str
    profile_id: str | None
    display_name: str | None
    confidence: float | None
    ambiguous: bool
    sample_count: int
    total_duration: float


class IdentifyResponse(BaseModel):
    clusters: list[ClusterMatchPayload]


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "diarization_model": config.diarization_pipeline,
        "embedding_model": config.embedding_model,
        "device": config.device,
        "diarization_loaded": diarization_service.is_loaded,
        "embedding_loaded": embedding_service.is_loaded,
        "hf_token_configured": bool(config.hf_token),
    }


@app.post("/diarize", response_model=DiarizeResponse)
def diarize(request: DiarizeRequest) -> DiarizeResponse:
    try:
        turns = diarization_service.diarize(
            audio_path=request.audio_path,
            num_speakers=request.num_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )
        return DiarizeResponse(
            model=config.diarization_pipeline,
            turns=[TurnPayload(**asdict(turn)) for turn in turns],
        )
    except Exception as exc:
        logger.exception("Diarization failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/embed-cluster", response_model=EmbedClusterResponse)
def embed_cluster(request: EmbedClusterRequest) -> EmbedClusterResponse:
    try:
        turns = [_to_turn(turn) for turn in request.turns]
        embedding = embedding_service.embed_cluster(
            audio_path=request.audio_path,
            turns=turns,
            max_turns=request.max_turns,
            min_turn_duration=request.min_turn_duration,
        )
        return EmbedClusterResponse(
            embedding_model=embedding.embedding_model,
            embedding_dim=len(embedding.embedding),
            embedding=embedding.embedding,
            sample_count=embedding.sample_count,
            total_duration=embedding.total_duration,
        )
    except Exception as exc:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/identify", response_model=IdentifyResponse)
def identify(request: IdentifyRequest) -> IdentifyResponse:
    try:
        turns = [_to_turn(turn) for turn in request.turns]
        profiles = [
            ProfileEmbedding(
                profile_id=profile.profile_id,
                display_name=profile.display_name,
                embedding_model=profile.embedding_model,
                embedding=profile.embedding,
            )
            for profile in request.profiles
        ]
        threshold = request.threshold if request.threshold is not None else config.recognition_threshold
        ambiguity_margin = (
            request.ambiguity_margin
            if request.ambiguity_margin is not None
            else config.ambiguity_margin
        )

        matches = []
        for cluster_label, cluster_turns in group_turns_by_cluster(turns).items():
            match = embedding_service.identify_cluster(
                audio_path=request.audio_path,
                cluster_label=cluster_label,
                turns=cluster_turns,
                profiles=profiles,
                threshold=threshold,
                ambiguity_margin=ambiguity_margin,
            )
            matches.append(ClusterMatchPayload(**asdict(match)))

        matches.sort(key=lambda match: match.cluster_label)
        return IdentifyResponse(clusters=matches)
    except Exception as exc:
        logger.exception("Identification failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _to_turn(payload: TurnPayload) -> SpeakerTurn:
    return SpeakerTurn(
        cluster_label=payload.cluster_label,
        start=payload.start,
        end=payload.end,
    )
