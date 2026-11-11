from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["transcribe", "subtitles", "burn", "dubbing", "all"] | None = None
    dubbing: bool | None = None
    target_language: str | None = Field(default=None, min_length=1)
    source_language: str | None = Field(default=None, min_length=1)
    glossary_domain: str | None = None


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    subtitles: str | None = None
    title: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    glossary_domain: str | None = None
    stage: Literal["transcribe", "subtitles", "dubbing", "all"] = "subtitles"
    dubbing: bool = False
    autorun: bool = False


class InvalidateStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str = Field(min_length=1)
    autorun: bool = False
    run_stage: Literal["transcribe", "subtitles", "dubbing", "all"] = "subtitles"


class TermsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    terms: list[dict[str, Any]] | None = None
    summary: str | None = None
    glossary_domain: str | None = None


class DomainTermSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["add", "remove"]
    terms: list[dict[str, Any]] | None = None
    srcs: list[str] | None = None


class CreateDomainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain: str | None = None
    name: str | None = None


class UtterancesUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    utterances: list[dict[str, Any]]


class ConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    updates: dict[str, Any]


class TaskLogEntryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    msg: str = Field(min_length=1)
    tag: Literal["asr", "llm", "warn"] = "llm"
    label: str = Field(default="RUN", min_length=1, max_length=16)
    warn: bool = False
    time: str | None = None


class PickFilesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[
        "media", "video", "subtitle", "cookies", "directory", "save_video"
    ] = "media"
    multi: bool = True
    default_name: str | None = None


class StandaloneBurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_path: str = Field(min_length=1)
    subtitle_path: str = Field(min_length=1)
    secondary_subtitle_path: str | None = None
    output_path: str | None = None
    ffmpeg_gpu: bool | None = None


class LLMProbeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None


class DetectMaiRegionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str = Field(min_length=1)
