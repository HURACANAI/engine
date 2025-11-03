"""Artifact management service for S3 uploads and checksum validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Dict

import boto3

from ..config.settings import S3Settings


@dataclass
class ArtifactBundle:
    files: Dict[str, bytes]


class ArtifactPublisher:
    """Writes models, configs, metrics, and reports to object storage."""

    def __init__(self, settings: S3Settings) -> None:
        if not settings.bucket:
            raise ValueError("S3 bucket must be configured")
        self._settings = settings
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key,
        )

    def publish(self, run_date: date, symbol: str, bundle: ArtifactBundle) -> str:
        prefix = self._prefix_for(run_date, symbol)
        checksum_lines = []
        for name, payload in bundle.files.items():
            key = f"{prefix}{name}"
            self._client.put_object(Bucket=self._settings.bucket, Key=key, Body=payload)
            checksum = hashlib.sha256(payload).hexdigest()
            checksum_lines.append(f"{checksum}  {name}")
        manifest = "\n".join(checksum_lines).encode("utf-8")
        self._client.put_object(
            Bucket=self._settings.bucket,
            Key=f"{prefix}checksums/sha256.txt",
            Body=manifest,
        )
        return f"s3://{self._settings.bucket}/{prefix}"

    def _prefix_for(self, run_date: date, symbol: str) -> str:
        safe_symbol = symbol.replace("/", "-")
        return (
            f"{self._settings.prefix}/{run_date:%Y/%m/%d}/"
            f"baseline_{safe_symbol}_{run_date:%Y%m%d}/"
        )
