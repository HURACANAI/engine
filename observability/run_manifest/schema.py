"""
Run Manifest Schema

JSON schema for run manifest validation.
"""

# Schema version
MANIFEST_SCHEMA_VERSION = "1.0.0"

# JSON Schema for manifest validation
MANIFEST_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "RunManifest",
    "description": "Complete reproducibility manifest for a training run",
    "type": "object",
    "required": [
        "run_manifest_id",
        "run_date",
        "created_at",
        "git",
        "settings_hash",
        "feature_set_id",
        "dataset",
        "environment",
        "schema_version"
    ],
    "properties": {
        "run_manifest_id": {
            "type": "string",
            "description": "Unique manifest identifier",
            "pattern": "^run_\\d{8}_\\d{6}$"
        },
        "run_date": {
            "type": "string",
            "format": "date",
            "description": "Training run date (YYYY-MM-DD)"
        },
        "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "Manifest creation timestamp"
        },
        "git": {
            "type": "object",
            "required": ["commit", "branch", "dirty"],
            "properties": {
                "commit": {"type": "string", "minLength": 40, "maxLength": 40},
                "branch": {"type": "string"},
                "dirty": {"type": "boolean"},
                "remote_url": {"type": ["string", "null"]},
                "commit_message": {"type": ["string", "null"]},
                "commit_author": {"type": ["string", "null"]},
                "commit_date": {"type": ["string", "null"]}
            }
        },
        "settings_hash": {
            "type": "string",
            "description": "SHA256 hash of settings",
            "minLength": 64,
            "maxLength": 64
        },
        "settings_snapshot": {
            "type": "object",
            "description": "Full settings snapshot"
        },
        "feature_set_id": {
            "type": "string",
            "description": "Feature set identifier"
        },
        "dataset": {
            "type": "object",
            "required": ["symbols", "checksums"],
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                },
                "checksums": {
                    "type": "object",
                    "description": "SHA256 checksums per symbol"
                },
                "total_candles": {"type": ["integer", "null"]},
                "date_range": {
                    "type": ["object", "null"],
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"}
                    }
                }
            }
        },
        "environment": {
            "type": "object",
            "required": ["python_version", "platform", "package_versions"],
            "properties": {
                "python_version": {"type": "string"},
                "platform": {"type": "string"},
                "platform_version": {"type": "string"},
                "package_versions": {
                    "type": "object",
                    "description": "Package versions dict"
                },
                "ray_cluster_info": {
                    "type": ["object", "null"]
                }
            }
        },
        "results": {
            "type": ["object", "null"],
            "properties": {
                "models_trained": {"type": "integer", "minimum": 0},
                "models_published": {"type": "integer", "minimum": 0},
                "models_shadowed": {"type": "integer", "minimum": 0},
                "models_rejected": {"type": "integer", "minimum": 0},
                "total_runtime_seconds": {"type": ["number", "null"]},
                "errors_encountered": {"type": "integer", "minimum": 0}
            }
        },
        "schema_version": {
            "type": "string",
            "const": MANIFEST_SCHEMA_VERSION
        }
    }
}


def validate_manifest_json(manifest_dict: dict) -> tuple[bool, list[str]]:
    """
    Validate manifest against JSON schema

    Args:
        manifest_dict: Manifest as dictionary

    Returns:
        (is_valid, errors)
    """
    try:
        import jsonschema
        jsonschema.validate(manifest_dict, MANIFEST_JSON_SCHEMA)
        return True, []
    except ImportError:
        # jsonschema not installed, skip validation
        return True, ["jsonschema not installed - skipping validation"]
    except jsonschema.ValidationError as e:
        return False, [str(e)]
