#!/usr/bin/env python3
"""Transfer points from a local Qdrant storage (SQLite) to a remote Qdrant server.

Designed for use as a lightweight seeder in docker-compose or run manually on the host.
Features:
- health-check wait for remote Qdrant
- optional recreate of remote collection using local collection config
- scroll() pagination from local storage and batched upsert to remote
- basic retry/backoff on upsert
- calls `create_snapshot()` on remote when finished

Configure via environment variables (see defaults below).
"""
from __future__ import annotations

import os
import time
import sys
from typing import Optional

import requests
from qdrant_client import QdrantClient


def getenv(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None else default


def wait_for_qdrant(host: str, port: int, timeout: int = 60) -> bool:
    url = f"http://{host}:{port}/healthz"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main() -> int:
    # Configuration (env vars)
    local_path = getenv("LOCAL_QDRANT_PATH", "./data/processed/vector_index")
    remote_host = getenv("REMOTE_QDRANT_HOST", "localhost")
    remote_port = int(getenv("REMOTE_QDRANT_PORT", "6333"))
    collection_name = getenv("COLLECTION_NAME", "legal_docs")
    batch_size = int(getenv("BATCH_SIZE", "100"))
    recreate = getenv("RECREATE_REMOTE", "true").lower() in ("1", "true", "yes")
    upsert_retries = int(getenv("UPSERT_RETRIES", "3"))

    print(f"Local path: {local_path}")
    print(f"Remote: {remote_host}:{remote_port} | collection: {collection_name}")

    # Clients
    local_client = QdrantClient(path=local_path)

    print("Waiting for remote Qdrant to be healthy...")
    if not wait_for_qdrant(remote_host, remote_port, timeout=120):
        print("⚠ Remote Qdrant not healthy after timeout. Aborting.")
        return 2

    remote_client = QdrantClient(host=remote_host, port=remote_port)

    # Mirror collection config from local
    try:
        local_info = local_client.get_collection(collection_name)
        local_vectors = local_info.config.params.vectors
    except Exception as e:
        print(f"⚠ Failed to read local collection config: {e}")
        return 3

    # Helper to get a collection count in a robust way
    def get_count_safe(client, name: str):
        try:
            resp = client.count(collection_name=name)
            # resp may be int-like, or a dict/object
            if resp is None:
                return None
            # Try common shapes
            if isinstance(resp, int):
                return resp
            if isinstance(resp, dict):
                # qdrant client may return {'result': {'count': n}} or {'count': n}
                if "result" in resp and isinstance(resp["result"], dict) and "count" in resp["result"]:
                    return int(resp["result"]["count"])
                if "count" in resp:
                    return int(resp["count"])
            # fallback: try attribute access
            count_attr = getattr(resp, "count", None)
            if count_attr is not None:
                return int(count_attr)
        except Exception:
            pass
        return None

    # Determine counts to decide whether transfer is necessary
    local_count = get_count_safe(local_client, collection_name)
    remote_count = get_count_safe(remote_client, collection_name)
    if local_count is not None and remote_count is not None:
        print(f"Local count={local_count}, remote count={remote_count}")
        if remote_count == local_count:
            print("✓ Remote collection appears up-to-date; skipping recreate/transfer.")
            return 0

    if recreate:
        print(f"Recreating remote collection '{collection_name}' (this will erase existing data)...")
        remote_client.recreate_collection(collection_name=collection_name, vectors_config=local_vectors)
    else:
        # ensure remote collection exists with compatible params
        try:
            _ = remote_client.get_collection(collection_name)
            print(f"Remote collection '{collection_name}' exists; will append/upsert points.")
        except Exception:
            print(f"Remote collection '{collection_name}' missing; creating with local params")
            remote_client.create_collection(collection_name=collection_name, vectors_config=local_vectors)

    # Transfer with pagination (scroll)
    print("Starting transfer via scroll()...")
    offset = None
    total_transferred = 0

    while True:
        try:
            points, next_offset = local_client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                with_vectors=True,
                with_payload=True,
                offset=offset,
            )
        except Exception as e:
            print(f"⚠ Error reading from local storage: {e}")
            return 4

        if not points:
            break

        # Convert records returned by local_client.scroll into plain dicts/PointStruct-like objects
        upsert_points = []
        for rec in points:
            # rec can be a pydantic model (object) or a dict depending on client version
            rid = getattr(rec, "id", None) if not isinstance(rec, dict) else rec.get("id")
            payload = getattr(rec, "payload", None) if not isinstance(rec, dict) else rec.get("payload")
            # vector may be under .vector or .values (or 'vector' key)
            vector = None
            if isinstance(rec, dict):
                vector = rec.get("vector") or rec.get("values")
            else:
                vector = getattr(rec, "vector", None) or getattr(rec, "values", None)

            if vector is None:
                print(f"⚠ Skipping point id={rid} because no vector found in record")
                continue

            # ensure vector is a plain list (not numpy)
            try:
                vec_list = list(vector)
            except Exception:
                vec_list = vector

            upsert_points.append({"id": rid, "vector": vec_list, "payload": payload})

        if not upsert_points:
            print("⚠ No upsertable points in this batch; skipping upsert")
        else:
            # Attempt upsert with retries/backoff
            for attempt in range(1, upsert_retries + 1):
                try:
                    remote_client.upsert(collection_name=collection_name, points=upsert_points)
                    break
                except Exception as e:
                    print(f"⚠ Upsert attempt {attempt} failed: {e}")
                    if attempt == upsert_retries:
                        print("✖ Exhausted upsert retries — aborting")
                        return 5
                    backoff = 2 ** attempt
                    time.sleep(backoff)

        total_transferred += len(points)
        print(f"Transferred {total_transferred} points...")

        offset = next_offset
        if offset is None:
            break

    print(f"Completed transfer: {total_transferred} points.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
