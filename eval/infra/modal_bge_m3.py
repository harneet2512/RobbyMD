"""Modal deployment of BAAI/bge-m3 as a batched embedding endpoint.

Mirrors `eval/infra/modal_qwen.py` — same profile (`glitch112213`), same
L4-GPU convention, same persistent HF cache volume. bge-m3 is ~2 GB on an
L4's 24 GB so we have generous headroom; batch size 32 is comfortable.

Why a thin FastAPI wrapper around `sentence-transformers` instead of
`text-embeddings-inference` (TEI)? TEI is Apache-2.0 and would work, but
it needs its own binary image and shipping a second Docker pattern adds
complexity we don't need for hackathon-tier retrieval. `sentence_transformers`
already exposes `normalize_embeddings=True` server-side with one line, and
the payload shape is trivial. The choice is documented here so anyone
reading later sees the tradeoff.

Deploy (operator step — do not run from an agent worktree)
----------------------------------------------------------
    modal deploy --profile glitch112213 eval/infra/modal_bge_m3.py

Expected URL after deploy
-------------------------
    https://<workspace>--bge-m3-embeddings-serve.modal.run

Set that URL as `MODAL_BGE_M3_URL` in the smoke harness env:

    export MODAL_BGE_M3_URL=https://<workspace>--bge-m3-embeddings-serve.modal.run

Endpoint
--------
    POST /embed
    Body: {"texts": ["...", "..."]}
    Response: {
        "embeddings": [[...1024 floats...], ...],
        "model_version": "BAAI/bge-m3@<revision>",
        "count": <int>
    }

All vectors are L2-normalised server-side, so the caller can treat cosine
similarity as a dot product. 60-second client-side timeout on the caller
(`src/substrate/retrieval.py::EmbeddingClient._embed_modal`).

GPU / cold-start notes
----------------------
L4 (24 GB) is sufficient and cheapest; A10G is a drop-in fallback. First
invocation downloads ~2 GB of weights from HF (30–60 s); subsequent cold
starts hit the persisted volume and are 10–20 s.
"""
from __future__ import annotations

import modal

MINUTES = 60

# bge-m3 at the `main` HF branch. Pin by commit-sha later if we need strict
# reproducibility. Mirrors the version tag in `src/substrate/retrieval.py`
# (`BGE_M3_VERSION_TAG = "BAAI/bge-m3@main"`).
MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "main"
MODEL_VERSION = f"{MODEL_ID}@{MODEL_REVISION}"

app = modal.App("bge-m3-embeddings")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers==3.3.1",
        "fastapi==0.115.0",
        "huggingface-hub[hf-transfer]==0.27.1",
        "hf-transfer==0.1.9",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    scaledown_window=15 * MINUTES,
    timeout=30 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def serve():
    """FastAPI app exposing POST /embed."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from sentence_transformers import SentenceTransformer

    # Load once per container. Subsequent requests reuse the model.
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)

    class EmbedRequest(BaseModel):
        texts: list[str]

    class EmbedResponse(BaseModel):
        embeddings: list[list[float]]
        model_version: str
        count: int

    api = FastAPI()

    @api.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest) -> EmbedResponse:
        if not req.texts:
            return EmbedResponse(embeddings=[], model_version=MODEL_VERSION, count=0)
        if len(req.texts) > 512:
            raise HTTPException(
                status_code=413,
                detail=f"batch too large: {len(req.texts)} texts (max 512)",
            )
        # normalize_embeddings=True → unit-length cosine-ready vectors.
        # batch_size=32 fits comfortably in L4 VRAM with bge-m3.
        vecs = model.encode(
            req.texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return EmbedResponse(
            embeddings=[[float(x) for x in v] for v in vecs],
            model_version=MODEL_VERSION,
            count=len(req.texts),
        )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": MODEL_VERSION}

    return api
