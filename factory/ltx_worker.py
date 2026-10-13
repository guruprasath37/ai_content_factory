"""LTX-Video 2B worker (runs as its own process; see aivideo.py).

Fits an 8 GB GPU by (1) encoding all prompts with the T5-XXL text encoder on the CPU first, then freeing it,
and (2) running only the 2B transformer + VAE on the GPU with CPU offload and tiled VAE decoding.

Usage: python -m factory.ltx_worker spec.json   (or python factory/ltx_worker.py spec.json)
spec: {"model_dir", "width", "height", "frames", "steps", "guidance", "seed", "negative",
       "jobs": [{"prompt", "image" (optional path), "out"}]}
Prints one JSON line per event: {"event": "...", ...}
"""
import gc, json, sys, time
from pathlib import Path

import torch


def emit(**kw): print(json.dumps(kw), flush=True)


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    model, jobs = spec["model_dir"], spec["jobs"]
    W, H, F = spec["width"], spec["height"], spec["frames"]
    t0 = time.time()

    # 1) prompts -> embeddings on CPU
    from transformers import T5EncoderModel, T5TokenizerFast
    tok = T5TokenizerFast.from_pretrained(model, subfolder="tokenizer")
    enc = T5EncoderModel.from_pretrained(model, subfolder="text_encoder", torch_dtype=torch.bfloat16).eval()
    emit(event="text_encoder_loaded", sec=round(time.time() - t0))
    texts = [j["prompt"] for j in jobs] + [spec["negative"]]
    ids = tok(texts, padding="max_length", max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        emb = enc(ids.input_ids)[0]
    mask = ids.attention_mask
    del enc; gc.collect()
    emit(event="prompts_encoded", sec=round(time.time() - t0))
    neg_e, neg_m = emb[-1:], mask[-1:]

    # 2) video model on GPU
    from diffusers import LTXImageToVideoPipeline, LTXPipeline
    from diffusers.utils import export_to_video, load_image
    kw = dict(text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16)
    i2v = LTXImageToVideoPipeline.from_pretrained(model, **kw)
    i2v.enable_model_cpu_offload(); i2v.vae.enable_tiling()
    t2v = LTXPipeline(**{k: v for k, v in i2v.components.items() if k in ("transformer", "vae", "scheduler")}, text_encoder=None, tokenizer=None)
    t2v.enable_model_cpu_offload()
    emit(event="model_loaded", sec=round(time.time() - t0))

    for i, job in enumerate(jobs):
        t1 = time.time()
        try:
            dev = "cuda"   # embeddings were computed on the CPU; the offloaded transformer expects them on the GPU
            common = dict(prompt_embeds=emb[i:i + 1].to(dev, torch.bfloat16), prompt_attention_mask=mask[i:i + 1].to(dev),
                          negative_prompt_embeds=neg_e.to(dev, torch.bfloat16), negative_prompt_attention_mask=neg_m.to(dev),
                          width=W, height=H, num_frames=F, num_inference_steps=spec["steps"], guidance_scale=spec["guidance"],
                          generator=torch.Generator("cpu").manual_seed(spec["seed"] + i), decode_timestep=0.03, decode_noise_scale=0.025)
            if job.get("image"):
                frames = i2v(image=load_image(job["image"]), **common).frames[0]
            else:
                frames = t2v(**common).frames[0]
            export_to_video(frames, job["out"], fps=24)
            emit(event="clip_done", index=i, out=job["out"], sec=round(time.time() - t1), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        except Exception as e:      # one bad clip must not lose the others
            emit(event="clip_failed", index=i, error=f"{type(e).__name__}: {str(e)[:300]}")
        torch.cuda.empty_cache(); gc.collect()
    emit(event="finished", sec=round(time.time() - t0))


if __name__ == "__main__":
    main()
