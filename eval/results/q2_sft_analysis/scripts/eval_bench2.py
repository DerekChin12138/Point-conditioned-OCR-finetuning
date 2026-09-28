import json, sys, time, gc
from pathlib import Path
sys.path.insert(0, 'src'); sys.path.insert(0, 'train')
import torch
from PIL import Image
from point_ocr.image_resize import resize_for_ovis
from point_ocr.infer import (
    generate_point_text, apply_point_chat_template, point_eos_token_ids,
    strip_format_leak, POINT_STOP_STRINGS, unwrap_text_tokenizer, resolve_point_prompt,
)

VAL = Path('data/splits_stage_q2_ocr_mt/val.jsonl')
rows = [json.loads(l) for l in open(VAL, encoding='utf-8')][:24]
imgs, prompts = [], []
for r in rows:
    im = Image.open(r['images'][0]).convert('RGB')
    im = resize_for_ovis(im, min_pixels=448*448, max_pixels=2880*2880)
    p = r['messages'][0]['content']
    if p.startswith('<image>'): p = p[len('<image>'):]
    imgs.append(im); prompts.append(p)

from unsloth import FastVisionModel
from peft import PeftModel
model, tok = FastVisionModel.from_pretrained('checkpoints/q1_grpo_merged', load_in_4bit=True, use_gradient_checkpointing='unsloth')
model = PeftModel.from_pretrained(model, 'checkpoints/q2_ocr_mt/adapter_final')
FastVisionModel.for_inference(model)
inner = getattr(tok, 'tokenizer', tok)


def texts():
    out = []
    for im, p in zip(imgs, prompts):
        w, h = im.size
        t = resolve_point_prompt(image_w=w, image_h=h, prompt=p)
        out.append(apply_point_chat_template(tok, [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}]))
    return out


def run(bs, mmax):
    inner.padding_side = 'left'
    tx = texts()
    order = sorted(range(len(imgs)), key=lambda i: imgs[i].size[0] * imgs[i].size[1])
    res = [None] * len(imgs)
    eos = point_eos_token_ids(tok); tt = unwrap_text_tokenizer(tok)
    torch.cuda.reset_peak_memory_stats()
    for s in range(0, len(order), bs):
        idx = order[s:s+bs]
        inputs = tok(images=[imgs[i] for i in idx], text=[tx[i] for i in idx], padding=True, return_tensors='pt')
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
        plen = inputs['input_ids'].shape[1]
        kw = dict(max_new_tokens=mmax, use_cache=True, do_sample=False)
        if eos: kw['eos_token_id'] = eos if len(eos) > 1 else eos[0]
        out = model.generate(**inputs, stop_strings=list(POINT_STOP_STRINGS), tokenizer=tt, **kw)
        for j, i in enumerate(idx):
            res[i] = strip_format_leak(tok.decode(out[j][plen:], skip_special_tokens=True).strip()).cleaned
    return res, torch.cuda.max_memory_allocated() / 1e9


run(2, 32)
t0 = time.time(); b1, _ = run(1, 512); t1 = time.time() - t0
print(f'batch1 {t1:5.1f}s {len(imgs)/t1:.2f} samp/s')
for bs in (8, 12, 16):
    t0 = time.time(); out, mem = run(bs, 512); dt = time.time() - t0
    same = sum(1 for a, b in zip(b1, out) if a == b)
    print(f'batch{bs:<2d} {dt:5.1f}s {len(imgs)/dt:.2f} samp/s peak_alloc={mem:.2f}GB identical_to_batch1={same}/{len(imgs)}')
