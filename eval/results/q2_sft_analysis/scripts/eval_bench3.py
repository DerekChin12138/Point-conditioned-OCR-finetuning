import json, sys, time
from pathlib import Path
sys.path.insert(0, 'src'); sys.path.insert(0, 'train')
import torch
from PIL import Image
from point_ocr.image_resize import resize_for_ovis
from point_ocr.infer import (
    generate_point_text, apply_point_chat_template, point_eos_token_ids,
    strip_format_leak, POINT_STOP_STRINGS, unwrap_text_tokenizer, resolve_point_prompt,
)

MODE = sys.argv[1] if len(sys.argv) > 1 else 'bf16'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24

VAL = Path('data/splits_stage_q2_ocr_mt/val.jsonl')
allrows = [json.loads(l) for l in open(VAL, encoding='utf-8')]
pos = [r for r in allrows if not r['metadata'].get('is_negative')]
neg = [r for r in allrows if r['metadata'].get('is_negative')]
rows = (pos[:int(N * 0.7)] + neg[:N - int(N * 0.7)])
imgs, prompts, negs = [], [], []
for r in rows:
    im = Image.open(r['images'][0]).convert('RGB')
    im = resize_for_ovis(im, min_pixels=448*448, max_pixels=2880*2880)
    p = r['messages'][0]['content']
    if p.startswith('<image>'): p = p[len('<image>'):]
    imgs.append(im); prompts.append(p); negs.append(bool(r['metadata'].get('is_negative')))

from unsloth import FastVisionModel
from peft import PeftModel
kw = dict(load_in_4bit=True, use_gradient_checkpointing='unsloth') if MODE == '4bit' else dict(load_in_4bit=False, dtype=torch.bfloat16, use_gradient_checkpointing='unsloth')
model, tok = FastVisionModel.from_pretrained('checkpoints/q1_grpo_merged', **kw)
model = PeftModel.from_pretrained(model, 'checkpoints/q2_ocr_mt/adapter_final')
FastVisionModel.for_inference(model)
inner = getattr(tok, 'tokenizer', tok)
print(f'[{MODE}] loaded', flush=True)


def texts():
    out = []
    for im, p in zip(imgs, prompts):
        w, h = im.size
        t = resolve_point_prompt(image_w=w, image_h=h, prompt=p)
        out.append(apply_point_chat_template(tok, [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}]))
    return out


def run(bs, mmax=512):
    inner.padding_side = 'left'
    tx = texts()
    order = sorted(range(len(imgs)), key=lambda i: imgs[i].size[0] * imgs[i].size[1])
    res = [None] * len(imgs)
    eos = point_eos_token_ids(tok); tt = unwrap_text_tokenizer(tok)
    if bs == 1:
        for i in order:
            r = generate_point_text(model, tok, imgs[i], prompts[i], max_new_tokens=mmax, clean=True)
            res[i] = r.cleaned
        return res
    for s in range(0, len(order), bs):
        idx = order[s:s+bs]
        inputs = tok(images=[imgs[i] for i in idx], text=[tx[i] for i in idx], padding=True, return_tensors='pt')
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
        plen = inputs['input_ids'].shape[1]
        gkw = dict(max_new_tokens=mmax, use_cache=True, do_sample=False)
        if eos: gkw['eos_token_id'] = eos if len(eos) > 1 else eos[0]
        out = model.generate(**inputs, stop_strings=list(POINT_STOP_STRINGS), tokenizer=tt, **gkw)
        for j, i in enumerate(idx):
            res[i] = strip_format_leak(tok.decode(out[j][plen:], skip_special_tokens=True).strip()).cleaned
    return res


run(2, 32)
t0 = time.time(); b1 = run(1); t1 = time.time() - t0
print(f'[{MODE}] batch1 {t1:5.1f}s {len(imgs)/t1:.2f} samp/s', flush=True)
for bs in (8, 16):
    t0 = time.time(); out = run(bs); dt = time.time() - t0
    same = sum(1 for a, b in zip(b1, out) if a == b)
    eff1 = sum(1 for a, n in zip(b1, negs) if n and not a.strip()) / max(1, sum(negs))
    eff8 = sum(1 for a, n in zip(out, negs) if n and not a.strip()) / max(1, sum(negs))
    print(f'[{MODE}] batch{bs:<2d} {dt:5.1f}s {len(imgs)/dt:.2f} samp/s identical={same}/{len(imgs)} eff_empty b1={eff1:.2f} b{bs}={eff8:.2f}', flush=True)
