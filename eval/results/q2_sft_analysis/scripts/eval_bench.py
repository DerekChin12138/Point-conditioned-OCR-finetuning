import json, sys, time
from pathlib import Path
sys.path.insert(0, 'src'); sys.path.insert(0, 'train')
from PIL import Image
from point_ocr.image_resize import resize_for_ovis
from point_ocr.infer import (
    generate_point_text, apply_point_chat_template, point_eos_token_ids,
    strip_format_leak, POINT_STOP_STRINGS, unwrap_text_tokenizer, resolve_point_prompt,
)

VAL = Path('data/splits_stage_q2_ocr_mt/val.jsonl')
rows = [json.loads(l) for l in open(VAL, encoding='utf-8')][:16]
imgs, prompts, gts = [], [], []
for r in rows:
    im = Image.open(r['images'][0]).convert('RGB')
    im = resize_for_ovis(im, min_pixels=448 * 448, max_pixels=2880 * 2880)
    p = r['messages'][0]['content']
    if p.startswith('<image>'):
        p = p[len('<image>'):]
    imgs.append(im); prompts.append(p); gts.append(r['messages'][-1]['content'])

from unsloth import FastVisionModel
from peft import PeftModel
t0 = time.time()
model, tok = FastVisionModel.from_pretrained('checkpoints/q1_grpo_merged', load_in_4bit=True, use_gradient_checkpointing='unsloth')
model = PeftModel.from_pretrained(model, 'checkpoints/q2_ocr_mt/adapter_final')
FastVisionModel.for_inference(model)
print('load', round(time.time() - t0, 1), 's; processor', type(tok).__name__)
inner = getattr(tok, 'tokenizer', tok)
print('  inner', type(inner).__name__, 'pad', inner.pad_token, inner.pad_token_id, 'padding_side', inner.padding_side)


def run_batch1():
    out = []
    for im, p in zip(imgs, prompts):
        r = generate_point_text(model, tok, im, p, max_new_tokens=512, clean=True)
        out.append(r.cleaned)
    return out


def build_texts():
    texts = []
    for im, p in zip(imgs, prompts):
        w, h = im.size
        text = resolve_point_prompt(image_w=w, image_h=h, prompt=p)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        texts.append(apply_point_chat_template(tok, msgs))
    return texts


def run_batched(bs):
    inner.padding_side = 'left'
    texts = build_texts()
    order = sorted(range(len(imgs)), key=lambda i: imgs[i].size[0] * imgs[i].size[1])
    results = [None] * len(imgs)
    eos = point_eos_token_ids(tok)
    text_tok = unwrap_text_tokenizer(tok)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch_imgs = [imgs[i] for i in idx]
        batch_texts = [texts[i] for i in idx]
        inputs = tok(images=batch_imgs, text=batch_texts, padding=True, return_tensors='pt')
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
        plen = inputs['input_ids'].shape[1]
        kw = dict(max_new_tokens=512, use_cache=True, do_sample=False)
        if eos:
            kw['eos_token_id'] = eos if len(eos) > 1 else eos[0]
        out = model.generate(**inputs, stop_strings=list(POINT_STOP_STRINGS), tokenizer=text_tok, **kw)
        for j, i in enumerate(idx):
            gen = out[j][plen:]
            raw = tok.decode(gen, skip_special_tokens=True).strip()
            results[i] = strip_format_leak(raw).cleaned
    return results


generate_point_text(model, tok, imgs[0], prompts[0], max_new_tokens=32, clean=True)
run_batched(2)

for name, fn in [('batch1', run_batch1), ('batch2', lambda: run_batched(2)),
                 ('batch4', lambda: run_batched(4)), ('batch8', lambda: run_batched(8))]:
    t0 = time.time(); out = fn(); dt = time.time() - t0
    print(f'{name:8s} {dt:6.1f}s  {len(imgs)/dt:5.2f} samp/s  empty={sum(1 for o in out if not o.strip())}/{len(out)}')
