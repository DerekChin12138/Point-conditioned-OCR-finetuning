import json, sys, re, time
from pathlib import Path
sys.path.insert(0,'src')
from PIL import Image
from point_ocr.marker import spec_for_image, draw_crosshair
from point_ocr.image_resize import resize_for_ovis
from point_ocr.infer import generate_point_text
from point_ocr.prompts import POINT_PROMPT_OCR_MT_V1

def load_neg(path, region='special_table'):
    out=[]
    for l in open(path,encoding='utf-8'):
        r=json.loads(l); m=r['metadata']
        if m.get('is_negative') and str(m.get('region'))==region:
            out.append(m)
    return out

q2=load_neg('data/splits_stage_q2_ocr_mt/val.jsonl')
q1=load_neg('data/splits_stage_q1_withreal/val.jsonl')
# pick distinct pages
def pick(rows,n):
    seen=set(); out=[]
    for m in rows:
        pid=m['page_id']
        if pid in seen: continue
        seen.add(pid); out.append(m)
        if len(out)>=n: break
    return out
cases=[]
for m in pick(q2,4):
    for frac in (0.005,0.00425,0.0025):
        cases.append(('Q2',m,frac,'data/pools_q2'))
for m in pick(q1,3):
    cases.append(('Q1',m,0.005,'data/pools_q'))

from unsloth import FastVisionModel
print('loading base',flush=True)
model,tok=FastVisionModel.from_pretrained('checkpoints/q1_grpo_merged', load_in_4bit=True, use_gradient_checkpointing='unsloth')
FastVisionModel.for_inference(model)
res=[]
for tag,m,frac,root in cases:
    pid=m['page_id']
    pool=m['pool_id']
    render=Path(root)/pool/'renders'/f'{pid}.png'
    if not render.is_file():
        print('MISSING',render); continue
    im=Image.open(render).convert('RGB')
    W,H=im.size
    spec=spec_for_image(W,H,area_frac=frac)
    im=draw_crosshair(im,m['point'][0],m['point'][1],spec)
    img=resize_for_ovis(im,min_pixels=448*448,max_pixels=2880*2880)
    out=generate_point_text(model,tok,img,POINT_PROMPT_OCR_MT_V1,max_new_tokens=256,clean=True)
    p=out.cleaned.strip()
    kind='empty' if not p else 'LEAK'
    print(f'{tag} {pid} frac={frac} -> {kind}: {p[:90]!r}',flush=True)
    res.append({'tag':tag,'page':pid,'frac':frac,'kind':kind,'pred':p[:200]})
json.dump(res,open('/tmp/marker_size_probe_out.json','w'),ensure_ascii=False,indent=1)
