import json, re, sys, time
from pathlib import Path
sys.path.insert(0,'src'); sys.path.insert(0,'train')
from PIL import Image
from point_ocr.image_resize import resize_for_ovis
from point_ocr.infer import generate_point_text

VAL=Path('data/splits_stage_q2_ocr_mt/val.jsonl')
rows=[json.loads(l) for l in open(VAL,encoding='utf-8')]
# same negatives, first N in file order
neg=[r for r in rows if (r['metadata'].get('is_negative') or not r['messages'][-1]['content'].strip())]
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
neg=neg[:N]

from unsloth import FastVisionModel
from peft import PeftModel
base=sys.argv[2] if len(sys.argv)>2 else 'checkpoints/q1_grpo_merged'
print('loading',base,flush=True)
model,tok=FastVisionModel.from_pretrained(base, load_in_4bit=True, use_gradient_checkpointing='unsloth')
FastVisionModel.for_inference(model)
t0=time.time()
cats={}
outs=[]
for i,r in enumerate(neg):
    m=r['metadata']; img=Image.open(r['images'][0]).convert('RGB')
    img=resize_for_ovis(img,min_pixels=448*448,max_pixels=2880*2880)
    prompt=r['messages'][0]['content']
    if prompt.startswith('<image>'): prompt=prompt[len('<image>'):]
    out=generate_point_text(model,tok,img,prompt,max_new_tokens=256,clean=True)
    p=out.cleaned.strip()
    s=re.search(r'<source>\s*(.*?)\s*</source>',p,re.S|re.I); t=re.search(r'<translation>\s*(.*?)\s*</translation>',p,re.S|re.I)
    ss=s.group(1) if s else None; tt=t.group(1) if t else None
    if not p: k='empty'
    elif ss is not None and tt is not None and not ss.strip() and not tt.strip(): k='empty_xml_shell'
    elif p=='<source>': k='trunc_source'
    elif ss is None and tt is None: k='plain_text'
    else: k='leak_xml'
    cats[k]=cats.get(k,0)+1
    outs.append({'sample_id':m.get('sample_id'),'region':m.get('region'),'cat':k,'pred':p[:200]})
    if (i+1)%10==0:
        el=time.time()-t0; print(f'{i+1}/{len(neg)} {el:.0f}s eta={(len(neg)-i-1)/((i+1)/el):.0f}s',flush=True)
eff=cats.get('empty',0)+cats.get('empty_xml_shell',0)+cats.get('trunc_source',0)
print('BASE',base)
print('cats',cats)
print(f'effective-empty {eff}/{len(neg)} = {eff/len(neg):.3f}')
json.dump(outs,open('/tmp/base_neg_probe_out.json','w'),ensure_ascii=False,indent=1)
