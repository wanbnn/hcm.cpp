import concurrent.futures, hashlib, json, re, urllib.request
from pathlib import Path
ids=[15668,17186,18982,19013,23201,24412,24853,25148,27715,28055,28122,30068,31093,31190,33572,34625,34626,36404,36560,39086,39107,40093,41389,69209,69596,70662,78897]
style=Path('/run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v32/style/gutenberg')
style.mkdir(parents=True,exist_ok=True)
existing={p.name.split('_',1)[0] for p in style.glob('*_*.txt')}
UA='TsetlinPortugueseCorpus/3.2 (domain books)'
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':UA})
    with urllib.request.urlopen(req,timeout=120) as r: return r.read()
def strip(t):
    a=re.search(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',t,re.I|re.S)
    b=re.search(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',t,re.I|re.S)
    body=t[a.end() if a else 0:b.start() if b else len(t)]
    return re.sub(r'\s+',' ',body).strip()
def slug(s): return re.sub(r'[^\w\-]+','_',s,flags=re.UNICODE).strip('_')[:80]
def one(i):
    if str(i) in existing: return None
    url=f'https://gutenberg.pglaf.org/cache/epub/{i}/pg{i}.txt'
    try: data=get(url)
    except Exception as e: return {'id':i,'error':str(e)}
    text=data.decode('utf-8',errors='replace'); body=strip(text)
    if len(body.split())<3000: return {'id':i,'skip':'short'}
    title=re.search(r'Title:\s*(.+)',text)
    name=f'{i:05d}_{slug(title.group(1).strip() if title else str(i))}'
    (style/(name+'.txt')).write_text(body+'\n',encoding='utf-8')
    return {'id':i,'title':title.group(1).strip() if title else '', 'url':url,
            'sha256':hashlib.sha256(data).hexdigest(),'characters':len(body),
            'words':len(body.split()),'text_file':str((style/(name+'.txt')).relative_to(style.parent.parent))}
out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    for r in concurrent.futures.as_completed([ex.submit(one,i) for i in ids]):
        r=r.result()
        if r: out.append(r); print(r,flush=True)
Path('/run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v32/domain_books_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('done',len(out))
