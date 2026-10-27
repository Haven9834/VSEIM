"""Static local gallery; no telemetry, external assets or server required."""
import json


def write_gallery(path, records):
    data = json.dumps(records, ensure_ascii=True).replace("<", "\\u003c")
    template = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Vessel observations</title><style>
body{font:16px system-ui;margin:24px;background:#f5f5f5}header{position:sticky;top:0;background:white;padding:12px}
article{background:white;padding:10px;border:1px solid #bbb}img{width:100%}#grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
button,input{padding:8px;margin:4px}small{display:block}#pager{margin:12px 0}
</style><header><input id="query" placeholder="Camera, clip or frame"><button id="export">Export selection</button><span id="count"></span></header>
<div id="pager"></div><div id="grid"></div><script>
const records=__DATA__,chosen=new Set();let page=0;const size=40;
const el=id=>document.getElementById(id);
function draw(){const q=el('query').value.toLowerCase(),shown=records.filter(r=>[r.camera,r.clip,r.frame].join(' ').toLowerCase().includes(q));
const max=Math.max(0,Math.ceil(shown.length/size)-1);page=Math.min(page,max);el('grid').replaceChildren();el('pager').replaceChildren();
for(const [label,delta]of [['Previous',-1],['Next',1]]){const b=document.createElement('button');b.textContent=label;b.onclick=()=>{page=Math.max(0,Math.min(max,page+delta));draw()};el('pager').append(b)}
el('pager').append(document.createTextNode(` Page ${page+1}/${max+1}; ${shown.length} images`));
for(const r of shown.slice(page*size,(page+1)*size)){const a=document.createElement('article'),link=document.createElement('a'),im=document.createElement('img');
link.href=r.image;link.target='_blank';im.src=r.image;im.loading='lazy';link.append(im);a.append(link);
const lab=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.checked=chosen.has(r.id);
box.onchange=()=>{box.checked?chosen.add(r.id):chosen.delete(r.id);el('count').textContent=`${chosen.size} selected`};
lab.append(box,document.createTextNode(`${r.camera} / ${r.clip} / ${r.frame}`));a.append(lab);el('grid').append(a)}}
el('query').oninput=()=>{page=0;draw()};el('export').onclick=()=>{const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify({ids:[...chosen]},null,2)],{type:'application/json'}));a.download='selection.json';a.click();URL.revokeObjectURL(a.href)};draw();
</script></html>'''
    path.write_text(template.replace("__DATA__", data), encoding="utf-8")
