"""Builds one self-contained viewer page (outputs/viewer/index.html). Standard library only."""
import csv
import json
from pathlib import Path

PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>AFM DNA Analiz Görüntüleyici</title>
<style>
:root{--bg:#f6f7f9;--fg:#1c2330;--card:#fff;--line:#d9dee6;--acc:#2255c4}
@media(prefers-color-scheme:dark){:root{--bg:#161a21;--fg:#e6e9ef;--card:#1f242d;--line:#343b48;--acc:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px -apple-system,system-ui,sans-serif}
header{padding:12px 20px;border-bottom:1px solid var(--line);background:var(--card)}
header b{font-size:17px}header span{opacity:.7;margin-left:10px;font-size:13px}
nav{display:flex;gap:4px;padding:8px 20px;border-bottom:1px solid var(--line);background:var(--card)}
nav button{border:0;background:none;color:var(--fg);padding:8px 14px;border-radius:8px;cursor:pointer;font-size:14px}
nav button.on{background:var(--acc);color:#fff}
main{padding:16px 20px}.tab{display:none}.tab.on{display:block}
.row{display:flex;gap:16px;flex-wrap:wrap}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
img.big{width:100%;max-width:1100px;image-rendering:pixelated;border-radius:8px;display:block}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--card)}tr.click{cursor:pointer}tr.click:hover{background:rgba(120,140,200,.15)}
input[type=range]{width:100%}input[type=text],select{padding:7px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.btn{padding:7px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.warn{background:rgba(220,160,0,.15);border:1px solid rgba(220,160,0,.5);padding:8px 12px;border-radius:8px;margin-bottom:12px;font-size:13px}
canvas{width:100%;height:260px;background:var(--card);border:1px solid var(--line);border-radius:12px}
.k{opacity:.7}.scroll{max-height:70vh;overflow:auto}pre{white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace}
.figs img{width:100%;max-width:640px;border-radius:8px;border:1px solid var(--line);margin:6px 0}
</style></head><body>
<header><b>AFM DNA Analiz Görüntüleyici</b><span>ön çalışma · ölçümler piksel cinsinden · hiçbir olay "kesin" değil</span></header>
<nav id="nav"></nav><main>
<div class="tab" id="t_frames">
<div class="warn">Sıra geçici, ölçek ve zaman tahmin. Parlak noktalar "parlak özellik / aday bağlı parçacık"tır, kimlikleri bilinmiyor.</div>
<div class="row"><div class="card" style="flex:1;min-width:320px">
<div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap">
<select id="layer"><option value="overlay">Analiz katmanı</option><option value="orig">Orijinal</option><option value="mask">Maske</option><option value="skel">İskelet</option></select>
<button class="btn" id="prev">◀</button><button class="btn" id="play">▶ Oynat</button><button class="btn" id="next">▶</button>
<span id="lbl" class="k" style="align-self:center"></span></div>
<input type="range" id="sl" min="0" value="0"><img class="big" id="img"></div>
<div class="card" style="width:330px"><b>Bu kare</b><table id="meas"></table></div></div>
<div class="card" style="margin-top:16px"><b>Seçili ölçüm, kare sırasına göre</b>
<select id="metric"></select><canvas id="chart" width="1200" height="260"></canvas></div></div>
<div class="tab" id="t_events"><div class="card scroll"><div class="warn">Sadece aday olay. Satıra tıklayınca o kareye gidersin.</div><table id="evt"></table></div></div>
<div class="tab" id="t_know"><div class="card"><input type="text" id="q" placeholder="ara (ör. DNase, persistence)" style="width:100%;margin-bottom:8px">
<div class="warn">Hepsi "unverified". Orijinal makaleden kontrol etmeden yayına koyma.</div><div class="scroll"><table id="kn"></table></div></div></div>
<div class="tab" id="t_fig"><div class="figs" id="figs"></div></div>
<div class="tab" id="t_rep"><div class="card"><pre id="rep"></pre></div></div>
</main><script>
const D=__DATA__;
const tabs=[["frames","Kareler"],["events","Aday olaylar"],["know","Kitaplık"],["fig","Grafikler"],["rep","Rapor"]];
const nav=document.getElementById("nav");
tabs.forEach(([k,n],i)=>{const b=document.createElement("button");b.textContent=n;b.onclick=()=>show(k);b.id="b_"+k;nav.appendChild(b)});
function show(k){tabs.forEach(([t])=>{document.getElementById("t_"+t).classList.toggle("on",t==k);document.getElementById("b_"+t).classList.toggle("on",t==k)})}
show("frames");
const F=D.frames;let cur=0,timer=null;
const sl=document.getElementById("sl");sl.max=F.length-1;
function src(f,l){return l=="orig"?D.raw+f.filename:l=="mask"?D.p2+"masks/"+f.stem+"_mask.png":l=="skel"?D.p2+"skeletons/"+f.stem+"_skeleton.png":D.p2+"overlays/"+f.stem+".png"}
function go(i){cur=Math.max(0,Math.min(F.length-1,i));sl.value=cur;const f=F[cur];
document.getElementById("img").src=src(f,document.getElementById("layer").value);
document.getElementById("lbl").textContent="sıra "+f.rank+" / "+F.length+" · dosya "+f.filename;
document.getElementById("meas").innerHTML=Object.entries(f.m).map(([k,v])=>"<tr><td class=k>"+k+"</td><td>"+v+"</td></tr>").join("");draw()}
sl.oninput=()=>go(+sl.value);document.getElementById("layer").onchange=()=>go(cur);
document.getElementById("prev").onclick=()=>go(cur-1);document.getElementById("next").onclick=()=>go(cur+1);
document.getElementById("play").onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent="▶ Oynat"}else{this.textContent="⏸ Durdur";timer=setInterval(()=>{if(cur>=F.length-1)cur=-1;go(cur+1)},350)}};
document.addEventListener("keydown",e=>{if(e.key=="ArrowRight")go(cur+1);if(e.key=="ArrowLeft")go(cur-1)});
const ms=document.getElementById("metric");D.metrics.forEach(m=>{const o=document.createElement("option");o.value=m;o.textContent=m;ms.appendChild(o)});ms.onchange=draw;
function draw(){const c=document.getElementById("chart"),g=c.getContext("2d"),m=ms.value;if(!m)return;
const y=F.map(f=>parseFloat(f.m[m])),ok=y.filter(v=>!isNaN(v));g.clearRect(0,0,c.width,c.height);if(!ok.length)return;
const lo=Math.min(...ok),hi=Math.max(...ok),W=c.width,H=c.height,p=36,sx=i=>p+i*(W-2*p)/Math.max(1,F.length-1),sy=v=>H-p-(hi==lo?.5:(v-lo)/(hi-lo))*(H-2*p);
const st=getComputedStyle(document.body);g.fillStyle=st.color;g.font="16px sans-serif";g.fillText(hi.toFixed(2),2,p);g.fillText(lo.toFixed(2),2,H-p+4);
g.strokeStyle="#4a7be0";g.lineWidth=2;g.beginPath();let s=false;y.forEach((v,i)=>{if(isNaN(v)){s=false;return}s?g.lineTo(sx(i),sy(v)):g.moveTo(sx(i),sy(v));s=true});g.stroke();
g.strokeStyle="#e04a4a";g.beginPath();g.moveTo(sx(cur),p/2);g.lineTo(sx(cur),H-p);g.stroke()}
document.getElementById("chart").onclick=e=>{const r=e.target.getBoundingClientRect();go(Math.round(((e.clientX-r.left)/r.width*1200-36)/(1200-72)*(F.length-1)))};
const ev=document.getElementById("evt");if(D.events.length){const ks=Object.keys(D.events[0]);ev.innerHTML="<tr>"+ks.map(k=>"<th>"+k+"</th>").join("")+"</tr>"+D.events.map(e=>"<tr class=click data-r='"+e.rank_to+"'>"+ks.map(k=>"<td>"+e[k]+"</td>").join("")+"</tr>").join("");
ev.querySelectorAll("tr.click").forEach(r=>r.onclick=()=>{show("frames");go(F.findIndex(f=>f.rank==r.dataset.r))})}else ev.innerHTML="<tr><td>Önce run_all.sh çalıştır.</td></tr>";
const kn=document.getElementById("kn");function kr(){const q=document.getElementById("q").value.toLowerCase();
kn.innerHTML="<tr><th>konu</th><th>değer</th><th>birim</th><th>alıntı</th><th>kaynak</th><th>durum</th></tr>"+D.know.filter(r=>JSON.stringify(r).toLowerCase().includes(q)).map(r=>"<tr><td>"+r.topic+"</td><td>"+r.value+"</td><td>"+r.unit+"</td><td>"+r.quote+(r.context_note?"<br><span class=k>"+r.context_note+"</span>":"")+"</td><td>"+r.src+"</td><td>"+r.verified+"</td></tr>").join("")}
document.getElementById("q").oninput=kr;kr();
document.getElementById("figs").innerHTML=D.figs.map(f=>"<div><div class=k>"+f+"</div><img src='"+D.p5+f+"'></div>").join("")||"<div class=card>Grafik yok, önce run_all.sh çalıştır.</div>";
document.getElementById("rep").textContent=D.report||"Rapor yok, önce run_all.sh çalıştır.";
go(0);
</script></body></html>"""


def _csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        x = float(v)
        return round(x, 2)
    except (TypeError, ValueError):
        return v


def build(root):
    root = Path(root)
    out = root / "outputs"
    p2 = _csv(out / "phase2" / "frames.csv")
    p4 = {r["filename"]: r for r in _csv(out / "phase4" / "frame_measurements.csv")}
    frames, metrics = [], []
    for r in p2:
        stem = f"{int(r['rank']):02d}_{Path(r['filename']).stem}"
        m = {k: _num(v) for k, v in {**r, **p4.get(r["filename"], {})}.items()
             if k not in ("rank", "filename")}
        frames.append({"rank": int(r["rank"]), "filename": r["filename"], "stem": stem, "m": m})
    if frames:
        metrics = [k for k, v in frames[0]["m"].items() if isinstance(v, float)]
    lib = {r["source_id"]: r for r in _csv(root / "knowledge" / "library.csv")}
    know = []
    for r in _csv(root / "knowledge" / "evidence.csv"):
        s = lib.get(r["source_id"], {})
        r["src"] = f"{s.get('authors', '').split(';')[0]} {s.get('year', '')}".strip() or r["source_id"]
        know.append(r)
    rep = out / "phase5" / "report.md"
    figs = sorted(p.name for p in (out / "phase5").glob("fig_*.png")) if (out / "phase5").exists() else []
    data = {"frames": frames, "metrics": metrics, "events": _csv(out / "phase3" / "events.csv"),
            "know": know, "figs": figs, "report": rep.read_text(encoding="utf-8") if rep.exists() else "",
            "raw": "../../data/raw/", "p2": "../phase2/", "p5": "../phase5/"}
    dest = out / "viewer"
    dest.mkdir(parents=True, exist_ok=True)
    html = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (dest / "index.html").write_text(html, encoding="utf-8")
    return dest / "index.html"
