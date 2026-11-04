"""Builds one self-contained viewer page (outputs/viewer/index.html). Standard library only."""
import base64
import csv
import io
import json
from pathlib import Path

KNOWN_FIGS = {"fig_loss_time_map", "fig_contour_coverage_intensity_time", "fig_model_llr_vs_R", "fig_behaviour_classes",
              "fig_feature_kymograph", "fig_feature_size_brightness_time", "fig_feature_size_brightness_groups",
              "fig_event_vs_random", "fig_changepoint"}

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
.figs img{width:100%;max-width:1100px;border-radius:8px;border:1px solid var(--line);margin:6px 0}
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
<div class="tab" id="t_time"><div class="warn">Yükseklik yok: elimizde yalnızca JPG yoğunluğu var, kalibre değil. Konum ve zaman bilgisi piksel ve kare sırası cinsinden; sıra geçici, zaman tahmini.</div>
<div class="card" style="margin-bottom:12px"><div id="p8txt"></div></div><div class="figs" id="figs8"></div>
<div class="card scroll" style="margin:12px 0"><b>Literatürle büyüklük sırası kıyası</b> <span class="k">(Abdelhady 2003, kitaplıkta DOĞRULANMADI; bizde enzim ekleme zamanı bilinmiyor, sonuç sayılmaz)</span><table id="lit"></table></div>
<div class="card" style="margin-bottom:12px"><div id="p7txt"></div></div><div class="figs" id="figs7a"></div></div>
<div class="tab" id="t_mech"><div class="warn">Hocanın üç mekanizma sorusu. Parlak özelliklerin ne olduğu (enzim mi dendrimer mi) bilinmiyor. Tek molekül, tek deney, kontrol yok: sayılar <b>tanımlayıcıdır</b>, hiçbir mekanizma bu veriyle doğrulanmış ya da çürütülmüş sayılmaz.</div>
<div id="mechcards"></div><div class="card" style="margin:10px 0"><b>Parlak özelliklerin davranış sınıfları</b> <span class="k">(kimlik ataması değil)</span><table id="cls"></table></div><div class="figs" id="figs6"></div>
<div class="card scroll" style="margin-top:12px"><b>Parlak özellik izleri</b> <span class="k">(en az 4 görünüm; sıra: ilk kare)</span><table id="trk"></table></div>
<div class="card scroll" style="margin-top:12px"><b>Her kopma adayının çevresindeki parlak özellikler</b><table id="evf"></table></div></div>
<div class="tab" id="t_fix"><div class="card"><div class="warn">Maskeyi elle düzeltme aracı. Resimde yanlış bulunan yere tıkla: <b>SİL</b> = DNA olmayan yeri maskeden çıkar, <b>EKLE</b> = kaçan DNA'yı maskeye ekle. Bu işaretler hocanın "elle doğru" referansı olur ve doğruluk ölçümü için kullanılır. Kareyi üstteki Kareler sekmesinden seç (ya da ok tuşları).</div>
<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:8px"><b id="fixlbl"></b>
<label><input type="radio" name="fm" value="erase" checked> SİL</label><label><input type="radio" name="fm" value="add"> EKLE</label>
<label>yarıçap (piksel) <input type="number" id="fr" value="6" min="1" max="60" style="width:70px"></label>
<label><input type="checkbox" id="fmask" checked> maskeyi göster</label>
<button class="btn" id="fprev">◀</button><button class="btn" id="fnext">▶</button>
<button class="btn" id="fdl">corrections.csv indir</button><button class="btn" id="fclr">bu karedekileri temizle</button></div>
<div class="row"><canvas id="fc" style="width:100%;max-width:760px;height:auto;cursor:crosshair"></canvas>
<div class="card" style="flex:1;min-width:260px"><b>İşaretler (tüm kareler: <span id="fcnt">0</span>)</b><div class="scroll" id="flist" style="max-height:50vh"></div>
<p class="k">İndirdiğin dosyayı <code>afm_project/annotations/corrections.csv</code> olarak kaydet, sonra <code>bash run_all.sh</code> çalıştır. Maske buna göre düzelir. Tarayıcı işaretleri bu bilgisayarda hatırlar.</p></div></div></div></div>
<div class="tab" id="t_fig"><div class="figs" id="figs"></div></div>
<div class="tab" id="t_rep"><div class="card"><pre id="rep"></pre></div></div>
</main><script>
const D=__DATA__;
const tabs=[["frames","Kareler"],["events","Aday olaylar"],["know","Kitaplık"],["time","Zaman ve yer"],["mech","Mekanizmalar"],["fix","Düzeltme"],["fig","Grafikler"],["rep","Rapor"]];
const nav=document.getElementById("nav");
tabs.forEach(([k,n],i)=>{const b=document.createElement("button");b.textContent=n;b.onclick=()=>show(k);b.id="b_"+k;nav.appendChild(b)});
function show(k){tabs.forEach(([t])=>{document.getElementById("t_"+t).classList.toggle("on",t==k);document.getElementById("b_"+t).classList.toggle("on",t==k)})}
show("frames");
const TR={width_px:"genişlik (piksel)",height_px:"yükseklik (piksel)",crop_top:"kırpma üst",crop_bottom:"kırpma alt",crop_left:"kırpma sol",crop_right:"kırpma sağ",noise_level:"gürültü seviyesi",mask_source:"maske kaynağı",n_objects:"nesne sayısı",n_closed_loop_candidates:"kapalı halka adayı",n_open_candidates:"açık yapı adayı",n_small_objects:"küçük nesne sayısı",n_loop_if_gaps_bridged:"boşluklar kapanırsa halka sayısı",n_scan_line_suspects:"tarama çizgisi şüphelisi",mask_area_px:"maske alanı (piksel)",skeleton_length_px_total:"toplam iskelet uzunluğu (piksel)",largest_object_skeleton_px:"en uzun parça (piksel)",n_bright_features:"parlak özellik sayısı",nm_per_pixel_used:"nm/piksel (tahmin)",scale_status:"ölçek durumu",skeleton_length_nm_estimated:"iskelet uzunluğu (nm, tahmini)",time_earliest_s:"en erken zaman (sn)",time_latest_s:"en geç zaman (sn)",time_status:"zaman durumu",n_objects_tracked:"izlenen nesne",largest_object_px:"en uzun parça (piksel)",total_skeleton_px:"toplam iskelet (piksel)",continuity_fraction:"süreklilik oranı",mean_bright_area_px:"ort. parlak alan (piksel)",split_candidates_reaching_frame:"bu kareye ulaşan kopma adayı",persistent_split_candidates:"kalıcı kopma adayı",largest_object_nm_estimated:"en uzun parça (nm, tahmini)",split_candidates_cumulative:"birikmiş kopma adayı",persistent_splits_cumulative:"birikmiş kalıcı kopma",bright_kept_from_previous:"önceki kareden korunan parlak özellik"};
const T=k=>TR[k]||k;
const VT={estimated:"tahmini",unknown:"bilinmiyor",confirmed:"doğrulanmış",unverified:"doğrulanmadı",user_verified:"kullanıcı doğruladı",supervisor_verified:"hoca doğruladı",automatic:"otomatik",continuation:"devam",split_candidate:"kopma adayı",merge_candidate:"birleşme adayı",object_lost:"nesne kayboldu",new_object:"yeni nesne",complex_change:"karmaşık değişim",persistent:"kalıcı",transient_rejoined:"geçici, tekrar birleşti",unresolved_end_of_series:"seri bitti, belirsiz",gap_small_possible_artifact:"küçük boşluk, olası görüntü hatası","candidate only; not a confirmed cleavage":"sadece aday; doğrulanmış kesim değil"};
const V=v=>{v=String(v);return v.split(";").map(x=>VT[x.trim()]||x).join("; ")};
const ET={rank_from:"sıra (önce)",rank_to:"sıra (sonra)",filename_from:"dosya (önce)",filename_to:"dosya (sonra)",event:"olay",from_objects:"önceki nesneler",to_objects:"sonraki nesneler",n_pieces_after:"sonraki parça sayısı",min_gap_px:"en küçük boşluk (piksel)",confidence:"güven",registration_shift_px:"hizalama kayması (piksel)",flags:"uyarılar",persistence:"kalıcılık",note:"not",event_y_px:"olay y (piksel)",event_x_px:"olay x (piksel)"};
const TOPIC={E001:"kullanılan plazmit boyutu",E002:"dendrimer (G4 PAMAM, 64 yüzey amini)",E003:"AFM kare süresi",E004:"AFM görüntü çözünürlüğü",E005:"enzim dozu",E006:"çıplak DNA: bozunma başlangıcı",E007:"çıplak DNA: tam parçalanma",E008:"ömür (çıplak / 0.5:1 G4:DNA)",E009:"ömür (1:1 G4:DNA)",E010:"bozunma başlangıcı (0.5:1 ve 1:1)",E011:"5:1 oranında koruma",E012:"2 saat inkübasyondan sonra parçalanma",E013:"süslenmiş DNA genişlik / yükseklik (0. dk)",E014:"süslenmemiş DNA genişlik / yükseklik",E015:"gevşemiş DNA uzama aralığı",E016:"çok halkalı molekül uzunluğu",E017:"açık halka yolunun oranı",E018:"pBR322 uzunluğu",E019:"DNA kalıcılık uzunluğu (sıvı, 3B)",E020:"DNA kalıcılık uzunluğu (hava, 2B)",E021:"kalıcılık uzunluğu referansı (optik cımbız)",E022:"sıvıda tarama alanı ve piksel",E023:"sıvıda ölçülen pBR322 uzunluğu",E024:"G4 / G5 formül ağırlığı",E025:"plazmit DNA ortalama uzunluğu (arka plan)",E026:"G4 dendripleks genişlik / yükseklik (RNA)",E027:"DNase I, G4 + pDNA, 15 dk inkübasyon (önceki çalışma)",E028:"DNase I, G4 + pDNA, 2 saat inkübasyon (önceki çalışma)",E029:"DNA kalıcılık uzunluğu",E030:"o çalışmada AFM piksel boyutu"};
const F=D.frames;let cur=0,timer=null;
const sl=document.getElementById("sl");sl.max=F.length-1;
function src(f,l){const e=D.img&&D.img[l+"/"+f.stem];if(e)return e;return l=="orig"?D.raw+f.filename:l=="mask"?D.p2+"masks/"+f.stem+"_mask.png":l=="skel"?D.p2+"skeletons/"+f.stem+"_skeleton.png":D.p2+"overlays/"+f.stem+".png"}
function go(i){cur=Math.max(0,Math.min(F.length-1,i));sl.value=cur;const f=F[cur];
document.getElementById("img").src=src(f,document.getElementById("layer").value);
document.getElementById("lbl").textContent="sıra "+f.rank+" / "+F.length+" · dosya "+f.filename;
document.getElementById("meas").innerHTML=Object.entries(f.m).map(([k,v])=>"<tr><td class=k>"+T(k)+"</td><td>"+V(v)+"</td></tr>").join("");draw()}
sl.oninput=()=>go(+sl.value);document.getElementById("layer").onchange=()=>go(cur);
document.getElementById("prev").onclick=()=>go(cur-1);document.getElementById("next").onclick=()=>go(cur+1);
document.getElementById("play").onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent="▶ Oynat"}else{this.textContent="⏸ Durdur";timer=setInterval(()=>{if(cur>=F.length-1)cur=-1;go(cur+1)},350)}};
document.addEventListener("keydown",e=>{if(e.key=="ArrowRight")go(cur+1);if(e.key=="ArrowLeft")go(cur-1)});
const ms=document.getElementById("metric");D.metrics.forEach(m=>{const o=document.createElement("option");o.value=m;o.textContent=T(m);ms.appendChild(o)});ms.onchange=draw;
function draw(){const c=document.getElementById("chart"),g=c.getContext("2d"),m=ms.value;if(!m)return;
const y=F.map(f=>parseFloat(f.m[m])),ok=y.filter(v=>!isNaN(v));g.clearRect(0,0,c.width,c.height);if(!ok.length)return;
const lo=Math.min(...ok),hi=Math.max(...ok),W=c.width,H=c.height,p=36,sx=i=>p+i*(W-2*p)/Math.max(1,F.length-1),sy=v=>H-p-(hi==lo?.5:(v-lo)/(hi-lo))*(H-2*p);
const st=getComputedStyle(document.body);g.fillStyle=st.color;g.font="16px sans-serif";g.fillText(hi.toFixed(2),2,p);g.fillText(lo.toFixed(2),2,H-p+4);
g.strokeStyle="#4a7be0";g.lineWidth=2;g.beginPath();let s=false;y.forEach((v,i)=>{if(isNaN(v)){s=false;return}s?g.lineTo(sx(i),sy(v)):g.moveTo(sx(i),sy(v));s=true});g.stroke();
g.strokeStyle="#e04a4a";g.beginPath();g.moveTo(sx(cur),p/2);g.lineTo(sx(cur),H-p);g.stroke()}
document.getElementById("chart").onclick=e=>{const r=e.target.getBoundingClientRect();go(Math.round(((e.clientX-r.left)/r.width*1200-36)/(1200-72)*(F.length-1)))};
const ev=document.getElementById("evt");if(D.events.length){const ks=Object.keys(D.events[0]);ev.innerHTML="<tr>"+ks.map(k=>"<th>"+(ET[k]||k)+"</th>").join("")+"</tr>"+D.events.map(e=>"<tr class=click data-r='"+e.rank_to+"'>"+ks.map(k=>"<td>"+V(e[k])+"</td>").join("")+"</tr>").join("");
ev.querySelectorAll("tr.click").forEach(r=>r.onclick=()=>{show("frames");go(F.findIndex(f=>f.rank==r.dataset.r))})}else ev.innerHTML="<tr><td>Önce run_all.sh çalıştır.</td></tr>";
const kn=document.getElementById("kn");function kr(){const q=document.getElementById("q").value.toLowerCase();
kn.innerHTML="<tr><th>konu</th><th>değer</th><th>birim</th><th>orijinal alıntı (İngilizce, değiştirilmedi)</th><th>kaynak</th><th>durum</th></tr>"+D.know.filter(r=>JSON.stringify(r).toLowerCase().includes(q)).map(r=>"<tr><td>"+(TOPIC[r.evidence_id]||r.topic)+"</td><td>"+r.value+"</td><td>"+r.unit+"</td><td>"+r.quote+(r.context_note?"<br><span class=k>"+r.context_note+"</span>":"")+"</td><td>"+r.src+"</td><td>"+V(r.verified)+"</td></tr>").join("")}
document.getElementById("q").oninput=kr;kr();
document.getElementById("figs").innerHTML=D.figs.map(f=>"<div><div class=k>"+f+"</div><img src='"+((D.figimg&&D.figimg[f])||D.p5+f)+"'></div>").join("")||"<div class=card>Grafik yok, önce run_all.sh çalıştır.</div>";
document.getElementById("rep").textContent=D.report||"Rapor yok, önce run_all.sh çalıştır.";

let FIX=[];try{FIX=JSON.parse(localStorage.getItem("afm_fix")||"[]")}catch(e){}
function fsave(){try{localStorage.setItem("afm_fix",JSON.stringify(FIX))}catch(e){}}
const fc=document.getElementById("fc"),fg=fc.getContext("2d"),fim=new Image(),fmk=new Image();let fok=false,mok=false;
function fdraw(){const f=F[cur];document.getElementById("fixlbl").textContent="sıra "+f.rank+" · "+f.filename;
if(!fok)return;fc.width=fim.naturalWidth;fc.height=fim.naturalHeight;fg.drawImage(fim,0,0);
if(mok&&document.getElementById("fmask").checked){fg.globalAlpha=.4;fg.drawImage(fmk,+f.m.crop_left||0,+f.m.crop_top||0);fg.globalAlpha=1}
FIX.filter(x=>x.filename==f.filename).forEach(x=>{fg.strokeStyle=x.action=="erase"?"#ff2d2d":"#22dd55";fg.lineWidth=2;fg.beginPath();fg.arc(x.x,x.y,x.r,0,7);fg.stroke()});
document.getElementById("fcnt").textContent=FIX.length;
document.getElementById("flist").innerHTML=FIX.map((x,i)=>x.filename==f.filename?"<div>"+(x.action=="erase"?"SİL":"EKLE")+" ("+x.x+", "+x.y+") r="+x.r+" <a href=# data-i="+i+">x</a></div>":"").join("")||"<span class=k>Bu karede işaret yok.</span>";
document.querySelectorAll("#flist a").forEach(a=>a.onclick=e=>{e.preventDefault();FIX.splice(+a.dataset.i,1);fsave();fdraw()})}
function fload(){const f=F[cur];fok=mok=false;fim.onload=()=>{fok=true;fdraw()};fim.src=src(f,"orig");fmk.onload=()=>{mok=true;fdraw()};fmk.src=src(f,"mask")}
fc.onclick=e=>{const r=fc.getBoundingClientRect(),sc=fc.width/r.width,f=F[cur];
FIX.push({filename:f.filename,action:document.querySelector("input[name=fm]:checked").value,x:Math.round((e.clientX-r.left)*sc),y:Math.round((e.clientY-r.top)*sc),r:+document.getElementById("fr").value||6});fsave();fdraw()};
document.getElementById("fmask").onchange=fdraw;
document.getElementById("fprev").onclick=()=>go(cur-1);document.getElementById("fnext").onclick=()=>go(cur+1);
document.getElementById("fclr").onclick=()=>{const n=F[cur].filename;FIX=FIX.filter(x=>x.filename!=n);fsave();fdraw()};
document.getElementById("fdl").onclick=()=>{const t="filename,action,x,y,radius_px\n"+FIX.map(x=>[x.filename,x.action,x.x,x.y,x.r].join(",")).join("\n")+"\n";
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([t],{type:"text/csv"}));a.download="corrections.csv";a.click()};
const _go=go;go=function(i){_go(i);fload()};

const TT={behaviour_class:"davranış sınıfı",n_tracks:"iz sayısı",median_area_px:"ortanca alan (piksel)",median_relative_intensity:"ortanca göreli parlaklık",median_lifespan_frames:"ortanca ömür (kare)",track_id:"iz no",first_rank:"ilk kare",last_rank:"son kare",n_detections:"görünme sayısı",lifespan_frames:"ömür (kare)",missed_frames:"kaçırılan kare",motion_class:"hareket",max_dist_from_median_px:"en büyük sapma (piksel)",net_displacement_px:"net yer değiştirme (piksel)",s_first_px:"ilk konum s",s_last_px:"son konum s",net_drift_along_contour_px:"DNA boyunca kayma (piksel)",mean_area_px:"ort. alan (piksel)",area_change_px:"alan değişimi",mean_relative_intensity:"ort. göreli parlaklık",intensity_change:"parlaklık değişimi",fate:"son durum",rank_from:"sıra (önce)",rank_to:"sıra (sonra)",persistence:"kalıcılık",min_gap_px:"boşluk (piksel)",event_s_px:"olay konumu s",n_features_in_frame:"karedeki özellik",nearest_feature_px:"en yakın özellik (piksel)",n_features_near:"yakındaki özellik",near_kept_in_place:"yerinde kaldı",near_moved:"yer değiştirdi",near_lost:"kayboldu",near_unknown:"bilinmiyor"};
const VT2={persistent_stationary:"kalıcı ve sabit",transient_short:"kısa ömürlü",intermediate:"ara",stationary:"sabit",moved:"yer değiştirdi",short_track_uncertain:"kısa iz, belirsiz",wandering_uncertain:"dolaşıyor, belirsiz",present_to_last_frame:"son kareye kadar görünüyor"};
function V2(v){v=String(v);if(VT2[v])return VT2[v];if(v.startsWith("not_seen_after_rank_"))return v.replace("not_seen_after_rank_","kare ")+" sonrası görünmüyor";return V(v)}
function tbl(el,rows,cols){el.innerHTML="<tr>"+cols.map(k=>"<th>"+(TT[k]||k)+"</th>").join("")+"</tr>"+rows.map(r=>"<tr>"+cols.map(k=>"<td>"+V2(r[k])+"</td>").join("")+"</tr>").join("")}
if(D.p6&&D.p6.mech.length){document.getElementById("mechcards").innerHTML=D.p6.mech.map(m=>"<div class=card style='margin-bottom:10px'><b>"+m.hypothesis+": "+m.hypothesis_tr+"</b><p><span class=k>Doğruysa beklenen:</span> "+m.expected_if_true_tr+"</p><p><span class=k>Gözlenen:</span> "+m.observed+"</p><p class=k>Sınırlama: "+m.limitation_tr+"</p></div>").join("");
tbl(document.getElementById("trk"),D.p6.tracks.filter(t=>+t.n_detections>=4),["track_id","first_rank","last_rank","n_detections","motion_class","max_dist_from_median_px","net_drift_along_contour_px","mean_area_px","area_change_px","mean_relative_intensity","fate"]);
tbl(document.getElementById("evf"),D.p6.events,["rank_from","rank_to","persistence","min_gap_px","event_s_px","nearest_feature_px","n_features_near","near_kept_in_place","near_moved","near_lost"]);
document.getElementById("figs6").innerHTML=Object.entries(D.p6.figs).map(([n,u])=>"<div><div class=k>"+n+"</div><img src='"+u+"'></div>").join("")}
else document.getElementById("mechcards").innerHTML="<div class=card>Mekanizma sonuçları yok. Önce Terminal'de run_all.sh çalıştır.</div>";

const CAP={fig_loss_time_map:"Kopma nerede, ne zaman: ilk 22 karede DNA konturu üzerindeki boşluklar (renk = ilk boşluk karesi, x = aday kopma, yanındaki sayı kare)",fig_contour_coverage_intensity_time:"Soldan sağa: kontur örtüşmesi ile süreklilik, DNA parlaklık seviyesi, soluk iskelet oranı (hepsi JPG yoğunluğu, yükseklik değil)",fig_model_llr_vs_R:"Olasılık modeli: kopma yerinin parlak özelliğe yakın / uzak olması, bağımsız modele göre (0'ın altı = bağımsızdan kötü)",fig_behaviour_classes:"Parlak özelliklerin davranış sınıfları (boyut, parlaklık, ömür)",fig_feature_kymograph:"Parlak özellikler DNA boyunca zamanla (nokta büyüklüğü = alan)",fig_feature_size_brightness_time:"Her izin boyutu ve göreli parlaklığı zamanla",fig_feature_size_brightness_groups:"İzlerin boyut-parlaklık dağılımı ve ömürleri",fig_changepoint:"Kopma ne zaman: en uzun parça ve süreklilik serisinde kırılma noktası (kırmızı basamak, açık kırmızı = %95 aralık, noktalı çizgi = kalıcı kopma adayı), sağda 36 eşik ayarında kırılma noktası",fig_event_vs_random:"Kopma yerleri ile rastgele DNA noktalarının parlak özelliğe uzaklığı"};
function figs(el,obj){el.innerHTML=Object.entries(obj).map(([n,u])=>"<div><div class=k>"+(CAP[n.replace(".png","")]||n)+"</div><img src='"+u+"'></div>").join("")}
if(D.p7&&D.p7.figs){figs(document.getElementById("figs7a"),Object.fromEntries(Object.entries(D.p7.figs).filter(([n])=>/loss_time|coverage/.test(n))));
document.getElementById("p7txt").innerHTML=(D.p7.summary||"").split("\n").filter(x=>x.trim()).map(x=>"<p>"+x.replace(/^[-#]+ ?/,"").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>").join("");
tbl(document.getElementById("cls"),D.p7.classes,["behaviour_class","n_tracks","median_area_px","median_relative_intensity","median_lifespan_frames"]);
if(D.p7.mech.length){document.getElementById("mechcards").innerHTML=D.p7.mech.map(m=>"<div class=card style='margin-bottom:10px'><b>"+m.hypothesis+": "+m.hypothesis_tr+"</b><p><span class=k>Model sayıları:</span> "+m.model_numbers+"</p><p><span class=k>Okuma:</span> "+m.reading_tr+"</p></div>").join("")}
figs(document.getElementById("figs6"),Object.fromEntries(Object.entries({...D.p6.figs,...D.p7.figs}).filter(([n])=>!/loss_time|coverage/.test(n))))}

if(D.p8&&D.p8.summary){document.getElementById("p8txt").innerHTML=D.p8.summary.split("\n").filter(x=>x.trim()).map(x=>"<p>"+x.replace(/^[-#]+ ?/,"").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>").join("");
figs(document.getElementById("figs8"),D.p8.figs);
const LT={source:"kaynak",what_tr:"ne",value:"değer",quote:"orijinal alıntı / not",verified:"durum"};
const lt=document.getElementById("lit");const lc=["source","what_tr","value","quote","verified"];
lt.innerHTML="<tr>"+lc.map(k=>"<th>"+LT[k]+"</th>").join("")+"</tr>"+D.p8.lit.map(r=>"<tr>"+lc.map(k=>"<td>"+V(r[k])+"</td>").join("")+"</tr>").join("")}
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


def _data_uri(path, jpeg=False):
    """Embed an image in the page so it works wherever the file is opened."""
    path = Path(path)
    if not path.exists():
        return None
    if jpeg:
        try:
            from PIL import Image
            buf = io.BytesIO()
            Image.open(path).convert("RGB").save(buf, "JPEG", quality=88)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


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
    img = {}
    for f in frames:
        for layer, path, jpg in (
                ("overlay", out / "phase2" / "overlays" / (f["stem"] + ".png"), True),
                ("mask", out / "phase2" / "masks" / (f["stem"] + "_mask.png"), False),
                ("skel", out / "phase2" / "skeletons" / (f["stem"] + "_skeleton.png"), False),
                ("orig", root / "data" / "raw" / f["filename"], False)):
            uri = _data_uri(path, jpg)
            if uri:
                img[f"{layer}/{f['stem']}"] = uri
    figs_b = {}
    p6d = out / "phase6"
    p6 = {"mech": _csv(p6d / "mechanism_summary.csv"), "tracks": _csv(p6d / "feature_tracks.csv"),
          "events": _csv(p6d / "event_feature_association.csv"),
          "figs": {q.name: _data_uri(q) for q in sorted(p6d.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p6d.exists() else {}}
    p7d = out / "phase7"
    p7 = {"mech": _csv(p7d / "mechanism_model_summary.csv"), "classes": _csv(p7d / "behaviour_class_summary.csv"),
          "summary": (p7d / "summary_tr.md").read_text(encoding="utf-8") if (p7d / "summary_tr.md").exists() else "",
          "figs": {q.name: _data_uri(q) for q in sorted(p7d.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p7d.exists() else {}}
    p8d = out / "phase8"
    p8 = {"summary": (p8d / "summary_tr.md").read_text(encoding="utf-8") if (p8d / "summary_tr.md").exists() else "",
          "lit": _csv(p8d / "literature_comparison.csv"),
          "figs": {q.name: _data_uri(q) for q in sorted(p8d.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p8d.exists() else {}}
    figimg = {n: _data_uri(out / "phase5" / n) for n in figs}
    data = {"p8": p8, "p7": p7, "p6": p6, "img": img, "figimg": figimg, "frames": frames, "metrics": metrics, "events": _csv(out / "phase3" / "events.csv"),
            "know": know, "figs": figs, "report": rep.read_text(encoding="utf-8") if rep.exists() else "",
            "raw": "../../data/raw/", "p2": "../phase2/", "p5": "../phase5/"}
    dest = out / "viewer"
    dest.mkdir(parents=True, exist_ok=True)
    html = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (dest / "index.html").write_text(html, encoding="utf-8")
    return dest / "index.html"
