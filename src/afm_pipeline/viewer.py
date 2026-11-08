"""Builds one self-contained viewer page (outputs/viewer/index.html). Standard library only."""
import base64
import csv
import io
import json
from pathlib import Path

KNOWN_FIGS = {"fig_mechanisms", "fig_when_where", "fig_contour_length_time", "fig_height_profile", "fig_feature_populations", "fig_track_montage", "fig_pair_spacing", "fig_loss_time_map", "fig_contour_coverage_intensity_time", "fig_model_llr_vs_R", "fig_behaviour_classes",
              "fig_feature_kymograph", "fig_feature_size_brightness_time", "fig_feature_size_brightness_groups",
              "fig_event_vs_random", "fig_changepoint"}

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>AFM DNA Analysis Viewer</title>
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
<header><b>AFM DNA Analysis Viewer</b><span>preliminary work · measurements in pixels · no event is "certain"</span></header>
<nav id="nav"></nav><main>
<div class="tab" id="t_frames">
<div class="warn">Order is provisional, scale and time are estimates. Bright spots are "bright features / candidate bound particles"; their identity is unknown.</div>
<div class="row"><div class="card" style="flex:1;min-width:320px">
<div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap">
<select id="layer"><option value="overlay">Analysis layer</option><option value="orig">Original</option><option value="mask">Mask</option><option value="skel">Skeleton</option></select>
<button class="btn" id="prev">◀</button><button class="btn" id="play">▶ Play</button><button class="btn" id="next">▶</button>
<span id="lbl" class="k" style="align-self:center"></span></div>
<input type="range" id="sl" min="0" value="0"><img class="big" id="img"></div>
<div class="card" style="width:330px"><b>This frame</b><table id="meas"></table></div></div>
<div class="card" style="margin-top:16px"><b>Selected measurement, in frame order</b>
<select id="metric"></select><canvas id="chart" width="1200" height="260"></canvas></div></div>
<div class="tab" id="t_events"><div class="card scroll"><div class="warn">Candidate events only. Click a row to go to that frame.</div><table id="evt"></table></div></div>
<div class="tab" id="t_know"><div class="card"><input type="text" id="q" placeholder="search (e.g. DNase, persistence)" style="width:100%;margin-bottom:8px">
<div class="warn">All "unverified". Do not publish without checking against the original paper.</div><div class="scroll"><table id="kn"></table></div></div></div>
<div class="tab" id="t_time"><div class="warn">No height: we only have JPG intensity, which is not calibrated. Position and time are in pixels and frame order; the order is provisional and the time is an estimate.</div>
<div class="card" style="margin-bottom:12px"><div id="p8txt"></div></div><div class="figs" id="figs8"></div>
<div class="card scroll" style="margin:12px 0"><b>Order-of-magnitude comparison with the literature</b> <span class="k">(Abdelhady 2003, NOT VERIFIED in the library; our enzyme addition time is unknown, so this is not a result)</span><table id="lit"></table></div>
<div class="card" style="margin-bottom:12px"><div id="p7txt"></div></div><div class="figs" id="figs7a"></div></div>
<div class="tab" id="t_mech"><div class="warn" style="background:#fff3cd"><b>This tab is the first version (Phase 7, strong detections only).</b> For the current, more careful evaluation see the "6. Mechanism model" tab.</div><div class="warn">The supervisor's three-mechanism question. What the bright features are (enzyme or dendrimer) is unknown. One molecule, one experiment, no control: the numbers are <b>descriptive</b>, and no mechanism counts as confirmed or refuted with these data.</div>
<div id="mechcards"></div><div class="card" style="margin:10px 0"><b>Behaviour classes of the bright features</b> <span class="k">(not an identity assignment)</span><table id="cls"></table></div><div class="figs" id="figs6"></div>
<div class="card scroll" style="margin-top:12px"><b>Bright-feature tracks</b> <span class="k">(at least 4 appearances; order: first frame)</span><table id="trk"></table></div>
<div class="card scroll" style="margin-top:12px"><b>Bright features around each break candidate</b><table id="evf"></table></div></div>
<div class="tab" id="t_fix"><div class="card"><div class="warn">Manual mask-correction tool. Click where the picture is wrong: <b>ERASE</b> = remove a non-DNA spot from the mask, <b>ADD</b> = add missed DNA to the mask. These marks become the supervisor's "manually correct" reference and are used to measure accuracy. Choose the frame in the Frames tab above (or with the arrow keys).</div>
<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:8px"><b id="fixlbl"></b>
<label><input type="radio" name="fm" value="erase" checked> ERASE</label><label><input type="radio" name="fm" value="add"> ADD</label>
<label>radius (pixels) <input type="number" id="fr" value="6" min="1" max="60" style="width:70px"></label>
<label><input type="checkbox" id="fmask" checked> show mask</label>
<button class="btn" id="fprev">◀</button><button class="btn" id="fnext">▶</button>
<button class="btn" id="fdl">download corrections.csv</button><button class="btn" id="fclr">clear marks in this frame</button></div>
<div class="row"><canvas id="fc" style="width:100%;max-width:760px;height:auto;cursor:crosshair"></canvas>
<div class="card" style="flex:1;min-width:260px"><b>Marks (all frames: <span id="fcnt">0</span>)</b><div class="scroll" id="flist" style="max-height:50vh"></div>
<p class="k">Save the downloaded file as <code>afm_project/annotations/corrections.csv</code>, then run <code>bash run_all.sh</code>. The mask is corrected accordingly. The browser remembers the marks on this computer.</p></div></div></div></div>
<div class="tab" id="t_h1"><div class="card"><div id="p6btxt"></div></div><div class="card"><b>Track counts with strong and weak detections</b><div class="scroll" id="p6bcmp"></div></div><div class="card"><b>Confidence level of each track</b><div class="scroll" id="p6btrk" style="max-height:50vh"></div></div><div class="figs" id="figs6b"></div></div>
<div class="tab" id="t_h2"><div class="card"><div id="p6ctxt"></div></div><div class="card"><b>Population summary</b><div class="scroll" id="p6cgrp"></div></div><div class="card"><b>Size and brightness of each feature</b><div class="scroll" id="p6cdesc" style="max-height:50vh"></div></div><div class="figs" id="figs6c"></div></div>
<div class="tab" id="t_h3"><div class="card" style="border-left:4px solid #c33"><b>These heights are ASSUMED, not measured.</b><div id="p6dtxt"></div></div><div class="card"><b>Assumed height of the bright features</b><div class="scroll" id="p6dfeat" style="max-height:45vh"></div></div><div class="card"><b>DNA height, frame by frame</b><div class="scroll" id="p6dfr" style="max-height:45vh"></div></div><div class="figs" id="figs6d"></div></div>
<div class="tab" id="t_h4"><div class="card"><div id="p6etxt"></div></div><div class="card"><b>Length frame by frame (corrected pixels; nm at the estimated scale)</b><div class="scroll" id="p6efr" style="max-height:50vh"></div></div><div class="figs" id="figs6e"></div></div>
<div class="tab" id="t_h5"><div class="card"><div id="p6ftxt"></div></div><div class="card"><b>Each break candidate: time and place</b><div class="scroll" id="p6fev" style="max-height:50vh"></div></div><div class="figs" id="figs6f"></div></div>
<div class="tab" id="t_h6"><div class="card"><div id="p6gtxt"></div></div><div class="card"><b>Evidence table</b><div class="scroll" id="p6gev" style="max-height:55vh"></div></div><div class="figs" id="figs6g"></div></div>
<div class="tab" id="t_ov"><div class="warn">We split the supervisor's three requests into six headings. Each row: what was done, what came out, where it is limited. No result <b>confirms</b> that a break happened or what a feature is.</div><div id="ovrows"></div></div>
<div class="tab" id="t_fig"><div class="figs" id="figs"></div></div>
<div class="tab" id="t_rep"><div class="card"><pre id="rep"></pre></div></div>
</main><script>
const D=__DATA__;
const tabs=[["ov","Summary: 6 headings"],["frames","Frames"],["h1","1. Bright-feature tracking"],["h2","2. Dendrimer vs enzyme"],["h3","3. Height profile"],["h4","4. Contour length"],["h5","5. Breaks: when, where"],["h6","6. Mechanism model"],["events","Candidate events"],["know","Library"],["time","Time and place"],["mech","Mechanisms (first version)"],["fix","Correction"],["fig","Figures"],["rep","Report"]];
const nav=document.getElementById("nav");
tabs.forEach(([k,n],i)=>{const b=document.createElement("button");b.textContent=n;b.onclick=()=>show(k);b.id="b_"+k;nav.appendChild(b)});
function show(k){tabs.forEach(([t])=>{document.getElementById("t_"+t).classList.toggle("on",t==k);document.getElementById("b_"+t).classList.toggle("on",t==k)})}
show("ov");
const TR={width_px:"width (pixels)",height_px:"height (pixels)",crop_top:"crop top",crop_bottom:"crop bottom",crop_left:"crop left",crop_right:"crop right",noise_level:"noise level",mask_source:"mask source",n_objects:"number of objects",n_closed_loop_candidates:"closed-loop candidates",n_open_candidates:"open-structure candidates",n_small_objects:"number of small objects",n_loop_if_gaps_bridged:"loops if gaps are bridged",n_scan_line_suspects:"scan-line suspects",mask_area_px:"mask area (pixels)",skeleton_length_px_total:"total skeleton length (pixels)",largest_object_skeleton_px:"longest piece (pixels)",n_bright_features:"number of bright features",nm_per_pixel_used:"nm/pixel (estimate)",scale_status:"scale status",skeleton_length_nm_estimated:"skeleton length (nm, estimated)",time_earliest_s:"earliest time (s)",time_latest_s:"latest time (s)",time_status:"time status",n_objects_tracked:"objects tracked",largest_object_px:"longest piece (pixels)",total_skeleton_px:"total skeleton (pixels)",continuity_fraction:"continuity fraction",mean_bright_area_px:"mean bright area (pixels)",split_candidates_reaching_frame:"break candidates reaching this frame",persistent_split_candidates:"persistent break candidates",largest_object_nm_estimated:"longest piece (nm, estimated)",split_candidates_cumulative:"cumulative break candidates",persistent_splits_cumulative:"cumulative persistent breaks",bright_kept_from_previous:"bright features kept from the previous frame"};
const T=k=>TR[k]||k;
const VT={estimated:"estimated",unknown:"unknown",confirmed:"confirmed",unverified:"unverified",user_verified:"verified by user",supervisor_verified:"verified by supervisor",automatic:"automatic",continuation:"continuation",split_candidate:"break candidate",merge_candidate:"merge candidate",object_lost:"object lost",new_object:"new object",complex_change:"complex change",persistent:"persistent",transient_rejoined:"transient, rejoined",unresolved_end_of_series:"end of series, unresolved",gap_small_possible_artifact:"small gap, possible artefact","candidate only; not a confirmed cleavage":"candidate only; not a confirmed cleavage"};
const V=v=>{v=String(v);return v.split(";").map(x=>VT[x.trim()]||x).join("; ")};
const ET={rank_from:"frame (before)",rank_to:"frame (after)",filename_from:"file (before)",filename_to:"file (after)",event:"event",from_objects:"objects before",to_objects:"objects after",n_pieces_after:"number of pieces after",min_gap_px:"smallest gap (pixels)",confidence:"confidence",registration_shift_px:"registration shift (pixels)",flags:"flags",persistence:"persistence",note:"note",event_y_px:"event y (pixels)",event_x_px:"event x (pixels)"};
const TOPIC={E001:"plasmid size used",E002:"dendrimer (G4 PAMAM, 64 surface amines)",E003:"AFM frame time",E004:"AFM image resolution",E005:"enzyme dose",E006:"naked DNA: onset of degradation",E007:"naked DNA: complete fragmentation",E008:"lifetime (naked / 0.5:1 G4:DNA)",E009:"lifetime (1:1 G4:DNA)",E010:"onset of degradation (0.5:1 and 1:1)",E011:"protection at 5:1",E012:"fragmentation after 2 h incubation",E013:"decorated DNA width / height (0 min)",E014:"undecorated DNA width / height",E015:"relaxed DNA extension range",E016:"length of multi-loop molecule",E017:"share of open-loop path",E018:"pBR322 length",E019:"DNA persistence length (liquid, 3D)",E020:"DNA persistence length (air, 2D)",E021:"persistence length reference (optical tweezers)",E022:"scan area and pixel in liquid",E023:"pBR322 length measured in liquid",E024:"G4 / G5 formula weight",E025:"mean plasmid DNA length (background)",E026:"G4 dendriplex width / height (RNA)",E027:"DNase I, G4 + pDNA, 15 min incubation (previous work)",E028:"DNase I, G4 + pDNA, 2 h incubation (previous work)",E029:"DNA persistence length",E030:"AFM pixel size in that study"};
const F=D.frames;let cur=0,timer=null;
const sl=document.getElementById("sl");sl.max=F.length-1;
function src(f,l){const e=D.img&&D.img[l+"/"+f.stem];if(e)return e;return l=="orig"?D.raw+f.filename:l=="mask"?D.p2+"masks/"+f.stem+"_mask.png":l=="skel"?D.p2+"skeletons/"+f.stem+"_skeleton.png":D.p2+"overlays/"+f.stem+".png"}
function go(i){cur=Math.max(0,Math.min(F.length-1,i));sl.value=cur;const f=F[cur];
document.getElementById("img").src=src(f,document.getElementById("layer").value);
document.getElementById("lbl").textContent="frame "+f.rank+" / "+F.length+" · file "+f.filename;
document.getElementById("meas").innerHTML=Object.entries(f.m).map(([k,v])=>"<tr><td class=k>"+T(k)+"</td><td>"+V(v)+"</td></tr>").join("");draw()}
sl.oninput=()=>go(+sl.value);document.getElementById("layer").onchange=()=>go(cur);
document.getElementById("prev").onclick=()=>go(cur-1);document.getElementById("next").onclick=()=>go(cur+1);
document.getElementById("play").onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent="▶ Play"}else{this.textContent="⏸ Pause";timer=setInterval(()=>{if(cur>=F.length-1)cur=-1;go(cur+1)},350)}};
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
ev.querySelectorAll("tr.click").forEach(r=>r.onclick=()=>{show("frames");go(F.findIndex(f=>f.rank==r.dataset.r))})}else ev.innerHTML="<tr><td>Run run_all.sh first.</td></tr>";
const kn=document.getElementById("kn");function kr(){const q=document.getElementById("q").value.toLowerCase();
kn.innerHTML="<tr><th>topic</th><th>value</th><th>unit</th><th>original quote (unchanged)</th><th>source</th><th>status</th></tr>"+D.know.filter(r=>JSON.stringify(r).toLowerCase().includes(q)).map(r=>"<tr><td>"+(TOPIC[r.evidence_id]||r.topic)+"</td><td>"+r.value+"</td><td>"+r.unit+"</td><td>"+r.quote+(r.context_note?"<br><span class=k>"+r.context_note+"</span>":"")+"</td><td>"+r.src+"</td><td>"+V(r.verified)+"</td></tr>").join("")}
document.getElementById("q").oninput=kr;kr();
document.getElementById("figs").innerHTML=D.figs.map(f=>"<div><div class=k>"+f+"</div><img src='"+((D.figimg&&D.figimg[f])||D.p5+f)+"'></div>").join("")||"<div class=card>No figures, run run_all.sh first.</div>";
document.getElementById("rep").textContent=D.report||"No report, run run_all.sh first.";

let FIX=[];try{FIX=JSON.parse(localStorage.getItem("afm_fix")||"[]")}catch(e){}
function fsave(){try{localStorage.setItem("afm_fix",JSON.stringify(FIX))}catch(e){}}
const fc=document.getElementById("fc"),fg=fc.getContext("2d"),fim=new Image(),fmk=new Image();let fok=false,mok=false;
function fdraw(){const f=F[cur];document.getElementById("fixlbl").textContent="frame "+f.rank+" · "+f.filename;
if(!fok)return;fc.width=fim.naturalWidth;fc.height=fim.naturalHeight;fg.drawImage(fim,0,0);
if(mok&&document.getElementById("fmask").checked){fg.globalAlpha=.4;fg.drawImage(fmk,+f.m.crop_left||0,+f.m.crop_top||0);fg.globalAlpha=1}
FIX.filter(x=>x.filename==f.filename).forEach(x=>{fg.strokeStyle=x.action=="erase"?"#ff2d2d":"#22dd55";fg.lineWidth=2;fg.beginPath();fg.arc(x.x,x.y,x.r,0,7);fg.stroke()});
document.getElementById("fcnt").textContent=FIX.length;
document.getElementById("flist").innerHTML=FIX.map((x,i)=>x.filename==f.filename?"<div>"+(x.action=="erase"?"ERASE":"ADD")+" ("+x.x+", "+x.y+") r="+x.r+" <a href=# data-i="+i+">x</a></div>":"").join("")||"<span class=k>No marks in this frame.</span>";
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

const TT={behaviour_class:"behaviour class",n_tracks:"number of tracks",median_area_px:"median area (pixels)",median_relative_intensity:"median relative brightness",median_lifespan_frames:"median lifespan (frames)",track_id:"track no.",first_rank:"first frame",last_rank:"last frame",n_detections:"number of appearances",lifespan_frames:"lifespan (frames)",missed_frames:"missed frames",motion_class:"movement",max_dist_from_median_px:"largest deviation (pixels)",net_displacement_px:"net displacement (pixels)",s_first_px:"first position s",s_last_px:"last position s",net_drift_along_contour_px:"drift along the DNA (pixels)",mean_area_px:"mean area (pixels)",area_change_px:"area change",mean_relative_intensity:"mean relative brightness",intensity_change:"brightness change",fate:"final state",rank_from:"frame (before)",rank_to:"frame (after)",persistence:"persistence",min_gap_px:"gap (pixels)",event_s_px:"event position s",n_features_in_frame:"features in frame",nearest_feature_px:"nearest feature (pixels)",n_features_near:"nearby features",near_kept_in_place:"stayed in place",near_moved:"moved",near_lost:"disappeared",near_unknown:"unknown"};
const VT2={persistent_stationary:"persistent and stationary",transient_short:"short-lived",intermediate:"intermediate",stationary:"stationary",moved:"moved",short_track_uncertain:"short track, uncertain",wandering_uncertain:"wandering, uncertain",present_to_last_frame:"visible until the last frame"};
function V2(v){v=String(v);if(VT2[v])return VT2[v];if(v.startsWith("not_seen_after_rank_"))return "not seen after frame "+v.replace("not_seen_after_rank_","");return V(v)}
function tbl(el,rows,cols){el.innerHTML="<tr>"+cols.map(k=>"<th>"+(TT[k]||k)+"</th>").join("")+"</tr>"+rows.map(r=>"<tr>"+cols.map(k=>"<td>"+V2(r[k])+"</td>").join("")+"</tr>").join("")}
if(D.p6&&D.p6.mech.length){document.getElementById("mechcards").innerHTML=D.p6.mech.map(m=>"<div class=card style='margin-bottom:10px'><b>"+m.hypothesis+": "+m.hypothesis_name+"</b><p><span class=k>Expected if true:</span> "+m.expected_if_true+"</p><p><span class=k>Observed:</span> "+m.observed+"</p><p class=k>Limitation: "+m.limitation+"</p></div>").join("");
tbl(document.getElementById("trk"),D.p6.tracks.filter(t=>+t.n_detections>=4),["track_id","first_rank","last_rank","n_detections","motion_class","max_dist_from_median_px","net_drift_along_contour_px","mean_area_px","area_change_px","mean_relative_intensity","fate"]);
tbl(document.getElementById("evf"),D.p6.events,["rank_from","rank_to","persistence","min_gap_px","event_s_px","nearest_feature_px","n_features_near","near_kept_in_place","near_moved","near_lost"]);
document.getElementById("figs6").innerHTML=Object.entries(D.p6.figs).map(([n,u])=>"<div><div class=k>"+n+"</div><img src='"+u+"'></div>").join("")}
else document.getElementById("mechcards").innerHTML="<div class=card>No mechanism results. Run run_all.sh in Terminal first.</div>";

const CAP={fig_loss_time_map:"Where and when the break happens: gaps along the DNA contour in the first 22 frames (colour = first gap frame, x = candidate break, number next to it = frame)",fig_contour_coverage_intensity_time:"Left to right: contour overlap and continuity, DNA brightness level, faint skeleton fraction (all JPG intensity, not height)",fig_model_llr_vs_R:"Probability model: break site near / far from a bright feature, relative to the independent model (below 0 = worse than independent)",fig_behaviour_classes:"Behaviour classes of the bright features (size, brightness, lifespan)",fig_feature_kymograph:"Bright features along the DNA over time (dot size = area)",fig_feature_size_brightness_time:"Size and relative brightness of each track over time",fig_feature_size_brightness_groups:"Size-brightness distribution and lifespans of the tracks",fig_changepoint:"When the break happens: change point in the longest-piece and continuity series (red step, light red = 95% interval, dotted line = persistent break candidate), right: change point across 36 threshold settings",fig_event_vs_random:"Distance of break sites and of random DNA points to the nearest bright feature"};
function figs(el,obj){el.innerHTML=Object.entries(obj).map(([n,u])=>"<div><div class=k>"+(CAP[n.replace(".png","")]||n)+"</div><img src='"+u+"'></div>").join("")}
if(D.overview&&D.overview.length){document.getElementById("ovrows").innerHTML=D.overview.map(x=>"<div class=card style='margin-bottom:10px'><b>"+x.no+". "+x.request+"</b> <span class=k>["+x.status+"]</span><p><span class=k>Done:</span> "+x.done+"</p><p><span class=k>Result:</span> "+x.result+"</p><p><span class=k>Limit:</span> "+x.limit+"</p></div>").join("")}
if(D.p6b){document.getElementById("p6btxt").innerHTML=(D.p6b.summary||"").split("\n").filter(x=>x.trim()).map(x=>"<p>"+x.replace(/^[-#]+ ?/,"").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>").join("");
tbl(document.getElementById("p6bcmp"),D.p6b.cmp,["version","n_tracks","tracks_with_1_detection","tracks_with_6plus_detections","median_detections_per_track","total_detections"]);
tbl(document.getElementById("p6btrk"),D.p6b.tracks,["track_id","n_detections","n_strong_detections","n_recovered_detections","motion_class","confidence"]);
figs(document.getElementById("figs6b"),D.p6b.figs)}
if(D.p6c){document.getElementById("p6ctxt").innerHTML=(D.p6c.summary||"").split("\n").filter(x=>x.trim()).map(x=>"<p>"+x.replace(/^[-#]+ ?/,"").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>").join("");
tbl(document.getElementById("p6cgrp"),D.p6c.groups,["population","n_tracks","median_area_px","median_relative_intensity","median_lifespan_frames","n_stationary","n_moved"]);
tbl(document.getElementById("p6cdesc"),D.p6c.desc,["track_id","n_strong","median_area_px","median_relative_intensity","motion_class","lifespan_frames","confidence"]);
figs(document.getElementById("figs6c"),D.p6c.figs)}
if(D.p6d){document.getElementById("p6dtxt").innerHTML=(D.p6d.summary||"").split("\n").filter(x=>x.trim()).map(x=>"<p>"+x.replace(/^[-#]+ ?/,"").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>").join("");
tbl(document.getElementById("p6dfeat"),D.p6d.feat,["track_id","n_strong","median_peak_nm","min_peak_nm","max_peak_nm","median_above_ridge_nm","motion_class","confidence","height_status"]);
tbl(document.getElementById("p6dfr"),D.p6d.frames,["rank","ridge_height_nm","mean_contour_height_nm","contour_covered_fraction","aligned_valid","height_status"]);
figs(document.getElementById("figs6d"),D.p6d.figs)}
if(D.p6e){document.getElementById("p6etxt").innerHTML=(D.p6e.summary||"").split("\n").filter(x=>x.trim()).map(x=>"<p>"+x.replace(/^[-#]+ ?/,"").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>").join("");
tbl(document.getElementById("p6efr"),D.p6e.frames,["rank","n_pieces","total_corrected_px","largest_corrected_px","total_nm_estimated_scale","largest_nm_estimated_scale","length_below_min_not_counted_px"]);
figs(document.getElementById("figs6e"),D.p6e.figs)}
if(D.p6f){document.getElementById("p6ftxt").innerHTML=(D.p6f.summary||"").split("\n").filter(x=>x.trim()).map(x=>"<p>"+x.replace(/^[-#]+ ?/,"").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>").join("");
tbl(document.getElementById("p6fev"),D.p6f.events,["rank_from","rank_to","persistence","n_new_pieces","old_length_px","new_lengths_px","cut_fraction_shortest_piece","conserved","contour_position_reliable","relative_to_change_point"]);
figs(document.getElementById("figs6f"),D.p6f.figs)}
if(D.p6g){document.getElementById("p6gtxt").innerHTML=(D.p6g.summary||"").split("\n").filter(x=>x.trim()).map(x=>"<p>"+x.replace(/^[-#]+ ?/,"").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>").join("");
tbl(document.getElementById("p6gev"),D.p6g.evidence,["hypothesis","evidence","numbers","reading","direction"]);
figs(document.getElementById("figs6g"),D.p6g.figs)}
if(D.p7&&D.p7.figs){figs(document.getElementById("figs7a"),Object.fromEntries(Object.entries(D.p7.figs).filter(([n])=>/loss_time|coverage/.test(n))));
document.getElementById("p7txt").innerHTML=(D.p7.summary||"").split("\n").filter(x=>x.trim()).map(x=>"<p>"+x.replace(/^[-#]+ ?/,"").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>").join("");
tbl(document.getElementById("cls"),D.p7.classes,["behaviour_class","n_tracks","median_area_px","median_relative_intensity","median_lifespan_frames"]);
if(D.p7.mech.length){document.getElementById("mechcards").innerHTML=D.p7.mech.map(m=>"<div class=card style='margin-bottom:10px'><b>"+m.hypothesis+": "+m.hypothesis_name+"</b><p><span class=k>Model numbers:</span> "+m.model_numbers+"</p><p><span class=k>Reading:</span> "+m.reading+"</p></div>").join("")}
figs(document.getElementById("figs6"),Object.fromEntries(Object.entries({...D.p6.figs,...D.p7.figs}).filter(([n])=>!/loss_time|coverage/.test(n))))}

if(D.p8&&D.p8.summary){document.getElementById("p8txt").innerHTML=D.p8.summary.split("\n").filter(x=>x.trim()).map(x=>"<p>"+x.replace(/^[-#]+ ?/,"").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")+"</p>").join("");
figs(document.getElementById("figs8"),D.p8.figs);
const LT={source:"source",what:"what",value:"value",quote:"original quote / note",verified:"status"};
const lt=document.getElementById("lit");const lc=["source","what","value","quote","verified"];
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
          "summary": (p7d / "summary.md").read_text(encoding="utf-8") if (p7d / "summary.md").exists() else "",
          "figs": {q.name: _data_uri(q) for q in sorted(p7d.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p7d.exists() else {}}
    p6bd = out / "phase6b"
    p6b = {"summary": (p6bd / "summary.md").read_text(encoding="utf-8") if (p6bd / "summary.md").exists() else "",
           "cmp": _csv(p6bd / "track_comparison.csv"), "tracks": _csv(p6bd / "feature_tracks_v2.csv"),
           "figs": {q.name: _data_uri(q) for q in sorted(p6bd.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p6bd.exists() else {}}
    p6cd = out / "phase6c"
    p6c = {"summary": (p6cd / "summary.md").read_text(encoding="utf-8") if (p6cd / "summary.md").exists() else "",
           "groups": _csv(p6cd / "population_summary.csv"), "desc": _csv(p6cd / "feature_descriptors.csv"),
           "figs": {q.name: _data_uri(q) for q in sorted(p6cd.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p6cd.exists() else {}}
    p6dd = out / "heights"
    p6d = {"summary": (p6dd / "summary.md").read_text(encoding="utf-8") if (p6dd / "summary.md").exists() else "",
           "feat": _csv(p6dd / "feature_heights.csv"), "frames": _csv(p6dd / "dna_height_by_frame.csv"),
           "figs": {q.name: _data_uri(q) for q in sorted(p6dd.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p6dd.exists() else {}}
    p6ed = out / "lengths"
    p6e = {"summary": (p6ed / "summary.md").read_text(encoding="utf-8") if (p6ed / "summary.md").exists() else "",
           "frames": _csv(p6ed / "length_by_frame.csv"),
           "figs": {q.name: _data_uri(q) for q in sorted(p6ed.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p6ed.exists() else {}}
    p6fd = out / "where_when"
    p6f = {"summary": (p6fd / "summary.md").read_text(encoding="utf-8") if (p6fd / "summary.md").exists() else "",
           "events": _csv(p6fd / "split_candidates_when_where.csv"),
           "figs": {q.name: _data_uri(q) for q in sorted(p6fd.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p6fd.exists() else {}}
    p6gd = out / "mechanisms"
    p6g = {"summary": (p6gd / "summary.md").read_text(encoding="utf-8") if (p6gd / "summary.md").exists() else "",
           "evidence": _csv(p6gd / "hypothesis_evidence.csv"),
           "figs": {q.name: _data_uri(q) for q in sorted(p6gd.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p6gd.exists() else {}}
    ovp = out / "supervisor" / "overview_rows.json"
    overview = json.loads(ovp.read_text(encoding="utf-8")) if ovp.is_file() else []
    p8d = out / "phase8"
    p8 = {"summary": (p8d / "summary.md").read_text(encoding="utf-8") if (p8d / "summary.md").exists() else "",
          "lit": _csv(p8d / "literature_comparison.csv"),
          "figs": {q.name: _data_uri(q) for q in sorted(p8d.glob("fig_*.png")) if q.stem in KNOWN_FIGS} if p8d.exists() else {}}
    figimg = {n: _data_uri(out / "phase5" / n) for n in figs}
    data = {"overview": overview, "p6g": p6g, "p6f": p6f, "p6e": p6e, "p6d": p6d, "p6c": p6c, "p6b": p6b, "p8": p8, "p7": p7, "p6": p6, "img": img, "figimg": figimg, "frames": frames, "metrics": metrics, "events": _csv(out / "phase3" / "events.csv"),
            "know": know, "figs": figs, "report": rep.read_text(encoding="utf-8") if rep.exists() else "",
            "raw": "../../data/raw/", "p2": "../phase2/", "p5": "../phase5/"}
    dest = out / "viewer"
    dest.mkdir(parents=True, exist_ok=True)
    html = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (dest / "index.html").write_text(html, encoding="utf-8")
    return dest / "index.html"
