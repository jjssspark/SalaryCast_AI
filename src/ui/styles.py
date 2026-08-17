"""StoveLens AI — 화면 스타일.

호출 위치: app/app.py에서 CSS를 한 번 주입한다.
데이터 파일 없음.

승인된 목업 output/mockups/design_v3.html의 스타일을 그대로 옮기고,
Streamlit 기본 크롬(헤더·여백·입력창)을 덮는 규칙을 앞에 덧붙였다.
색·간격·움직임 값은 목업과 같다.
"""

CSS = """
<style>
:root{
  --ink:#050A07;
  --turf:#08110C;
  --panel:#0C1610;
  --panel-2:#132119;
  --line:rgba(255,255,255,.08);
  --line-2:rgba(255,255,255,.16);
  --chalk:#F2F6F3;
  --dim:#8C9B92;
  --dimmer:#54655B;
  --red:#FF4438;
  --led:#FFD34D;
  --dirt:#B07A4A;
  --sky:#7FD4F5;
  --mint:#5FE3A1;
  --sans:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Pretendard","Malgun Gothic",sans-serif;
  --num:"Helvetica Neue",Helvetica,Arial,sans-serif;
  --ease:cubic-bezier(.16,1,.3,1);
  --tc:#5D7466;
}

/* ─── Streamlit 크롬 정리 ─── */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer{display:none!important}

.stApp{background:var(--ink);color:var(--chalk);font-family:var(--sans);-webkit-font-smoothing:antialiased}
.stApp::before{
  content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(700px 360px at 12% -8%, rgba(255,211,77,.11), transparent 62%),
    radial-gradient(700px 360px at 88% -8%, rgba(255,211,77,.09), transparent 62%),
    repeating-linear-gradient(74deg, rgba(255,255,255,.026) 0 58px, rgba(0,0,0,0) 58px 116px),
    var(--turf);
  animation:breathe 9s ease-in-out infinite;
}
@keyframes breathe{0%,100%{opacity:1}50%{opacity:.88}}

[data-testid="stMainBlockContainer"],
.block-container{position:relative;z-index:1;max-width:1160px;padding:18px 24px 120px}
[data-testid="stVerticalBlock"]{gap:0}

/* 검색 입력 — 목업 .hsearch */
.stTextInput label{display:none}
.stTextInput input{
  width:100%;font-family:var(--sans);font-size:1.05rem!important;font-weight:700;
  color:var(--chalk)!important;background:rgba(0,0,0,.5)!important;
  border:1px solid var(--line-2)!important;border-radius:3px!important;
  padding:18px!important;height:auto!important;
  transition:border-color .2s,box-shadow .2s;
}
.stTextInput input::placeholder{color:var(--dimmer)!important;font-weight:600}
.stTextInput input:focus{border-color:var(--led)!important;box-shadow:0 0 0 4px rgba(255,211,77,.14)!important}
.stTextInput [data-baseweb="input"]{background:transparent!important;border:none!important}

/* 접이식 상세 */
[data-testid="stExpander"]{
  border:1px solid var(--line)!important;border-radius:4px!important;
  background:var(--panel)!important;margin-top:14px;
}
[data-testid="stExpander"] summary{font-size:.8rem;font-weight:700;color:var(--dim)}
[data-testid="stExpander"] summary:hover{color:var(--chalk)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:4px}

/* ─── 헤더 ─── */
.appbar{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:14px 0 6px}
.brand{display:flex;align-items:center;gap:11px;text-decoration:none;color:var(--chalk)}
.brand .logoball{width:32px;height:32px;flex-shrink:0;animation:spin 8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.brand b{font-size:1.05rem;font-weight:900;letter-spacing:-.02em}
.brand b i{color:var(--red);font-style:normal}
.backlink{font-size:.74rem;font-weight:800;letter-spacing:.08em;color:var(--dim);
  text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:7px 15px;transition:all .18s}
.backlink:hover{color:var(--chalk);border-color:var(--line-2)}

/* ─── 히어로 ─── */
.hero{position:relative;padding:56px 0 34px}
.hero .field{position:absolute;right:-90px;top:-70px;width:560px;height:560px;pointer-events:none}
.hero .field .ln{stroke-dasharray:1600;stroke-dashoffset:1600;animation:draw 2.6s var(--ease) forwards}
.hero .field .ln.d1{animation-delay:.45s}
.hero .field .ln.d2{animation-delay:.8s}
.hero .field .ln.d3{animation-delay:1.2s}
@keyframes draw{to{stroke-dashoffset:0}}
.hero .field .late{opacity:0;animation:fadeIn .6s ease forwards 1.7s}
.hero .field .dirt{opacity:0;animation:fadeIn 1.1s ease forwards .3s}
@keyframes fadeIn{to{opacity:1}}

.arc{position:absolute;left:0;top:60%;width:100%;height:0;pointer-events:none;z-index:2}
.arc .fly{position:absolute;left:-40px;top:0;animation:flyX 6.4s cubic-bezier(.34,0,.72,1) infinite}
.arc .bob{display:block;animation:flyY 6.4s cubic-bezier(.2,.85,.5,1) infinite}
.arc .rot{display:block;width:22px;height:22px;animation:tumble 1.1s linear infinite;
  filter:drop-shadow(0 0 10px rgba(255,255,255,.45))}
@keyframes flyX{0%{transform:translateX(0);opacity:0}5%{opacity:1}88%{opacity:1}100%{transform:translateX(1160px);opacity:0}}
@keyframes flyY{0%{transform:translateY(0)}48%{transform:translateY(-190px)}100%{transform:translateY(80px)}}
@keyframes tumble{to{transform:rotate(360deg)}}

.kicker{position:relative;z-index:3;display:flex;align-items:center;gap:11px;
  font-size:.66rem;font-weight:800;letter-spacing:.3em;color:var(--led);
  margin-bottom:26px;animation:rise .8s var(--ease) both}
.lamp{width:8px;height:8px;border-radius:50%;background:var(--red);
  box-shadow:0 0 10px var(--red);animation:blink 1.9s ease-in-out infinite;flex-shrink:0}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}

.hero h1{position:relative;z-index:3;font-size:clamp(2.8rem,8vw,5.6rem);
  font-weight:900;line-height:.88;letter-spacing:-.06em;margin:0}
.hero h1 span{display:block;overflow:hidden;padding-bottom:.05em}
.hero h1 span i{display:block;font-style:normal;animation:swing .95s var(--ease) both}
.hero h1 span:nth-child(1) i{animation-delay:.15s}
.hero h1 span:nth-child(2) i{animation-delay:.32s}
.hero h1 em{font-style:normal;color:var(--red)}
@keyframes swing{from{transform:translateY(105%) rotate(4deg)}to{transform:translateY(0) rotate(0)}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}

.searchhint{font-size:.74rem;color:var(--dimmer);margin-top:10px}
.searchhint b{color:var(--led)}

/* ─── 검색 결과 ─── */
.results{display:flex;flex-direction:column;gap:6px;margin-top:14px}
.rline{display:flex;align-items:center;gap:14px;padding:13px 18px;border-radius:4px;
  background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--tc);
  text-decoration:none;color:var(--chalk);transition:transform .35s var(--ease),background .18s}
.rline:hover{transform:translateX(5px);background:var(--panel-2)}
.rline .nm{font-weight:800;font-size:.95rem}
.rline .meta{font-size:.75rem;color:var(--dim)}
.rline .go{margin-left:auto;font-size:.72rem;color:var(--led);font-weight:800}
.empty{padding:18px 2px;font-size:.82rem;color:var(--dim)}

/* ─── 티커 ─── */
.ticker{position:relative;margin:34px 0 0;display:flex;align-items:stretch;
  border-top:1px solid var(--line-2);border-bottom:1px solid var(--line-2);
  background:#040A06;overflow:hidden;border-radius:2px}
.ticker .lbl{display:flex;align-items:center;gap:9px;padding:0 22px;flex-shrink:0;z-index:4;
  background:var(--red);color:#fff;font-size:.66rem;font-weight:900;letter-spacing:.2em}
.ticker .win{position:relative;flex:1;overflow:hidden}
.ticker .win::before{content:"";position:absolute;inset:0;pointer-events:none;z-index:2;
  background-image:radial-gradient(rgba(255,255,255,.08) .9px,transparent .9px);background-size:4px 4px}
.ticker .win::after{content:"";position:absolute;inset:0;pointer-events:none;z-index:3;
  background:linear-gradient(90deg,transparent 88%,#040A06 100%)}
.ticker .rail{display:flex;width:max-content;animation:roll 36s linear infinite}
.ticker:hover .rail{animation-play-state:paused}
@keyframes roll{to{transform:translateX(-50%)}}
.ticker .grp{display:flex;align-items:center;padding:15px 0}
.ticker .it{display:flex;align-items:center;gap:11px;padding:0 26px;white-space:nowrap;
  text-decoration:none;color:var(--chalk)}
.ticker .it:hover .nn{color:var(--led)}
.ticker .yy{font-family:var(--num);font-size:.8rem;font-weight:800;color:var(--led);font-variant-numeric:tabular-nums}
.ticker .nn{font-size:.92rem;font-weight:800;transition:color .18s}
.ticker .tt{font-size:.75rem;color:var(--dim)}
.plate{width:9px;height:9px;background:var(--dirt);flex-shrink:0;
  clip-path:polygon(0 0,100% 0,100% 55%,50% 100%,0 55%)}

/* ─── 섹션 제목 ─── */
.sect{display:flex;align-items:center;gap:14px;margin:60px 0 20px}
.sect .plate{width:15px;height:15px;background:var(--chalk)}
.sect h2{font-size:1.05rem;font-weight:900;letter-spacing:-.02em;flex-shrink:0;margin:0}
.sect .note{font-size:.72rem;color:var(--dimmer);flex-shrink:0}
.stitch{flex:1;height:12px;opacity:.65;margin-left:4px}

/* ─── 벤토 ─── */
.bento{display:grid;grid-template-columns:1.7fr 1fr 1fr;grid-auto-rows:150px;gap:12px}
.tile{position:relative;overflow:hidden;border-radius:5px;display:block;text-decoration:none;
  color:var(--chalk);background:var(--panel);border:1px solid var(--line);
  transition:transform .45s var(--ease),border-color .2s;animation:tileIn .85s var(--ease) both}
.tile:nth-child(1){animation-delay:.05s}
.tile:nth-child(2){animation-delay:.14s}
.tile:nth-child(3){animation-delay:.23s}
.tile:nth-child(4){animation-delay:.32s}
.tile:nth-child(5){animation-delay:.41s}
@keyframes tileIn{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:none}}
.tile:hover{transform:translateY(-6px);border-color:var(--line-2)}
.tile::before{content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(145deg,var(--tc) -30%,transparent 58%);opacity:.34;transition:opacity .3s}
.tile:hover::before{opacity:.58}

.tile.big{grid-row:span 2;padding:28px}
.tile.big .yy{position:relative;font-family:var(--num);font-size:.82rem;font-weight:800;color:var(--led)}
.tile.big .nn{position:relative;font-size:2.7rem;font-weight:900;letter-spacing:-.05em;line-height:1;margin-top:12px}
.tile.big .tt{position:relative;font-size:.84rem;color:var(--dim);margin-top:10px}
.tile.big .pred{position:absolute;left:28px;bottom:28px;right:28px}
.tile.big .pred .k{font-size:.6rem;letter-spacing:.2em;color:var(--dimmer);font-weight:800}
.tile.big .pred .v{font-family:var(--num);font-size:2.4rem;font-weight:800;letter-spacing:-.05em;
  color:#fff;margin-top:4px;font-variant-numeric:tabular-nums}
.tile.big .pred .v small{font-family:var(--sans);font-size:.9rem;color:var(--dim);
  font-weight:800;margin-left:6px;letter-spacing:0}
.tile.big .pred .ref{font-size:.72rem;color:var(--dim);margin-top:8px}
.tile.big .pred .ref b{color:var(--chalk);font-weight:800}
.tile.big .bat{position:absolute;right:-26px;bottom:-14px;width:230px;opacity:.08;
  pointer-events:none;transform:rotate(-14deg)}

.tile.md{padding:20px}
.tile.md .yy{position:relative;font-family:var(--num);font-size:.76rem;font-weight:800;color:var(--led)}
.tile.md .nn{position:relative;font-size:1.45rem;font-weight:900;letter-spacing:-.03em;margin-top:8px}
.tile.md .tt{position:relative;font-size:.74rem;color:var(--dim);margin-top:6px}
.tile.md .go{position:absolute;right:18px;bottom:16px;font-size:.72rem;color:var(--led);
  font-weight:800;opacity:0;transform:translateX(-8px);transition:all .3s var(--ease)}
.tile:hover .go{opacity:1;transform:none}

/* ─── 전광판 스트립 ─── */
.board{display:flex;margin-top:34px;position:relative;border:1px solid var(--line-2);
  border-radius:3px;background:#040A06;box-shadow:inset 0 0 40px rgba(0,0,0,.9)}
.board::before{content:"";position:absolute;inset:0;pointer-events:none;border-radius:3px;
  background-image:radial-gradient(rgba(255,255,255,.07) .8px,transparent .8px);background-size:4px 4px}
.board .cell{flex:1;padding:22px 24px;border-right:1px solid var(--line);position:relative}
.board .cell:last-child{border-right:none}
.board .cl{font-size:.6rem;letter-spacing:.2em;color:var(--dimmer);font-weight:800}
.board .cv{font-family:var(--num);font-size:2.1rem;font-weight:800;letter-spacing:-.04em;
  color:var(--led);margin-top:8px;font-variant-numeric:tabular-nums;text-shadow:0 0 26px rgba(255,211,77,.55)}

/* ─── 선수 히어로 ─── */
.phero{position:relative;overflow:hidden;border-radius:6px;margin-top:6px;
  border:1px solid var(--line);background:var(--panel);display:flex;align-items:center;
  min-height:240px;animation:wipe 1s var(--ease) both}
@keyframes wipe{from{clip-path:inset(0 100% 0 0)}to{clip-path:inset(0 0 0 0)}}
.phero::before{content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(100deg,var(--tc) -14%,transparent 48%);opacity:.32}
.phero .wm{position:absolute;right:26px;bottom:-30px;font-family:var(--num);font-size:11rem;
  font-weight:800;letter-spacing:-.07em;color:rgba(255,255,255,.035);line-height:1;pointer-events:none}
.phero .slab{width:196px;flex-shrink:0;align-self:stretch;background:rgba(255,255,255,.05);
  clip-path:polygon(0 0,100% 0,82% 100%,0 100%);display:flex;align-items:center;justify-content:center}
.phero .ph{width:124px;height:124px;border-radius:50%;margin-right:18px;overflow:hidden;
  background:linear-gradient(150deg,#1D2E24,#0A1310);border:2px solid rgba(255,255,255,.14);
  display:flex;align-items:center;justify-content:center;font-size:2.6rem;font-weight:900;
  color:rgba(255,255,255,.4)}
.phero .ph img{width:100%;height:100%;object-fit:cover;display:block}
.phero .body{position:relative;flex:1;padding:32px 38px}
.chips{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.chip{font-size:.66rem;font-weight:800;letter-spacing:.05em;padding:5px 10px;
  border-radius:2px;background:rgba(255,255,255,.09)}
.chip.team{background:var(--tc);color:#fff}
.chip.hot{background:var(--red);color:#fff}
.chip.soon{background:rgba(255,211,77,.2);color:var(--led)}
.phero h2{font-size:clamp(2.4rem,5.5vw,3.6rem);font-weight:900;letter-spacing:-.055em;line-height:1;margin:0}
.phero .sub{color:var(--dim);font-size:.82rem;margin-top:14px}
.stars{color:var(--led);letter-spacing:.14em}

/* ─── 패널 ─── */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;align-items:start}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:30px 28px}
.pane-t{display:flex;align-items:baseline;gap:10px;margin-bottom:24px;flex-wrap:wrap}
.pane-t b{font-size:.62rem;font-weight:800;letter-spacing:.22em;color:var(--dimmer)}
.pane-t s{text-decoration:none;font-size:.68rem;color:var(--dimmer);opacity:.85}

.money{display:flex;align-items:flex-end;gap:10px;line-height:.82}
.money .n{font-family:var(--num);font-size:clamp(3.4rem,8vw,5rem);font-weight:800;
  letter-spacing:-.06em;font-variant-numeric:tabular-nums;color:#fff;text-shadow:0 0 50px rgba(255,255,255,.22)}
.money .u{font-size:1.15rem;font-weight:800;color:var(--dim);padding-bottom:11px}
.money-ref{margin-top:16px;font-size:.8rem;color:var(--dim);line-height:1.65}
.money-ref b{color:var(--chalk);font-weight:800}

.gauge{margin-top:30px}
.gauge .lbl{font-size:.62rem;color:var(--dimmer);letter-spacing:.22em;font-weight:800;margin-bottom:14px}
.gauge .track{position:relative;height:6px;border-radius:3px;background:rgba(255,255,255,.07)}
.gauge .band{position:absolute;top:0;bottom:0;border-radius:3px;
  background:linear-gradient(90deg,rgba(255,211,77,.25),rgba(255,211,77,.7),rgba(255,211,77,.25));
  animation:grow 1.1s var(--ease) both .3s;transform-origin:left}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.gauge .dot{position:absolute;top:50%;width:14px;height:14px;border-radius:50%;background:#fff;
  border:3px solid var(--led);transform:translate(-50%,-50%);box-shadow:0 0 20px rgba(255,211,77,.8);
  animation:pop .5s var(--ease) both .9s}
@keyframes pop{from{transform:translate(-50%,-50%) scale(0)}to{transform:translate(-50%,-50%) scale(1)}}
.gauge .ends{display:flex;justify-content:space-between;margin-top:12px;font-size:.75rem;
  color:var(--dim);font-family:var(--num);font-weight:700}

/* ─── 전광판 스탯 ─── */
.sboard{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line-2);border:2px solid var(--line-2)}
.sb{position:relative;background:#040A06;padding:20px 18px 18px;transition:background .18s}
.sb::before{content:"";position:absolute;inset:0;pointer-events:none;
  background-image:radial-gradient(rgba(255,255,255,.06) .8px,transparent .8px);background-size:4px 4px}
.sb:hover{background:#0A150F}
.sb>*{position:relative}
.sb .k{display:flex;align-items:center;font-size:.6rem;letter-spacing:.2em;color:var(--dimmer);font-weight:800}
.sb .v{font-family:var(--num);font-size:1.9rem;font-weight:800;letter-spacing:-.045em;color:var(--led);
  margin-top:8px;font-variant-numeric:tabular-nums;text-shadow:0 0 24px rgba(255,211,77,.45)}

.scale{position:relative;height:5px;border-radius:3px;background:rgba(255,255,255,.09);margin-top:18px}
.scale i{display:block;height:100%;border-radius:3px;background:var(--led);opacity:.8;
  animation:grow .95s var(--ease) both;transform-origin:left}
.scale .avg{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--chalk);opacity:.8}
.scale .avg::after{content:"리그 평균";position:absolute;top:-15px;left:50%;transform:translateX(-50%);
  font-size:.53rem;font-weight:800;color:var(--dim);letter-spacing:.04em;white-space:nowrap}
.cmp{display:flex;justify-content:space-between;align-items:baseline;margin-top:11px;
  font-size:.68rem;color:var(--dimmer);gap:8px}
.cmp .rank{color:var(--led);font-weight:800;white-space:nowrap}

/* 툴팁 */
.tip{position:relative;display:inline-flex;align-items:center;justify-content:center;
  width:14px;height:14px;margin-left:6px;border-radius:50%;flex-shrink:0;
  border:1px solid var(--dimmer);color:var(--dimmer);font-size:.58rem;font-weight:800;
  cursor:help;letter-spacing:0;transition:border-color .18s,color .18s}
.tip:hover{border-color:var(--led);color:var(--led)}
.tip::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 9px);left:50%;
  transform:translateX(-50%);width:236px;padding:11px 13px;background:#0A1512;
  border:1px solid var(--line-2);border-radius:4px;font-size:.72rem;font-weight:600;
  color:var(--chalk);line-height:1.55;text-align:left;letter-spacing:0;opacity:0;
  pointer-events:none;transition:opacity .18s;z-index:70;box-shadow:0 16px 40px rgba(0,0,0,.7)}
.tip:hover::after{opacity:1}

/* ─── 예측 vs 실제 ─── */
.versus{display:grid;grid-template-columns:1fr auto 1fr;gap:20px;align-items:center}
.vs-box{text-align:center}
.vs-box .k{font-size:.62rem;letter-spacing:.18em;color:var(--dimmer);font-weight:800}
.vs-box .v{font-family:var(--num);font-size:2.7rem;font-weight:800;letter-spacing:-.05em;
  margin-top:10px;font-variant-numeric:tabular-nums}
.vs-box.ai .v{color:var(--sky)}
.vs-box.real .v{color:var(--mint)}
.vs-mid{width:26px;height:26px;background:var(--dirt);opacity:.55;
  clip-path:polygon(0 0,100% 0,100% 55%,50% 100%,0 55%)}
.verdict{margin-top:28px;text-align:center}
.verdict .big{font-size:1.5rem;font-weight:900;letter-spacing:-.03em;color:var(--mint)}
.verdict.over .big{color:var(--red)}
.verdict.under .big{color:var(--sky)}
.verdict .txt{font-size:.8rem;color:var(--dim);margin-top:8px;line-height:1.65}
.verdict .txt b{color:var(--chalk);font-weight:800}

/* ─── 핵심 요소 ─── */
.factors{display:flex;flex-direction:column;gap:28px}
.f .fhead{display:flex;align-items:center;gap:12px}
.f .base{width:11px;height:11px;background:var(--dirt);transform:rotate(45deg);flex-shrink:0}
.f .fnm{display:flex;align-items:center;font-size:.95rem;font-weight:800}
.f .fval{margin-left:auto;font-family:var(--num);font-size:1.3rem;font-weight:800;
  letter-spacing:-.03em;font-variant-numeric:tabular-nums}
.f .fval small{font-family:var(--sans);font-size:.72rem;color:var(--dim);font-weight:700;
  margin-left:5px;letter-spacing:0}
.f .scale{margin-top:16px}
.f .scale i{background:linear-gradient(90deg,var(--red),#FFA07A);opacity:1}
.f .fsub{display:flex;justify-content:space-between;margin-top:10px;font-size:.72rem;
  color:var(--dimmer);gap:10px}
.f .fsub b{color:var(--dim);font-weight:800}
.f.down .scale i{background:linear-gradient(90deg,#4E86DE,#8FB8F0)}
.f.down .fval{color:#8FB8F0}

/* ─── 구단별 제시가 ─── */
.offers{display:flex;flex-direction:column;gap:8px}
.of{display:flex;align-items:center;gap:18px;padding:15px 20px;border-radius:4px;
  background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--tc);
  transition:transform .4s var(--ease),background .18s}
.of:hover{transform:translateX(5px);background:var(--panel-2)}
.of .tm{font-weight:800;font-size:.92rem;width:52px;flex-shrink:0;color:var(--tc)}
.of .track{flex:1;height:10px;border-radius:5px;background:rgba(255,255,255,.05);overflow:hidden}
.of .track i{display:block;height:100%;border-radius:5px;background:var(--tc);
  animation:grow 1.05s var(--ease) both;transform-origin:left}
.of .amt{font-family:var(--num);font-size:.92rem;font-weight:800;width:120px;text-align:right;
  flex-shrink:0;font-variant-numeric:tabular-nums}
.of .need{font-size:.66rem;color:var(--dimmer);width:78px;flex-shrink:0;text-align:right}

/* ─── 지표 설명 ─── */
.glossary{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line)}
.gl{background:var(--panel);padding:17px 20px}
.gl dt{font-size:.86rem;font-weight:800;color:var(--chalk);margin-bottom:6px}
.gl dt em{font-style:normal;color:var(--led);font-family:var(--num);font-size:.78rem;margin-left:7px}
.gl dd{font-size:.78rem;color:var(--dim);line-height:1.65;margin:0}

/* ─── 시즌 기록 표 ─── */
.stable{margin-top:12px;overflow-x:auto}
.stable table{width:100%;border-collapse:collapse;font-family:var(--num);font-variant-numeric:tabular-nums}
.stable th{font-family:var(--sans);font-size:.62rem;letter-spacing:.12em;color:var(--dimmer);
  font-weight:800;text-align:right;padding:0 0 12px;white-space:nowrap}
.stable th:first-child,.stable td:first-child{text-align:left}
.stable th:nth-child(2),.stable td:nth-child(2){text-align:left}
.stable td{font-size:.84rem;font-weight:700;color:var(--chalk);text-align:right;
  padding:11px 0;border-top:1px solid var(--line);white-space:nowrap}
.stable tbody tr:first-child td{color:var(--led)}

.basis{margin-top:16px;font-size:.73rem;color:var(--dimmer);line-height:1.75}
.basis b{color:var(--dim)}

.notice{margin-top:14px;padding:16px 18px;border-radius:4px;background:var(--panel);
  border:1px solid var(--line);border-left:3px solid var(--led);
  font-size:.82rem;color:var(--dim);line-height:1.7}
.notice b{color:var(--chalk);font-weight:800}

/* 스플래시 */
.splash{display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:66vh;text-align:center}
.splash .logoball{width:64px;height:64px;animation:spin 1.5s linear infinite}
.splash .nm{font-size:2.2rem;font-weight:900;color:#fff;margin:20px 0 6px;letter-spacing:-1px}
.splash .nm i{color:var(--red);font-style:normal}
.splash .msg{font-size:.9rem;color:var(--dim)}

@media (max-width:900px){
  .bento{grid-template-columns:1fr;grid-auto-rows:auto}
  .tile.big{grid-row:auto;min-height:320px}
  .tile.md{min-height:112px}
  .hero .field{opacity:.25;right:-260px}
  .hero{padding:34px 0 24px}
  .ticker .lbl{display:none}
  .cols{grid-template-columns:1fr}
  .glossary{grid-template-columns:1fr}
  .phero{flex-direction:column;min-height:0}
  .phero .slab{width:100%;clip-path:none;padding:26px 0}
  .phero .ph{margin:0}
  .phero .wm{display:none}
  .phero .body{padding:26px 22px 32px}
  .board{flex-wrap:wrap}
  .board .cell{min-width:50%;border-bottom:1px solid var(--line)}
  .versus{grid-template-columns:1fr;gap:20px}
  .vs-mid{display:none}
  .tip::after{width:190px}
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation:none!important;transition:none!important}
  .phero{clip-path:none}
  .hero .field .ln{stroke-dashoffset:0}
  .hero .field .late,.hero .field .dirt{opacity:1}
  .arc{display:none}
}
</style>
"""
