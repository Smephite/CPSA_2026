"""The node's web page: live dashboard stream plus the tuning panel (served by sinks.MjpegSink)."""

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Guardian Node</title>
<style>
  :root { --bg:#18181a; --panel:#232326; --line:#34343a; --text:#e6e6e6; --muted:#9a9aa2; --accent:#ffb000;
          --err:#ff6b6b; --ok:#5ad17a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif; }
  .wrap { display:flex; flex-wrap:wrap; gap:16px; padding:16px; }
  .stream { flex:1 1 640px; min-width:0; }
  .stream img { width:100%; max-width:1280px; display:block; border:1px solid var(--line); }
  .tune { flex:1 1 380px; max-width:560px; }
  .bar { display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
  .bar input[type=search] { flex:1; min-width:120px; }
  #status { color:var(--muted); font-size:12px; min-height:1.4em; }
  details { background:var(--panel); border:1px solid var(--line); border-radius:6px; margin-bottom:8px; }
  summary { cursor:pointer; padding:8px 10px; font-weight:600; }
  .row { display:grid; grid-template-columns: minmax(0,1fr) auto; gap:4px 8px; padding:6px 10px;
         border-top:1px solid var(--line); }
  .row.changed .name { color:var(--accent); }
  .row.err .name { color:var(--err); }
  .name { font-family:ui-monospace, monospace; font-size:12px; overflow-wrap:anywhere; position:relative;
          cursor:help; text-decoration:underline dotted var(--muted); text-underline-offset:3px; }
  .name .tip { display:none; position:absolute; left:0; top:calc(100% + 4px); z-index:10; width:min(360px, 80vw);
               background:#0e0e10; color:var(--text); border:1px solid var(--muted); border-radius:4px;
               padding:6px 8px; font:12px/1.4 system-ui, sans-serif; }
  .name:hover .tip, .name:focus .tip { display:block; }
  .help { grid-column:1 / -1; color:var(--muted); font-size:12px; }
  .ctl { grid-column:1 / -1; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .ctl input[type=range] { flex:1; min-width:120px; }
  .ctl input[type=number] { width:90px; }
  input, select, button { background:#111114; color:var(--text); border:1px solid var(--line); border-radius:4px;
                          padding:4px 6px; font:inherit; }
  button { cursor:pointer; } button:hover { border-color:var(--muted); }
  .def { color:var(--muted); font-size:12px; }
  .msg { grid-column:1 / -1; color:var(--err); font-size:12px; }
  label.chk { display:inline-flex; gap:4px; align-items:center; margin-right:8px; }
</style></head>
<body><div class="wrap">
  <div class="stream"><img src="/stream" alt="dashboard stream"></div>
  <div class="tune" id="tune" hidden>
    <div class="bar">
      <input type="search" id="filter" placeholder="filter settings">
      <button id="resetAll" title="Every setting back to its startup value">reset all</button>
    </div>
    <div id="status"></div>
    <div id="groups"></div>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
let params = [];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

async function load() {
  const r = await fetch('/api/params');
  if (!r.ok) return;
  params = await r.json();
  $('#tune').hidden = false;
  render();
}

async function send(url, body) {
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const res = await r.json();
  const bad = Object.entries(res.errors || {});
  $('#status').textContent = bad.length ? bad.map(([k, v]) => k + ': ' + v).join('; ')
                                        : 'applied ' + Object.keys(res.ok || {}).length + ', saved';
  $('#status').style.color = bad.length ? 'var(--err)' : 'var(--ok)';
  document.querySelectorAll('.row').forEach(el => {
    el.classList.toggle('err', el.dataset.path in (res.errors || {}));
    const m = el.querySelector('.msg'); if (m) m.textContent = (res.errors || {})[el.dataset.path] || '';
  });
  setTimeout(load, 300);                     // the node applies changes at its next frame
}
const set = (path, value) => send('/api/params', {[path]: value});

function control(p) {
  const c = document.createElement('div'); c.className = 'ctl';
  const v = 'pending' in p ? p.pending : p.value;
  if (p.kind === 'float' || p.kind === 'int') {
    const range = Object.assign(document.createElement('input'), {type:'range', min:p.lo, max:p.hi, step:p.step, value:v});
    const num = Object.assign(document.createElement('input'), {type:'number', min:p.lo, max:p.hi, step:p.step, value:v});
    range.oninput = () => { num.value = range.value; };
    range.onchange = () => set(p.path, Number(range.value));
    num.onchange = () => set(p.path, Number(num.value));
    c.append(range, num);
  } else if (p.kind === 'bool') {
    const cb = Object.assign(document.createElement('input'), {type:'checkbox', checked:v});
    cb.onchange = () => set(p.path, cb.checked);
    c.append(cb);
  } else if (p.kind === 'choice') {
    const sel = document.createElement('select');
    p.choices.forEach(o => sel.add(new Option(o, o, false, o === v)));
    sel.onchange = () => set(p.path, sel.value);
    c.append(sel);
  } else if (p.kind === 'multi') {
    p.choices.forEach(o => {
      const l = document.createElement('label'); l.className = 'chk';
      const cb = Object.assign(document.createElement('input'), {type:'checkbox', checked:v.includes(o)});
      cb.onchange = () => set(p.path, [...c.querySelectorAll('input')].filter(i => i.checked).map(i => i.value));
      cb.value = o; l.append(cb, o); c.append(l);
    });
  }
  const def = document.createElement('span'); def.className = 'def';
  def.textContent = 'default ' + (Array.isArray(p.default) ? p.default.join(', ') : p.default);
  const rst = Object.assign(document.createElement('button'), {textContent:'reset', title:'Back to the startup value'});
  rst.onclick = () => send('/api/reset', {paths:[p.path]});
  c.append(def, rst);
  return c;
}

function render() {
  const open = new Set([...document.querySelectorAll('details[open]')].map(d => d.dataset.group));
  const f = $('#filter').value.toLowerCase();
  const groups = $('#groups'); groups.textContent = '';
  const byGroup = {};
  params.filter(p => !f || (p.path + ' ' + p.help).toLowerCase().includes(f))
        .forEach(p => (byGroup[p.group] = byGroup[p.group] || []).push(p));
  for (const [g, ps] of Object.entries(byGroup)) {
    const d = document.createElement('details'); d.dataset.group = g;
    d.open = open.has(g) || !!f;
    const changed = ps.filter(p => !same(p.value, p.default)).length;
    d.innerHTML = '<summary></summary>';
    d.firstChild.textContent = g[0].toUpperCase() + g.slice(1) + (changed ? '  (' + changed + ' changed)' : '');
    for (const p of ps) {
      const row = document.createElement('div'); row.className = 'row'; row.dataset.path = p.path;
      row.classList.toggle('changed', !same(p.value, p.default));
      const name = document.createElement('div'); name.className = 'name'; name.textContent = p.path;
      name.tabIndex = 0;                     // focus (tap / keyboard) also shows the tip
      const tip = document.createElement('span'); tip.className = 'tip'; tip.textContent = p.effect;
      name.append(tip);
      const help = document.createElement('div'); help.className = 'help'; help.textContent = p.help;
      const msg = document.createElement('div'); msg.className = 'msg';
      row.append(name, document.createElement('div'), help, control(p), msg);
      d.append(row);
    }
    groups.append(d);
  }
}

$('#filter').oninput = render;
$('#resetAll').onclick = () => {            // two clicks: arm, then confirm
  const b = $('#resetAll');
  if (b.dataset.armed) { delete b.dataset.armed; b.textContent = 'reset all'; send('/api/reset', {}); return; }
  b.dataset.armed = 1; b.textContent = 'click again to reset all';
  setTimeout(() => { delete b.dataset.armed; b.textContent = 'reset all'; }, 3000);
};
load();
</script></body></html>
"""
