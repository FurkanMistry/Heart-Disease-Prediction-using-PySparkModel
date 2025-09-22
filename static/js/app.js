const form = document.getElementById('predict-form');
const result = document.getElementById('result');
const labelEl = document.getElementById('label');
const confEl = document.getElementById('conf');
const barEl = document.getElementById('bar');
const recList = document.getElementById('rec-list');
const gaugeArc = document.getElementById('gaugeArc');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');

function bulletsFromRecommendation(text) {
  return text.split(/;|\.|\n/).map(s => s.trim()).filter(Boolean);
}

function inRange(input) {
  const min = input.getAttribute('min');
  const max = input.getAttribute('max');
  const v = Number(input.value);
  if (min !== null && v < Number(min)) return false;
  if (max !== null && v > Number(max)) return false;
  return true;
}

function recalcBMI() {
  const h = Number(document.getElementById('height')?.value);
  const w = Number(document.getElementById('weight')?.value);
  const bmiEl = document.getElementById('BMI');
  if (!bmiEl) return;
  if (Number.isFinite(h) && h > 0 && Number.isFinite(w)) {
    const bmi = w / ((h/100) ** 2);
    bmiEl.value = bmi.toFixed(2);
  }
}

['height','weight'].forEach(id => {
  const el = document.getElementById(id);
  el?.addEventListener('input', recalcBMI);
});

// Initialize BMI if height/weight are already present from previous session autofill
window.addEventListener('DOMContentLoaded', recalcBMI);

form?.addEventListener('submit', async (e) => {
  e.preventDefault();
  // Clear previous error styles
  form.querySelectorAll('input, select').forEach(el => el.classList.remove('error'));

  const data = {};
  const fields = form.querySelectorAll('input, select');
  for (const el of fields) {
    if (!el.value && !el.hasAttribute('readonly')) { el.reportValidity?.(); el.classList.add('error'); return; }
    const val = Number(el.value);
    if (!Number.isFinite(val)) { alert(`Invalid value for ${el.name}`); el.classList.add('error'); return; }
    if (el.tagName === 'INPUT' && el.type === 'number' && !inRange(el)) {
      alert(`${el.name} must be between ${el.getAttribute('min')} and ${el.getAttribute('max')}`);
      el.classList.add('error');
      return;
    }
    data[el.name] = val;
  }
  try {
    const prevText = predictBtn?.textContent;
    if (predictBtn) { predictBtn.disabled = true; predictBtn.innerHTML = '<span class="btn-icon">⏳</span><span>Predicting…</span>'; }
    const resp = await fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data) });
    const json = await resp.json();
    if (!resp.ok) {
      if (json.details) console.error(json.details, json.trace || '');
      if (json.errors || json.details) window.renderErrors?.(json.errors || json.details);
      alert(json.error || 'Prediction failed');
      if (predictBtn) { predictBtn.disabled = false; predictBtn.innerHTML = prevText || 'Predict'; }
      return;
    }
  labelEl.textContent = json.label + (json.confidence >= 70 ? ' 🎯' : json.confidence >= 40 ? ' ⚠️' : ' 🌿');
    confEl.textContent = `${json.confidence}%`;
    // Gauge stroke animation
    if (gaugeArc) {
      const total = 157; // from CSS stroke-dasharray
      const offset = total - (total * Math.max(0, Math.min(100, json.confidence)) / 100);
      gaugeArc.style.transition = 'stroke-dashoffset .9s cubic-bezier(.2,.8,.2,1)';
      requestAnimationFrame(()=>{ gaugeArc.style.strokeDashoffset = String(offset); });
    }
    if (barEl) barEl.style.width = `${json.confidence}%`;
    recList.innerHTML = '';
    bulletsFromRecommendation(json.recommendation).forEach(t => { const li = document.createElement('li'); li.textContent = t; recList.appendChild(li); });
    result.classList.remove('hidden');
    result.scrollIntoView({ behavior:'smooth', block:'center' });
    // confetti burst
    burstConfetti(json.confidence);
    // enable reset flow
    if (resetBtn) resetBtn.classList.remove('hidden');
    if (predictBtn) predictBtn.disabled = true;
  } catch (err) { console.error(err); alert('Network error'); }
  finally {
    if (predictBtn && !resetBtn?.classList.contains('hidden')) {
      // keep disabled until reset
    } else if (predictBtn) {
      predictBtn.disabled = false; predictBtn.innerHTML = '<span class="btn-icon">🧠</span><span>Predict</span>';
    }
  }
});

function burstConfetti(conf){
  try{
    const colors = ['#34d399','#22d3ee','#7c3aed','#fbbf24','#f87171'];
    const count = 12 + Math.round((conf||0)/10);
    for(let i=0;i<count;i++){
      const s = document.createElement('span');
      s.className = 'confetti';
      s.style.position='fixed';
      s.style.left = (50 + (Math.random()*20-10))+'%';
      s.style.top = '10%';
      s.style.width = s.style.height = (6+Math.random()*6)+'px';
      s.style.background = colors[Math.floor(Math.random()*colors.length)];
      s.style.borderRadius = Math.random()>.5?'2px':'50%';
      s.style.transform = `rotate(${Math.random()*360}deg)`;
      s.style.zIndex = 9999;
      s.style.opacity = '0.95';
      s.style.transition = 'transform 900ms ease, top 900ms ease, opacity 1200ms ease';
      document.body.appendChild(s);
      requestAnimationFrame(()=>{
        s.style.top = (80 + Math.random()*10)+'%';
        s.style.transform = `translate(${(Math.random()*200-100)}px, 0) rotate(${Math.random()*720-360}deg)`;
        s.style.opacity = '0';
      });
      setTimeout(()=> s.remove(), 1300);
    }
  }catch(_){ /* no-op */ }
}

resetBtn?.addEventListener('click', () => {
  try{
    // clear inputs EXCEPT readonly
    form?.querySelectorAll('input, select').forEach(el => {
      if (!el.hasAttribute('readonly')) {
        if (el.tagName === 'SELECT') el.selectedIndex = 0;
        else el.value = '';
      }
      el.classList.remove('error'); el.title = '';
    });
    // reset BMI if present
    recalcBMI();
    // reset gauge and bar
    if (gaugeArc) {
      gaugeArc.style.transition = 'stroke-dashoffset .3s ease';
      gaugeArc.style.strokeDashoffset = '157';
    }
    if (barEl) barEl.style.width = '0%';
    labelEl.textContent = '';
    confEl.textContent = '0%';
    recList.innerHTML = '';
    result.classList.add('hidden');
  } finally {
    // restore buttons
    if (predictBtn) { predictBtn.disabled = false; predictBtn.innerHTML = '<span class="btn-icon">🧠</span><span>Predict</span>'; }
    if (resetBtn) resetBtn.classList.add('hidden');
    window.scrollTo({ top:0, behavior:'smooth' });
  }
});
