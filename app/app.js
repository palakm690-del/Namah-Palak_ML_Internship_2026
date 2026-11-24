const form = document.querySelector('#profile-form');
const feedback = document.querySelector('#form-feedback');
const submitButton = form.querySelector('button[type="submit"]');
const statusChip = document.querySelector('#model-status');

const fieldNames = {
  Age: 'Age', Gender: 'Gender', Degree: 'Degree', Branch: 'Branch', CGPA: 'CGPA',
  Backlogs: 'Backlogs', Internships: 'Internships', Projects: 'Projects',
  Coding_Skills: 'Coding skills', Communication_Skills: 'Communication skills',
  Aptitude_Test_Score: 'Aptitude score', Soft_Skills_Rating: 'Soft skills', Certifications: 'Certifications',
};

function updateSlider(slider) {
  const output = document.getElementById(`${slider.id}-value`);
  if (output) output.value = slider.value;
  const min = Number(slider.min);
  const max = Number(slider.max);
  const progress = ((Number(slider.value) - min) / (max - min)) * 100;
  slider.style.setProperty('--range-progress', `${progress}%`);
}

document.querySelectorAll('.range-input').forEach((slider) => {
  updateSlider(slider);
  slider.addEventListener('input', () => updateSlider(slider));
});

async function checkModel() {
  try {
    const response = await fetch('/api/health');
    const state = await response.json();
    statusChip.innerHTML = `<i></i> ${state.ready ? 'Random Forest ready' : 'Model unavailable'}`;
    statusChip.classList.toggle('offline', !state.ready);
  } catch {
    statusChip.innerHTML = '<i></i> Start local app to connect model';
    statusChip.classList.add('offline');
  }
}

function renderImpacts(items) {
  const list = document.querySelector('#impact-list');
  if (!items.length) {
    list.innerHTML = '<p class="empty-explanation">No non-zero feature contributions were returned for this profile.</p>';
    return;
  }
  const max = Math.max(...items.map((item) => Math.abs(item.value)), 0.0001);
  list.innerHTML = items.map((item, index) => {
    const direction = item.value >= 0 ? 'positive' : 'negative';
    const width = Math.max(3, (Math.abs(item.value) / max) * 100);
    const sign = item.value > 0 ? '+' : '';
    const effect = `${sign}${(item.value * 100).toFixed(1)} pp`;
    return `<div class="impact-item ${direction}">
      <div class="impact-top"><span>${fieldNames[item.feature] || item.feature}</span><span class="impact-value">${effect}</span></div>
      <div class="impact-bar"><i style="width:${width}%"></i></div>
      <p>${item.value >= 0 ? 'Pushed the model toward Placed' : 'Pushed the model toward Not Placed'} · profile value: ${item.display_value} · rank ${String(index + 1).padStart(2, '0')}</p>
    </div>`;
  }).join('');
}

function renderRecommendations(items, current) {
  const container = document.querySelector('#growth-cards');
  if (!items.length) {
    container.innerHTML = '<article class="growth-card empty-growth"><span class="growth-icon">↗</span><h3>No single-field uplift found</h3><p>The model did not find a stronger result from changing one input at a time within the tested range. Use the SHAP factors as discussion points for a broader development plan.</p><span class="signal-label">MODEL SIMULATION</span></article>';
    return;
  }
  container.innerHTML = items.map((item, index) => {
    const name = fieldNames[item.feature] || item.feature;
    const movement = item.direction === 'decrease' ? 'Reduce' : 'Increase';
    const delta = (item.delta * 100).toFixed(1);
    return `<article class="growth-card">
      <span class="growth-number">${String(index + 1).padStart(2, '0')}</span>
      <span class="growth-icon">${item.direction === 'decrease' ? '↘' : '↗'}</span>
      <h3>${movement} ${name.toLowerCase()}</h3>
      <p>Model scenario: ${item.from} → ${item.to}. Estimated placement probability moves from ${Math.round(current * 100)}% to ${Math.round(item.probability * 100)}% (+${delta} percentage points).</p>
      <span class="signal-label">ONE-INPUT SCENARIO</span>
    </article>`;
  }).join('');
}

function renderPrediction(result) {
  const percent = Math.round(result.probability * 100);
  document.querySelector('#score-value').innerHTML = `${percent}<span>%</span>`;
  document.querySelector('#score-caption').innerHTML = 'Estimated by the fitted<br>Random Forest model';
  document.querySelector('#prediction-label').innerHTML = `<i></i> ${result.prediction}`;
  document.querySelector('#score-bar').style.width = `${percent}%`;
  document.querySelector('#score-marker').style.left = `${percent}%`;
  document.querySelector('#result-badge').innerHTML = '<i></i> MODEL RESULT';
  document.querySelector('#outcome-note').innerHTML = `<strong>Model output</strong><br>${result.prediction} · ${result.probability.toFixed(4)} probability`;
  renderImpacts(result.explanations);
  renderRecommendations(result.recommendations, result.probability);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  feedback.textContent = '';
  submitButton.disabled = true;
  submitButton.innerHTML = 'Scoring profile… <span>◌</span>';
  try {
    const profile = Object.fromEntries(new FormData(form).entries());
    for (const [key, value] of Object.entries(profile)) {
      if (key !== 'Gender' && key !== 'Degree' && key !== 'Branch') profile[key] = Number(value);
    }
    const response = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(profile),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The model could not score this profile.');
    renderPrediction(result);
    feedback.textContent = `Scored with the saved project model. SHAP base output: ${(result.base_value * 100).toFixed(1)}%.`;
    document.querySelector('.results-column').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    feedback.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = 'Explore placement outlook <span>→</span>';
  }
});

form.addEventListener('input', () => { feedback.textContent = ''; });
form.addEventListener('change', () => { feedback.textContent = ''; });
checkModel();
