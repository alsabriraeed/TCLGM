/**
 * TCLGM Web Dashboard Controller
 * Dynamic Path Recommendation, Mastery Visualizer & Graph Explorer
 */

(function () {
  'use strict';

  // State
  let currentDataset = 'ASSISST2012';
  let studentsList = [];
  let conceptsList = [];
  let graphData = { nodes: [], edges: [] };
  let currentRecommendation = null;

  // DOM Elements
  const datasetSelect = document.getElementById('datasetSelect');
  const studentSelect = document.getElementById('studentSelect');
  const timeBudgetSelect = document.getElementById('timeBudgetSelect');
  const targetConceptSelect = document.getElementById('targetConceptSelect');
  const generatePathBtn = document.getElementById('generatePathBtn');
  const studentCountBadge = document.getElementById('studentCountBadge');
  const studentDetailSnippet = document.getElementById('studentDetailSnippet');
  const loadingState = document.getElementById('loadingState');
  const recommendationResults = document.getElementById('recommendationResults');
  const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
  const rightSidebar = document.getElementById('rightSidebar');
  const knowledgeGraphSvg = document.getElementById('knowledgeGraphSvg');

  // Tab Switching
  const tabBtns = document.querySelectorAll('.sidebar-tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });

  // Sidebar toggle
  if (sidebarToggleBtn && rightSidebar) {
    sidebarToggleBtn.addEventListener('click', () => {
      rightSidebar.classList.toggle('collapsed');
    });
  }

  // Collapsible mastered concepts
  const toggleMastered = document.getElementById('toggleMasteredConcepts');
  const masteredListElem = document.getElementById('masteredConceptsList');
  if (toggleMastered && masteredListElem) {
    toggleMastered.addEventListener('click', () => {
      masteredListElem.classList.toggle('hidden');
      const chevron = toggleMastered.querySelector('.chevron-btn');
      if (chevron) {
        chevron.innerHTML = masteredListElem.classList.contains('hidden') ? '&#9656;' : '&#9662;';
      }
    });
  }

  // API Call Helpers
  async function fetchAPI(endpoint, options = {}) {
    try {
      const res = await fetch(endpoint, options);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error(`API error on ${endpoint}:`, err);
      return null;
    }
  }

  // Load Students
  async function loadStudents(dataset) {
    studentSelect.innerHTML = '<option value="" disabled selected>Loading students...</option>';
    studentCountBadge.textContent = 'Loading...';
    
    const data = await fetchAPI(`/api/students?dataset=${dataset}`);
    if (data && data.students && data.students.length > 0) {
      studentsList = data.students;
      studentCountBadge.textContent = `${studentsList.length} Students Available`;
      
      studentSelect.innerHTML = studentsList.map((s, idx) => `
        <option value="${s.uidx}" ${idx === 0 ? 'selected' : ''}>
          ${s.label}
        </option>
      `).join('');

      updateStudentSnippet(studentsList[0]);
    } else {
      studentSelect.innerHTML = '<option value="" disabled>No students found</option>';
      studentCountBadge.textContent = '0 Students';
    }
  }

  // Load Concepts for target dropdown
  async function loadConcepts(dataset) {
    const data = await fetchAPI(`/api/concepts?dataset=${dataset}`);
    if (data && data.concepts) {
      conceptsList = data.concepts;
      targetConceptSelect.innerHTML = `
        <option value="__auto__">✨ Auto-Detect Cognitive Frontier (Recommended)</option>
        ${conceptsList.map(c => `<option value="${c.idx}">${c.name}</option>`).join('')}
      `;
    }
  }

  // Load Graph Structure
  async function loadGraphData(dataset) {
    const data = await fetchAPI(`/api/graph-data?dataset=${dataset}`);
    if (data && data.nodes) {
      graphData = data;
      renderKnowledgeGraph(graphData, currentRecommendation);
    }
  }

  // Update Student detail snippet
  function updateStudentSnippet(student) {
    if (!student) return;
    studentDetailSnippet.innerHTML = `
      <strong>${student.n_interactions} past answers</strong> across ${student.n_unique_concepts} concepts. 
      <br>Topics: <em>${student.sample_concepts.slice(0, 3).join(', ')}...</em>
    `;
  }

  studentSelect.addEventListener('change', () => {
    const selected = studentsList.find(s => s.uidx === parseInt(studentSelect.value, 10));
    updateStudentSnippet(selected);
  });

  // Dataset Switcher
  datasetSelect.addEventListener('change', async () => {
    currentDataset = datasetSelect.value;
    recommendationResults.classList.add('hidden');
    await loadStudents(currentDataset);
    await loadConcepts(currentDataset);
    await loadGraphData(currentDataset);
  });

  // Trigger Recommendation Inference
  async function generatePath() {
    const uidx = parseInt(studentSelect.value, 10);
    const budgetMinutes = parseFloat(timeBudgetSelect.value);
    const targetVal = targetConceptSelect.value;

    const payload = {
      dataset: currentDataset,
      uidx: isNaN(uidx) ? 0 : uidx,
      budget_minutes: budgetMinutes,
    };

    if (targetVal && targetVal !== '__auto__') {
      payload.targets = [parseInt(targetVal, 10)];
    }

    loadingState.classList.remove('hidden');
    recommendationResults.classList.add('hidden');
    generatePathBtn.disabled = true;

    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `HTTP ${res.status}`);
      }

      const result = await res.json();
      currentRecommendation = result;
      renderVisualizations(result);
      renderKnowledgeGraph(graphData, result);
      renderStudentHistory(payload.uidx);
    } catch (err) {
      alert(`Recommendation failed: ${err.message}`);
    } finally {
      loadingState.classList.add('hidden');
      recommendationResults.classList.remove('hidden');
      generatePathBtn.disabled = false;
    }
  }

  generatePathBtn.addEventListener('click', generatePath);

  // Render Visualizations
  function renderVisualizations(result) {
    const d = result.diagnostics;
    if (!d) return;

    // 1. Gauges & Executive KPI
    const currentPct = Math.round(d.current_learning_level * 100);
    const predPct = Math.round(d.predicted_learning_level_after_path * 100);

    document.getElementById('currentLevelVal').textContent = `${currentPct}%`;
    document.getElementById('predictedLevelVal').textContent = `${predPct}%`;

    const circumference = 264;
    const currentOffset = circumference - (d.current_learning_level * circumference);
    const predOffset = circumference - (d.predicted_learning_level_after_path * circumference);

    document.getElementById('currentLevelCircle').style.strokeDashoffset = currentOffset;
    document.getElementById('predictedLevelCircle').style.strokeDashoffset = predOffset;

    document.getElementById('globalGainVal').textContent = (d.global_learning_gain >= 0 ? '+' : '') + d.global_learning_gain.toFixed(3);
    document.getElementById('targetGainVal').textContent = (d.target_learning_gain >= 0 ? '+' : '') + d.target_learning_gain.toFixed(3);
    document.getElementById('estimatedTimeVal').textContent = `${d.budget_minutes} min`;
    document.getElementById('masteredCountChange').textContent = `${d.n_concepts_mastered_before} → ${d.n_concepts_mastered_after}`;

    // 2. Mastered Concepts Cloud
    const masteredList = d.mastered_concepts || [];
    document.getElementById('masteredCountPill').textContent = masteredList.length;
    
    const masteredChipsContainer = document.getElementById('masteredConceptsList');
    if (masteredList.length === 0) {
      masteredChipsContainer.innerHTML = '<span class="label-hint">No concepts currently above 85% mastery threshold.</span>';
    } else {
      masteredChipsContainer.innerHTML = masteredList.map(item => `
        <div class="concept-chip" title="Difficulty: ${item.difficulty}">
          <span class="chip-name">${item.concept_name}</span>
          <span class="chip-score">${Math.round(item.mastery * 100)}%</span>
        </div>
      `).join('');
    }

    // 3. Recommended Path Pipeline Flow (DAG)
    const pathSteps = d.path_steps || [];
    document.getElementById('pathStepsCount').textContent = `${pathSteps.length} Concepts`;
    document.getElementById('nextConceptName').textContent = result.next_concept_name || (pathSteps[0] ? pathSteps[0].concept_name : 'Completed');

    const pipelineContainer = document.getElementById('pathPipelineContainer');
    if (pathSteps.length === 0) {
      pipelineContainer.innerHTML = '<div class="step-card">No additional steps needed. All target concepts mastered!</div>';
    } else {
      pipelineContainer.innerHTML = pathSteps.map((step, idx) => {
        const beforePct = Math.round(step.mastery_before * 100);
        const afterPct = Math.round(step.mastery_after * 100);
        const gainFormatted = (step.gain >= 0 ? '+' : '') + step.gain.toFixed(3);
        const isTargetClass = step.is_target ? 'is-target' : '';

        return `
          <div class="step-card ${isTargetClass}">
            <div class="step-top">
              <div class="step-badge-group">
                <span class="step-num-pill">${step.step}</span>
                <span class="step-concept-name">${step.concept_name}</span>
              </div>
              <span class="step-gain-badge">${gainFormatted}</span>
            </div>

            <!-- Animated Mastery Progression Meter -->
            <div class="step-meter-bar" title="Baseline: ${beforePct}% | Projected: ${afterPct}%">
              <div class="step-meter-after" style="width: ${afterPct}%"></div>
              <div class="step-meter-before" style="width: ${beforePct}%"></div>
            </div>

            <div class="step-meta-row">
              <span>Mastery: ${beforePct}% → <strong style="color: #34d399">${afterPct}%</strong></span>
              <span>Diff: ${step.difficulty} ${step.prerequisites.length ? `| Prereqs: ${step.prerequisites.length}` : ''}</span>
            </div>
          </div>
        `;
      }).join('');
    }

    // 4. Comparison Dual-Bar Chart
    const comparisonChart = document.getElementById('masteryComparisonChart');
    comparisonChart.innerHTML = pathSteps.map(step => {
      const beforePct = Math.round(step.mastery_before * 100);
      const afterPct = Math.round(step.mastery_after * 100);
      return `
        <div class="bar-row">
          <div class="bar-labels">
            <span class="bar-concept-title">${step.concept_name}</span>
            <span class="bar-values">${beforePct}% → <strong>${afterPct}%</strong></span>
          </div>
          <div class="dual-bar-track">
            <div class="dual-bar-projected" style="width: ${afterPct}%"></div>
            <div class="dual-bar-baseline" style="width: ${beforePct}%"></div>
          </div>
        </div>
      `;
    }).join('');

    // 5. Newly Mastered Milestones
    const newlyMastered = d.newly_mastered || {};
    const milestoneKeys = Object.keys(newlyMastered);
    const milestonesCard = document.getElementById('newlyMasteredCard');
    const milestoneChips = document.getElementById('newlyMasteredChips');

    if (milestoneKeys.length === 0) {
      milestoneChips.innerHTML = '<span class="label-hint">Continue study to cross the 85% mastery threshold on additional concepts.</span>';
    } else {
      milestoneChips.innerHTML = milestoneKeys.map(name => `
        <div class="milestone-chip">
          <span class="milestone-icon">🌟</span>
          <span>${name}</span>
          <span class="chip-score">${Math.round(newlyMastered[name] * 100)}%</span>
        </div>
      `).join('');
    }
  }

  // Render Student History
  function renderStudentHistory(uidx) {
    const student = studentsList.find(s => s.uidx === uidx);
    const timelineElem = document.getElementById('studentHistoryTimeline');
    if (!student || !timelineElem) return;

    const concepts = student.concepts || [];
    const correct = student.correct || [];
    const elapsed = student.elapsed_sec || [];

    if (concepts.length === 0) {
      timelineElem.innerHTML = '<p class="label-hint">No history logs for this student.</p>';
      return;
    }

    timelineElem.innerHTML = concepts.map((c, i) => {
      const isCor = correct[i] === 1;
      const conceptName = conceptsList[c] ? conceptsList[c].name : `Concept #${c}`;
      return `
        <div class="timeline-item ${isCor ? 'correct' : 'incorrect'}">
          <span class="item-concept">#${i + 1} ${conceptName}</span>
          <span class="item-badge ${isCor ? 'badge-correct' : 'badge-incorrect'}">
            ${isCor ? '✓ Correct' : '✗ Incorrect'} (${Math.round(elapsed[i] || 0)}s)
          </span>
        </div>
      `;
    }).join('');
  }

  // Render Knowledge Graph in SVG Canvas
  function renderKnowledgeGraph(graph, recommendation) {
    if (!knowledgeGraphSvg || !graph.nodes.length) return;

    const width = knowledgeGraphSvg.clientWidth || 800;
    const height = knowledgeGraphSvg.clientHeight || 600;
    knowledgeGraphSvg.innerHTML = '';

    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    knowledgeGraphSvg.appendChild(g);

    // Zoom & Pan state
    let scale = 1;
    let pointX = 0;
    let pointY = 0;
    let isPanning = false;
    let startX = 0;
    let startY = 0;

    function updateTransform() {
      g.setAttribute('transform', `translate(${pointX}, ${pointY}) scale(${scale})`);
    }

    knowledgeGraphSvg.onmousedown = function (e) {
      isPanning = true;
      startX = e.clientX - pointX;
      startY = e.clientY - pointY;
    };

    window.onmousemove = function (e) {
      if (!isPanning) return;
      pointX = e.clientX - startX;
      pointY = e.clientY - startY;
      updateTransform();
    };

    window.onmouseup = function () {
      isPanning = false;
    };

    document.getElementById('zoomInBtn').onclick = () => { scale *= 1.25; updateTransform(); };
    document.getElementById('zoomOutBtn').onclick = () => { scale /= 1.25; updateTransform(); };
    document.getElementById('resetGraphBtn').onclick = () => { scale = 1; pointX = 0; pointY = 0; updateTransform(); };

    // Layout nodes in a spiral/force layout
    const nodeCount = graph.nodes.length;
    const nodeCoords = {};
    const radius = Math.min(width, height) * 0.4;
    const centerX = width / 2;
    const centerY = height / 2;

    graph.nodes.forEach((n, i) => {
      const angle = i * (Math.PI * 2 / 13) + (i * 0.1);
      const r = 40 + (i / nodeCount) * radius * 1.5;
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;
      nodeCoords[n.id] = { x, y, name: n.name };
    });

    const pathConceptSet = new Set((recommendation && recommendation.path) || []);
    const masteredConceptSet = new Set(
      recommendation && recommendation.diagnostics && recommendation.diagnostics.mastered_concepts
        ? recommendation.diagnostics.mastered_concepts.map(m => m.concept_idx)
        : []
    );

    // Draw edges
    const sampleEdges = graph.edges.slice(0, 400); // render first 400 edges for clean visuals
    sampleEdges.forEach(e => {
      const s = nodeCoords[e.source];
      const t = nodeCoords[e.target];
      if (!s || !t) return;

      const isPathEdge = pathConceptSet.has(e.source) && pathConceptSet.has(e.target);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', s.x);
      line.setAttribute('y1', s.y);
      line.setAttribute('x2', t.x);
      line.setAttribute('y2', t.y);
      line.setAttribute('stroke', isPathEdge ? '#818cf8' : 'rgba(255, 255, 255, 0.06)');
      line.setAttribute('stroke-width', isPathEdge ? '2.5' : '1');
      if (isPathEdge) {
        line.setAttribute('stroke-dasharray', '4,4');
      }
      g.appendChild(line);
    });

    // Draw nodes
    graph.nodes.forEach(n => {
      const coord = nodeCoords[n.id];
      if (!coord) return;

      const isPath = pathConceptSet.has(n.id);
      const isMastered = masteredConceptSet.has(n.id);

      let fillColor = '#1e293b';
      let strokeColor = 'rgba(255, 255, 255, 0.15)';
      let r = 5;

      if (isPath) {
        fillColor = '#6366f1';
        strokeColor = '#a5b4fc';
        r = 9;
      } else if (isMastered) {
        fillColor = '#10b981';
        strokeColor = '#34d399';
        r = 7;
      }

      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', coord.x);
      circle.setAttribute('cy', coord.y);
      circle.setAttribute('r', r);
      circle.setAttribute('fill', fillColor);
      circle.setAttribute('stroke', strokeColor);
      circle.setAttribute('stroke-width', '2');
      circle.style.cursor = 'pointer';

      // Tooltip title
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `${n.name} (ID: ${n.id}, Diff: ${n.difficulty})`;
      circle.appendChild(title);

      circle.onclick = () => {
        alert(`Concept: ${n.name}\nID: ${n.id}\nDifficulty: ${n.difficulty}`);
      };

      g.appendChild(circle);
    });
  }

  // Initial Load
  async function init() {
    await loadStudents(currentDataset);
    await loadConcepts(currentDataset);
    await loadGraphData(currentDataset);

    // Auto-run first recommendation on page load
    setTimeout(() => {
      if (studentsList.length > 0) {
        generatePath();
      }
    }, 400);
  }

  init();
})();
