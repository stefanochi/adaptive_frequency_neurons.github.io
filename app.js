import { FmcwRadar, NumPy, fmcw_base_config } from './radarEngine.js';
import { AdaptiveResonate } from './adaptiveResonate.js';

// --- 1. Simulation Setup Lifecycle ---
const radar = new FmcwRadar(fmcw_base_config);
const timeAxis = NumPy.linspace(0, radar.t_chirp, radar.n_samples, false);

// --- 2. Live Dynamic State Registries ---
const MAX_HISTORY_POINTS = 200; 
let BATCH_SIZE = 4; 
let timeStepCounter = 0;

// Dynamic tracking stores
let activeTargetsList = []; 
let rawChirpSamples = new Array(fmcw_base_config.n_samples).fill({ re: 0, im: 0 });
let chirpPlot = new Array(fmcw_base_config.n_samples).fill(0);

// Keep track of runtime configuration for oscillators
let currentLambda = 5; // Default lambda scale (x 1e-6)
let resonator = null;
let frequencyHistoryBuffers = [];
let timeHistory = [];

// DOM Container Anchors
const targetContainer = document.getElementById('targetControlsContainer');
const addTargetBtn = document.getElementById('addTargetBtn');
const slider_lambda = document.getElementById('lambda_slider');
const display_lambda = document.getElementById('val_lambda');
const slider_speed = document.getElementById('speed_slider');
const display_speed = document.getElementById('val_speed');
const hannToggle = document.getElementById('hannToggle');

// --- 3. Dynamic Oscillator Network Instantiator ---
function rebuildOscillatorNetwork() {
    const n_units = activeTargetsList.length;

    if (n_units === 0) {
        resonator = null;
        frequencyHistoryBuffers = [];
        return;
    }

    const sim_time = fmcw_base_config.t_chirp;
    const t_res = fmcw_base_config.t_chirp / fmcw_base_config.n_samples;
    const w_scale = [new Array(n_units).fill(currentLambda * 1e-6)];

    // Reconstruct the adaptive resonator network matching the target count
    resonator = new AdaptiveResonate(n_units, sim_time, t_res, 1.0, 0.0, w_scale, 1);

    // Give each oscillator an initial starting range guess close to its corresponding target
    const starting_ranges = activeTargetsList.map(t => t.range - 1 > 0 ? t.range - 1 : t.range);
    resonator.set_starting_frequency(radar.get_freq_from_range(starting_ranges));

    // Reset plotting timelines for the new network footprint
    frequencyHistoryBuffers = Array.from({ length: n_units }, () => []);
    timeHistory = [];
    timeStepCounter = 0;

    // Redraw the frequency tracking chart with the correct number of traces
    rebuildFrequencyPlotTraces();
}

// --- 4. Dynamic Target UI Factory ---
let targetIdCounter = 0;

function createNewTargetUI() {
    targetIdCounter++;
    const id = `target_${targetIdCounter}`;
    let initialRange = Math.round(Math.random() * 4.6 * 10) / 10;
    // Register target state inside the global radar target list
    const targetStateObject = {
        id: id,
        range: initialRange,
        velocity: 0,
        angle: 0
    };
    activeTargetsList.push(targetStateObject);

    // Build the visual HTML slider row component
    const card = document.createElement('div');
    card.className = 'target-card';
    card.id = `card_${id}`;
    card.style = "display: flex; align-items: center; gap: 15px; background: #f3f4f6; padding: 10px; margin-bottom: 8px; border-radius: 6px;";
    card.innerHTML = `
        <strong style="min-width: 80px;">Target ${targetIdCounter}:</strong>
        <input type="range" id="input_${id}" min="0.1" max="4.6" value="${initialRange}" step="0.1" style="flex-grow: 1; cursor: pointer;">
        <span style="min-width: 45px; text-align: right;"><span id="display_${id}" style="font-weight: bold; color: #4f46e5;">${initialRange}</span> m</span>
        <button class="btn-remove" id="delete_${id}" style="background: #ef4444; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer;">Remove</button>
    `;

    targetContainer.appendChild(card);

    // Event hooks for the new dynamic slider
    const sliderInput = card.querySelector(`#input_${id}`);
    const valueDisplay = card.querySelector(`#display_${id}`);

    sliderInput.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        valueDisplay.innerText = val;
        targetStateObject.range = val; // Directly mutate the active array configuration object
        runSimulationPipeline();
    });

    // Event hooks for the target remover
    const deleteBtn = card.querySelector(`#delete_${id}`);
    deleteBtn.addEventListener('click', () => {
        activeTargetsList = activeTargetsList.filter(t => t.id !== id);
        card.remove();
        runSimulationPipeline();
        rebuildOscillatorNetwork();
    });

    // Sync pipeline changes immediately
    runSimulationPipeline();
    rebuildOscillatorNetwork();
}

// --- 5. Composite Signal Simulation Pipeline ---
function runSimulationPipeline() {
    if (activeTargetsList.length === 0) {
        rawChirpSamples = new Array(fmcw_base_config.n_samples).fill({ re: 0, im: 0 });
        chirpPlot = new Array(fmcw_base_config.n_samples).fill(0);
    } else {
        radar.set_targets(activeTargetsList);
        const matrixFrame = radar.generate_data_snr();
        rawChirpSamples = matrixFrame[0][0];
        chirpPlot = rawChirpSamples.map(sample => sample.re);
    }

    // Update time-domain chart layout canvas safely
    Plotly.restyle('radarChart', { y: [chirpPlot] });

    // Instantly recalibrate the neural oscillator network sizing
    // rebuildOscillatorNetwork();
}

// --- 6. Global Setup & Constant Controls Handlers ---
const layout = {
    title: 'Chirp Signal',
    xaxis: { title: 'Time (Seconds)', tickformat: '.2e', gridcolor: '#e5e7eb' },
    yaxis: { title: 'Signal Amplitude', range: [-2.5, 2.5], gridcolor: '#e5e7eb' },
    plot_bgcolor: '#ffffff', paper_bgcolor: '#ffffff'
};
Plotly.newPlot('radarChart', [{ x: timeAxis, y: chirpPlot, type: 'scatter', mode: 'lines', line: { color: '#4f46e5', width: 2 } }], layout, { responsive: true });

function handleLambdaUpdate() {
    currentLambda = parseFloat(slider_lambda.value);
    display_lambda.innerText = currentLambda;
    if (resonator) {
        resonator.w_scale = [new Array(resonator.nfreq).fill(currentLambda * 1e-6)];
    }
}

slider_lambda.addEventListener('input', handleLambdaUpdate);

hannToggle.addEventListener('change', (event) => {
    radar.enable_hann = event.target.checked;
    runSimulationPipeline();
});

// Bind targeted manual addition button action handler
addTargetBtn.addEventListener('click', () => createNewTargetUI());

slider_speed.addEventListener('input', (event) => {
    let val = parseInt(slider_speed.value);
    console.log(val);
    display_speed.innerText = val;
    BATCH_SIZE = val;
});

// --- 7. Dynamic Plotly Traces Management ---
const layout_freq = {
    title: 'Real-Time Adaptive Frequency Evolution',
    xaxis: { title: 'Simulation Step', range: [0, MAX_HISTORY_POINTS], gridcolor: '#e5e7eb' },
    yaxis: { title: 'Tracked Target Range (Meters)', range: [0, 4.7], gridcolor: '#e5e7eb' }, // Changed to Range domain for easy tracking visualization
    plot_bgcolor: '#ffffff', paper_bgcolor: '#ffffff', showlegend: true
};

function rebuildFrequencyPlotTraces() {
    const traces_freq = frequencyHistoryBuffers.map((_, index) => ({
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        name: `Oscillator ${index + 1}`,
        line: { width: 1.5 }
    }));

    Plotly.newPlot('frequencyHistory', traces_freq, layout_freq, { responsive: true });
}

// --- 8. Continuous Adaptive Simulation Loop ---
function liveSimulationLoop() {
    // If no network is instantiated because zero targets exist, skip processing frame step routines safely
    if (!resonator || activeTargetsList.length === 0) {
        requestAnimationFrame(liveSimulationLoop);
        return;
    }

    const newXValues = Array.from({ length: resonator.nfreq }, () => []);
    const newYValues = Array.from({ length: resonator.nfreq }, () => []);

    for (let step = 0; step < BATCH_SIZE; step++) {
        const sampleIndex = timeStepCounter % rawChirpSamples.length;
        const currentInputSignal = rawChirpSamples[sampleIndex];

        var { ws } = resonator.update_neurons(currentInputSignal);
        
        timeStepCounter++;

        for (let k = 0; k < resonator.nfreq; k++) {
            newXValues[k].push(timeStepCounter);
            let range_osc = radar.get_range_from_freq(ws[0][k] / (2 * Math.PI));
            newYValues[k].push(range_osc);
        }
    }

    const traceIndices = Array.from({ length: resonator.nfreq }, (_, i) => i);

    Plotly.extendTraces('frequencyHistory', {
        x: newXValues,
        y: newYValues
    }, traceIndices, MAX_HISTORY_POINTS);

    if (timeStepCounter > MAX_HISTORY_POINTS) {
        layout_freq.xaxis.range = [timeStepCounter - MAX_HISTORY_POINTS, timeStepCounter];
        Plotly.relayout('frequencyHistory', layout_freq);
    }

    requestAnimationFrame(liveSimulationLoop);
}

// Instantiate default configuration targets to mount system pipelines gracefully
runSimulationPipeline();
rebuildOscillatorNetwork();
createNewTargetUI(0.2);
createNewTargetUI(2.0);

// Kickstart execution
liveSimulationLoop();