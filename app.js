import { FmcwRadar, NumPy, fmcw_base_config } from './radarEngine.js';
import { AdaptiveResonate } from './adaptiveResonate.js';

// --- 1. Simulation Setup Lifecycle ---
const radar = new FmcwRadar(fmcw_base_config);
const timeAxis = NumPy.linspace(0, radar.t_chirp, radar.n_samples, false);

// --- 2. Live Dynamic State Registries ---
const MAX_HISTORY_POINTS = 200; 
let BATCH_SIZE = 4; 
let chirpCounter = 0;
let timeStepCounter = 0;

// Dynamic tracking stores
let activeTargetsList = []; 
let rawChirpSamples = new Array(fmcw_base_config.n_samples).fill({ re: 0, im: 0 });
let chirpPlot = new Array(fmcw_base_config.n_samples).fill(0);
let matrixFrame = radar.init_empty_frame();

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
        velocity: 0.0,
        angle: 0
    };
    activeTargetsList.push(targetStateObject);

    // Build the visual HTML slider row component
    const card = document.createElement('div');
    card.className = 'target-card';
    card.id = `card_${id}`;
    card.innerHTML = `
        <strong class="target-label">Target ${targetIdCounter}</strong>
        
        <input type="range" id="input_${id}" min="0.1" max="4.6" value="${initialRange}" step="0.01">
        <input type="range" id="input_${id}_vel" min="-20" max="20" value="0" step="0.01">
        
        <div class="target-card-meta">
            <span class="range-value">Range: <span id="display_${id}" class="value-display">${initialRange}</span> m</span>
            <span class="range-value">Speed: <span id="display_${id}_vel" class="value-display">0</span> m</span>
            <button class="btn-remove" id="delete_${id}">Remove</button>
        </div>
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

    const sliderInput_vel = card.querySelector(`#input_${id}_vel`);
    const valueDisplay_vel = card.querySelector(`#display_${id}_vel`);

    sliderInput_vel.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        valueDisplay_vel.innerText = val;
        targetStateObject.velocity = val; // Directly mutate the active array configuration object
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

// --- Optimized Library FFT 1D Vector Wrapper ---
function libraryFFT1D(complexArray) {
    const N = complexArray.length;
    
    // 1. Initialize the Signalsmith FFT runner for this array size
    const fftInstance = new FFT(N);
    
    // 2. Signalsmith uses interleaved flat Float64Arrays: [r0, i0, r1, i1, ...]
    // This requires exactly 2 * N entries
    const flatInput = new Float64Array(N * 2);
    for (let i = 0; i < N; i++) {
        flatInput[i * 2] = complexArray[i].re;
        flatInput[i * 2 + 1] = complexArray[i].im;
    }
    
    // Allocate the output container spectrum array
    const flatOutput = new Float64Array(N * 2);
    
    // 3. Execute the native compiled transform kernel
    fftInstance.fft(flatInput, flatOutput);
    
    // 4. Pack the interleaved data back out into your standard object structures
    const outputComplex = new Array(N);
    for (let i = 0; i < N; i++) {
        outputComplex[i] = {
            re: flatOutput[i * 2],
            im: flatOutput[i * 2 + 1]
        };
    }
    
    return outputComplex;
}

/**
 * Computes a 2D FFT on a radar frame matrix (Chirps x Samples) using an external FFT library.
 * Includes a vertical FFT shift to center zero-Doppler velocity.
 * * @param {Array<Array<{re: number, im: number}>>} frameMatrix - The 2D time-domain ADC data frame.
 * @returns {Array<Array<number>>} A 2D matrix of logarithmic spectral magnitudes (dB).
 */
function compute2DFFT(frameMatrix) {
    const nChirps = frameMatrix.length;
    const nSamples = frameMatrix[0].length;

    // ==========================================
    // STAGE 1: Horizontal RANGE FFT 
    // ==========================================
    const rangeProcessedMatrix = frameMatrix.map(chirpRow => libraryFFT1D(chirpRow));

    // ==========================================
    // STAGE 2: Vertical DOPPLER FFT
    // ==========================================
    const complex2DMatrix = Array.from({ length: nChirps }, () => new Array(nSamples));

    for (let col = 0; col < nSamples; col++) {
        const columnVector = [];
        for (let row = 0; row < nChirps; row++) {
            columnVector.push(rangeProcessedMatrix[row][col]);
        }

        // Run the 1D FFT vertically down the column vector using the library
        const dopplerProcessedVector = libraryFFT1D(columnVector);

        // Apply FFT-Shift to center the 0-velocity component vertically
        const halfChirps = nChirps / 2;
        for (let row = 0; row < nChirps; row++) {
            const shiftedRowIndex = (row + halfChirps) % nChirps;
            complex2DMatrix[shiftedRowIndex][col] = dopplerProcessedVector[row];
        }
    }

    // ==========================================
    // STAGE 3: Calculate Logarithmic Magnitudes (dB)
    // ==========================================
    return complex2DMatrix.map(row => {
        return row.map(complexSample => {
            const magnitude = Math.sqrt(complexSample.re * complexSample.re + complexSample.im * complexSample.im);
            return 20 * Math.log10(magnitude + 1e-6); // Safeguard against log(0)
        });
    });
}

// --- 5. Composite Signal Simulation Pipeline ---
function runSimulationPipeline() {
    let rdmMagnitudeMatrix = [];

    if (activeTargetsList.length === 0) {
        rawChirpSamples = new Array(fmcw_base_config.n_samples).fill({ re: 0, im: 0 });
        chirpPlot = new Array(fmcw_base_config.n_samples).fill(0);
        matrixFrame = radar.init_empty_frame();
    } else {
        radar.set_targets(activeTargetsList);
        matrixFrame = radar.generate_data_snr();
        rawChirpSamples = matrixFrame[0][0];
        chirpPlot = rawChirpSamples.map(sample => sample.re);
    }

    rdmMagnitudeMatrix = compute2DFFT(matrixFrame[0]);

    // Update time-domain chart layout canvas safely
    Plotly.restyle('radarChart', { y: [chirpPlot] });
    Plotly.restyle('rangeDopplerMap', { z: [rdmMagnitudeMatrix] });
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

// Plotly Configuration for the Range-Doppler Heatmap Canvas
const rangeAxisAxis = Array.from({ length: radar.n_samples }, (_, i) =>  i);
const dopplerAxisAxis = Array.from({ length: radar.n_chirps }, (_, i) => i - (radar.n_chirps / 2));

const rdmTrace = {
    x: rangeAxisAxis,
    y: dopplerAxisAxis,
    z: Array.from({ length: radar.n_chirps }, () => new Array(radar.n_samples).fill(0)),
    type: 'heatmap',
    colorscale: 'Jet',
    showscale: false
};

const rdmLayout = {
    title: 'Live 2D Range-Doppler Spectrogram Map',
    xaxis: { title: 'Range (Meters)', range: [0,   128], gridcolor: '#e5e7eb' },
    yaxis: { title: 'Doppler / Velocity (Bins)', gridcolor: '#e5e7eb' },
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    margin: { t: 50, b: 50, l: 50, r: 20 }
};

Plotly.newPlot('rangeDopplerMap', [rdmTrace], rdmLayout, { responsive: true });

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
        const chirpIndex = chirpCounter % radar.n_chirps;
        const currentInputSignal = matrixFrame[0][chirpIndex][sampleIndex];

        var { ws } = resonator.update_neurons(currentInputSignal);
        
        timeStepCounter++;
        if(timeStepCounter % radar.n_samples === 0){
            chirpCounter++;
        }

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