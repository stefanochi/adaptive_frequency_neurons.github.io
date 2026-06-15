import { FmcwRadar, NumPy, fmcw_base_config } from './radarEngine.js';
import { AdaptiveResonate } from './adaptiveResonate.js';

// --- 1. Simulation Setup Lifecycle ---
const radar = new FmcwRadar(fmcw_base_config);
const timeAxis = NumPy.linspace(0, radar.t_chirp, radar.n_samples, false);

// --- 2. Live Dynamic State Registries ---
const MAX_HISTORY_POINTS = 20000; 
const MAX_HISTORY_POINTS_DOPPLER = Math.floor(MAX_HISTORY_POINTS / fmcw_base_config.n_samples); 
let BATCH_SIZE = 200; 
let chirpCounter = 0;
let timeStepCounter = 0;
let timeStepCounter_doppler = 0;
const sim_time = fmcw_base_config.t_chirp;
const sim_time_doppler = fmcw_base_config.t_chirp * fmcw_base_config.n_chirps;

// Dynamic tracking stores
let activeTargetsList = []; 
let rawChirpSamples = new Array(fmcw_base_config.n_samples).fill({ re: 0, im: 0 });
let chirpPlot = new Array(fmcw_base_config.n_samples).fill(0);
let matrixFrame = radar.init_empty_frame();
const doppler_hann_window = NumPy.hanning(radar.n_chirps);

// Keep track of runtime configuration for oscillators
let currentLambda = 5; // Default lambda scale (x 1e-6)
let currentLambdaDoppler = 5; // Default lambda scale (x 1e-6)
let resonator = null;
let resonator_doppler = null;
let frequencyHistoryBuffers = [];
let timeHistory = [];
let frequencyHistoryBuffers_doppler = [];
let timeHistory_doppler = [];

// DOM Container Anchors
const targetContainer = document.getElementById('targetControlsContainer');
const addTargetBtn = document.getElementById('addTargetBtn');
const slider_lambda = document.getElementById('lambda_slider');
const display_lambda = document.getElementById('val_lambda');
const slider_lambda_doppler = document.getElementById('lambda_slider_doppler');
const display_lambda_doppler = document.getElementById('val_lambda_doppler');
const slider_speed = document.getElementById('speed_slider');
const display_speed = document.getElementById('val_speed');
const slider_noise = document.getElementById('noise_slider');
const display_noise = document.getElementById('val_noise');
const hannToggle = document.getElementById('hannToggle');

function rebuildRangeOscillators(){
    const n_units = activeTargetsList.length;

    const t_res = fmcw_base_config.t_chirp / fmcw_base_config.n_samples;
    const w_scale = [new Array(n_units).fill(currentLambda * 1e-6)];

    // Reconstruct the adaptive resonator network matching the target count
    resonator = new AdaptiveResonate(n_units, sim_time, t_res, 1.0, 0.0, w_scale, 1);

    // Give each oscillator an initial starting range guess close to its corresponding target
    const starting_ranges = activeTargetsList.map(t => t.range - 1 > 0 ? t.range - 1 : t.range);
    resonator.set_starting_frequency(radar.get_freq_from_range(starting_ranges));
}

function rebuildVelocityOscillators(){
    const n_units = 1; // always one neuron per range
    
    const t_res = fmcw_base_config.t_chirp;
    const w_scale = [] 
    for (let i=0; i<activeTargetsList.length; i++){
        let row = [new Array(n_units).fill(sim_time_doppler * 0.005 * currentLambdaDoppler)];
        w_scale.push(row);
    }
    // Reconstruct the adaptive resonator network matching the target count
    resonator_doppler = new AdaptiveResonate(n_units, sim_time_doppler, t_res, 1.0, 0.0, w_scale, activeTargetsList.length);
}

// --- 3. Dynamic Oscillator Network Instantiator ---
function rebuildOscillatorNetwork() {
    const n_units = activeTargetsList.length;

    if (n_units === 0) {
        resonator = null;
        resonator_doppler = null;
        frequencyHistoryBuffers = [];
        frequencyHistoryBuffers_doppler = [];
        return;
    }

    rebuildRangeOscillators();
    rebuildVelocityOscillators();

    // Reset plotting timelines for the new network footprint
    frequencyHistoryBuffers = Array.from({ length: n_units }, () => []);
    timeHistory = [];
    frequencyHistoryBuffers_doppler = Array.from({ length: n_units }, () => []);
    timeHistory_doppler = [];
    timeStepCounter = 0;
    timeStepCounter_doppler = 0;

    // Redraw the frequency tracking chart with the correct number of traces
    rebuildFrequencyPlotTraces();
}

// --- 4. Dynamic Target UI Factory ---
let targetIdCounter = 0;

function createNewTargetUI() {
    targetIdCounter++;
    const id = `target_${targetIdCounter}`;
    let initialRange = Math.round(Math.random() * 9.0 * 10) / 10;
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
            <div class="target-card-header">
                <strong class="target-label">Target ${targetIdCounter}</strong>
                <button class="btn-remove" id="delete_${id}">Remove</button>
            </div>
            
            <div class="target-control-row">
                <input type="range" id="input_${id}" min="0.1" max="10" value="${initialRange}" step="0.01">
                <span class="range-value">Range: <span id="display_${id}" class="value-display">${initialRange}</span> m</span>
            </div>
            
            <div class="target-control-row">
                <input type="range" id="input_${id}_vel" min="-20" max="20" value="0" step="0.01">
                <span class="range-value">Speed: <span id="display_${id}_vel" class="value-display">0</span> m/s</span>
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
 * Includes a vertical Hanning window on slow-time and a vertical FFT shift.
 * * @param {Array<Array<{re: number, im: number}>>} frameMatrix - The 2D time-domain ADC data frame.
 * @returns {Array<Array<number>>} A 2D matrix of logarithmic spectral magnitudes (dB).
 */
function compute2DFFT(frameMatrix) {
    const nChirps = frameMatrix.length;
    const nSamples = frameMatrix[0].length;

    // ==========================================
    // PRE-COMPUTE: Slow-Time (Doppler) Hanning Window
    // ==========================================
    const slowTimeHann = new Float64Array(nChirps);
    for (let row = 0; row < nChirps; row++) {
        // Standard Hanning Window formula
        slowTimeHann[row] = 0.5 * (1 - Math.cos((2 * Math.PI * row) / (nChirps - 1)));
    }

    // ==========================================
    // STAGE 1: Horizontal RANGE FFT 
    // ==========================================
    const rangeProcessedMatrix = frameMatrix.map(chirpRow => libraryFFT1D(chirpRow));

    // ==========================================
    // STAGE 2: Vertical DOPPLER FFT (with Hanning Window)
    // ==========================================
    const complex2DMatrix = Array.from({ length: nChirps }, () => new Array(nSamples));

    for (let col = 0; col < nSamples; col++) {
        const columnVector = [];
        for (let row = 0; row < nChirps; row++) {
            const rangeSample = rangeProcessedMatrix[row][col];
            const w = radar.enable_hann ? slowTimeHann[row] : 1.0; // Get window weight for this chirp index
            
            // Apply window to both real and imaginary parts without muting the original array
            columnVector.push({
                re: rangeSample.re * w,
                im: rangeSample.im * w
            });
        }

        // Run the 1D FFT vertically down the windowed column vector
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
const effectiveFs = fmcw_base_config.n_samples / fmcw_base_config.t_chirp;
const maxRangeMeters = radar.get_range_from_freq(effectiveFs - 1e-5); // Max detectable range (fs / 2)
const rangeAxisAxis = Array.from({ length: radar.n_samples }, (_, i) => {
    return (i / radar.n_samples) * maxRangeMeters;
});

// Calculate physical velocity vectors for the Doppler axis rows
const dopplerAxisAxis = Array.from({ length: radar.n_chirps }, (_, i) => {
    // Center-offset the index bin relative to the vertical FFT Shift
    const binOffset = i - (radar.n_chirps / 2);
    // Convert the bin offset index directly to physical m/s velocity units
    return radar.get_velocity_from_doppler_frequency(binOffset / (radar.t_chirp * radar.n_chirps));
});

const rdmTrace = {
    x: rangeAxisAxis,
    y: dopplerAxisAxis,
    z: Array.from({ length: radar.n_chirps }, () => new Array(radar.n_samples).fill(0)),
    type: 'heatmap',
    colorscale: 'Jet',
    showscale: false
};

// Trace 1: The overlay scatter points (Neurons/Targets tracker markers)
const scatterOverlayTrace = {
    x: [], // Will hold Range values dynamically
    y: [], // Will hold Doppler/Velocity values dynamically
    mode: 'markers',
    type: 'scatter',
    marker: {
        color: '#ffffff',      /* Bright contrast color over Jet heatmap */
        size: 15,
        symbol: 'circle-open', /* Open circle lets you see the peak underneath */
        line: { color: '#00ff00', width: 5 } /* Neon green outline */
    },
    name: 'Tracked Points'
};

const rdmLayout = {
    title: 'Live 2D Range-Doppler Map',
    xaxis: { title: 'Range (Meters)', range: [0,   10], gridcolor: '#e5e7eb' },
    yaxis: { title: 'Doppler / Velocity (Bins)', range: [-20, 20], gridcolor: '#e5e7eb' },
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    margin: { t: 50, b: 50, l: 50, r: 20 }
};

Plotly.newPlot('rangeDopplerMap', [rdmTrace, scatterOverlayTrace], rdmLayout, { responsive: true });

function updateScatterOverlay(currentRanges, currentVelocities) {
    // We update trace index 1 (our scatter trace)
    Plotly.restyle('rangeDopplerMap', {
        x: [currentRanges],
        y: [currentVelocities]
    }, [1]); // explicitly specifies to target the second trace index
}

function handleLambdaUpdate() {
    currentLambda = parseFloat(slider_lambda.value);
    display_lambda.innerText = currentLambda;
    if (resonator) {
        resonator.w_scale = [new Array(resonator.nfreq).fill(currentLambda * 1e-6)];
    }
}

function handleLambdaUpdateDoppler() {
    currentLambdaDoppler = parseFloat(slider_lambda_doppler.value);
    display_lambda_doppler.innerText = currentLambdaDoppler;
    if (resonator_doppler) {
        const w_scale = [] 
        for (let i=0; i<activeTargetsList.length; i++){
            let row = [new Array(1).fill(sim_time_doppler * 0.005 * currentLambdaDoppler)];
            w_scale.push(row);
        }
        resonator_doppler.w_scale = w_scale;
    }
}

slider_lambda.addEventListener('input', handleLambdaUpdate);
slider_lambda_doppler.addEventListener('input', handleLambdaUpdateDoppler);

hannToggle.addEventListener('change', (event) => {
    radar.enable_hann = event.target.checked;
    runSimulationPipeline();
});

// Bind targeted manual addition button action handler
addTargetBtn.addEventListener('click', () => createNewTargetUI());

slider_speed.addEventListener('input', (event) => {
    let rawVal = parseFloat(slider_speed.value);
    const max_speed = radar.n_samples * radar.n_chirps;
    let val = Math.max(1, Math.round(Math.pow(rawVal, 3) * max_speed));
    display_speed.innerText = parseInt(val);
    BATCH_SIZE = val;
});

slider_noise.addEventListener('input', (event) => {
    let val = parseFloat(slider_noise.value);
    display_noise.innerText = parseInt(val);
    radar.target_snr_db = val;
    runSimulationPipeline();
});

// --- 7. Dynamic Plotly Traces Management ---
const layout_freq = {
    title: 'Real-Time Adaptive Frequency Evolution',
    xaxis: { title: 'Simulation Step', range: [0, MAX_HISTORY_POINTS], gridcolor: '#e5e7eb' },
    yaxis: { title: 'Tracked Target Range (Meters)', range: [0, 10], gridcolor: '#e5e7eb' }, // Changed to Range domain for easy tracking visualization
    plot_bgcolor: '#ffffff', paper_bgcolor: '#ffffff', showlegend: true
};
const layout_freq_doppler = {
    title: 'Real-Time Adaptive Frequency Evolution Doppler',
    xaxis: { title: 'Simulation Step', range: [0, MAX_HISTORY_POINTS_DOPPLER], gridcolor: '#e5e7eb' },
    yaxis: { title: 'Tracked Target Velocity (m/s)', range: [-20, 20], gridcolor: '#e5e7eb' }, // Changed to Range domain for easy tracking visualization
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

    const traces_freq_doppler = frequencyHistoryBuffers_doppler.map((_, index) => ({
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        name: `Oscillator ${index + 1}`,
        line: { width: 1.5 }
    }));

    Plotly.newPlot('frequencyHistory', traces_freq, layout_freq, { responsive: true });
    Plotly.newPlot('frequencyHistoryDoppler', traces_freq_doppler, layout_freq_doppler, { responsive: true });
}

// --- 8. Continuous Adaptive Simulation Loop ---
function liveSimulationLoop() {
    // If no network is instantiated because zero targets exist, skip processing frame step routines safely
    if (!resonator || activeTargetsList.length === 0) {
        requestAnimationFrame(liveSimulationLoop);
        return;
    }

    let ws_doppler = resonator_doppler.ws;

    const newXValues = Array.from({ length: resonator.nfreq * resonator.n_rxs}, () => []);
    const newYValues = Array.from({ length: resonator.nfreq * resonator.n_rxs}, () => []);

    let updated_doppler = false;
    const newXValues_doppler = Array.from({ length: resonator_doppler.nfreq * resonator_doppler.n_rxs}, () => []);
    const newYValues_doppler = Array.from({ length: resonator_doppler.nfreq * resonator_doppler.n_rxs}, () => []);

    for (let step = 0; step < BATCH_SIZE; step++) {
        const sampleIndex = timeStepCounter % rawChirpSamples.length;
        const chirpIndex = chirpCounter % radar.n_chirps;

        for (let a=0; a<resonator.n_rxs; a++) {
            const currentInputSignal = matrixFrame[a][chirpIndex][sampleIndex];
            var { ws } = resonator.update_neurons_antenna(currentInputSignal, a);
        }

        if (sampleIndex == Math.floor(radar.n_samples / 2)) {
            updated_doppler = true;
            // update doppler resonators only once per chirp, in the middle
            for (let a=0; a<resonator_doppler.n_rxs; a++) {
                // hardcoded 0 antenna for range
                let currentInputSignal_doppler = resonator.vs[0][a] 
                currentInputSignal_doppler.re = currentInputSignal_doppler.re * doppler_hann_window[chirpIndex];
                currentInputSignal_doppler.im = currentInputSignal_doppler.im * doppler_hann_window[chirpIndex];
                var { ws: ws_doppler_tmp } = resonator_doppler.update_neurons_antenna(currentInputSignal_doppler, a);
                ws_doppler = ws_doppler_tmp;
            }
            timeStepCounter_doppler++;

            for (let a=0; a<resonator_doppler.n_rxs; a++) {
                for (let k = 0; k < resonator_doppler.nfreq; k++) {
                    newXValues_doppler[a*resonator_doppler.nfreq + k].push(timeStepCounter_doppler);
                    let raw_vel = radar.get_velocity_from_doppler_frequency(ws_doppler[a][k] / (2 * Math.PI));
                    let vel_osc = radar.wrapVelocitySymmetric(raw_vel);
                    newYValues_doppler[a*resonator_doppler.nfreq + k].push(vel_osc);
                }
            }
        }
        
        timeStepCounter++;
        if(timeStepCounter % radar.n_samples === 0){
            chirpCounter++;
            if (chirpCounter % radar.n_chirps === 0) {
                runSimulationPipeline();
            }
        }

        for (let a=0; a<resonator.n_rxs; a++) {
            for (let k = 0; k < resonator.nfreq; k++) {
                newXValues[a*resonator.nfreq + k].push(timeStepCounter);
                let range_osc = radar.get_range_from_freq(ws[a][k] / (2 * Math.PI));
                newYValues[a*resonator.nfreq + k].push(range_osc);
            }
        }
    }

    const traceIndices = Array.from({ length: resonator.nfreq * resonator.n_rxs}, (_, i) => i);
    const traceIndices_doppler = Array.from({ length: resonator_doppler.nfreq * resonator_doppler.n_rxs}, (_, i) => i);

    Plotly.extendTraces('frequencyHistory', {
        x: newXValues,
        y: newYValues
    }, traceIndices, MAX_HISTORY_POINTS);

    if (timeStepCounter > MAX_HISTORY_POINTS) {
        layout_freq.xaxis.range = [timeStepCounter - MAX_HISTORY_POINTS, timeStepCounter];
        Plotly.relayout('frequencyHistory', layout_freq);
    }

    if (updated_doppler) {
        Plotly.extendTraces('frequencyHistoryDoppler', {
            x: newXValues_doppler,
            y: newYValues_doppler
        }, traceIndices_doppler, MAX_HISTORY_POINTS_DOPPLER);

        if (timeStepCounter_doppler > MAX_HISTORY_POINTS_DOPPLER) {
            layout_freq_doppler.xaxis.range = [timeStepCounter_doppler - MAX_HISTORY_POINTS_DOPPLER, timeStepCounter_doppler];
            Plotly.relayout('frequencyHistoryDoppler', layout_freq_doppler);
        }
    }

    if (resonator && resonator_doppler && activeTargetsList.length > 0) {
    const overlayRanges = [];
    const overlayDopplers = [];

    // Extract the absolute latest single frequency tracking estimate per active unit 
    for (let k = 0; k < resonator.nfreq; k++) {
        // Map the current raw radian tracking value back to range metrics
        let latestWRange = resonator.ws[0][k]; 
        let rangeEst = radar.get_range_from_freq(latestWRange / (2 * Math.PI));
        overlayRanges.push(rangeEst);

        // Map the current raw Doppler tracking value back to velocity bins metrics
        let latestWDoppler = resonator_doppler.ws[k][0];
        let raw_vel = radar.get_velocity_from_doppler_frequency(latestWDoppler / (2 * Math.PI));
        let dopplerEst = radar.wrapVelocitySymmetric(raw_vel);
        overlayDopplers.push(dopplerEst);
    }

    // Push the fresh coordinates up to overlay on top of your background matrix heatmap frame
    updateScatterOverlay(overlayRanges, overlayDopplers);
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