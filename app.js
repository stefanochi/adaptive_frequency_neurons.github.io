import { FmcwRadar, NumPy, fmcw_base_config } from './radarEngine.js';
import { AdaptiveResonate } from './adaptiveResonate.js';

// --- 1. Simulation Setup Lifecycle ---
const radar = new FmcwRadar(fmcw_base_config);
const timeAxis = NumPy.linspace(0, radar.t_chirp, radar.n_samples, false);


// ####################
// Initiliaze oscillators
// #####################
// Let's create an Adaptive Resonator network with 32 tracking units
const n_units = 2;
const sim_time = fmcw_base_config.t_chirp; 
const t_res = fmcw_base_config.t_chirp / fmcw_base_config.n_samples; // Match radar sample time step

// w_scale needs to be a 2D matrix [n_rxs, n_units] filled with a scale factor (e.g., 100)
const w_scale = [new Array(n_units).fill(50e-6)]; 

const resonator = new AdaptiveResonate(n_units, sim_time, t_res, 1.0, 0.0, w_scale, 1);
let starting_ranges = [2, 4];
resonator.set_starting_frequency(radar.get_freq_from_range(starting_ranges));
// --- 2. Live State & Buffers ---
const MAX_HISTORY_POINTS = 200; // How many time-steps to show on screen at once
let timeStepCounter = 0;
const timeHistory = [];

// Create an array of empty history arrays—one for each neuron tracking frequency
const frequencyHistoryBuffers = Array.from({ length: n_units }, () => []);

// Generate a continuous stream of radar signal data to feed the loop
radar.set_targets([{ range: 3, velocity: 0, angle: 0 }]); // Target at 35 meters
const radarDataMatrix = radar.generate_data_snr();
let rawChirpSamples = radarDataMatrix[0][0]; // 1D Array of 512 samples
let chirpPlot = rawChirpSamples.map(sample => sample.re);
// ####################
// handle sliders
// #####################
// DOM Elements Bindings
const slider1 = document.getElementById('range1');
const slider2 = document.getElementById('range2');
const display1 = document.getElementById('val1');
const display2 = document.getElementById('val2');

// Capture state variations, load modifications into Engine, yield data
function runSimulationPipeline() {
    const r1 = parseFloat(slider1.value);
    const r2 = parseFloat(slider2.value);

    // Synchronize current slider value indicators
    display1.innerText = r1;
    display2.innerText = r2;

    // Direct assignment to simulation object
    radar.set_targets([
        { range: r1, velocity: 0, angle: 0 },
        { range: r2, velocity: 0, angle: 0 }
    ]);

    // Generate output matrix frame and slice the single primary array
    const matrixFrame = radar.generate_data_snr();
    rawChirpSamples = matrixFrame[0][0];
    chirpPlot = rawChirpSamples.map(sample => sample.re);
    return chirpPlot
}

// --- 2. Initial Plot Initialization ---
const trace = {
    x: timeAxis,
    y: runSimulationPipeline(),
    type: 'scatter',
    mode: 'lines',
    line: { color: '#4f46e5', width: 2 }
};

const layout = {
    title: 'Composite Time-Domain De-chirped Echo Signal',
    xaxis: { title: 'Time (Seconds)', tickformat: '.2e', gridcolor: '#e5e7eb' },
    yaxis: { title: 'Signal Voltage Amplitude', range: [-2.5, 2.5], gridcolor: '#e5e7eb' },
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff'
};

Plotly.newPlot('radarChart', [trace], layout, { responsive: true });
// --- 3. Reactive Event Handler Pipeline ---
function handleUpdate() {
    Plotly.restyle('radarChart', {
        y: [runSimulationPipeline()]
    });
}

// --- 3. Initialize Plotly Layout ---
// Setup a trace for EVERY tracking unit (neuron)
const traces_freq = frequencyHistoryBuffers.map((buffer, index) => ({
    x: timeHistory,
    y: buffer,
    type: 'scatter',
    mode: 'lines',
    name: `Unit ${index + 1}`,
    line: { width: 1.5 }
}));

const layout_freq = {
    title: 'Real-Time Adaptive Frequency Evolution Tracker',
    xaxis: { 
        title: 'Simulation Step', 
        range: [0, MAX_HISTORY_POINTS],
        gridcolor: '#e5e7eb' 
    },
    yaxis: { 
        title: 'Frequency (rad/s)', 
        range: [radar.get_freq_from_range(0.01), radar.get_freq_from_range(4.7)],
        gridcolor: '#e5e7eb' 
    },
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    showlegend: false // Hide legend to keep things fast and neat
};

Plotly.newPlot('frequencyHistory', traces_freq, layout_freq, { responsive: true });

// --- 3. Reactive Event Handler Pipeline ---
function handleUpdateFreq() {
    Plotly.restyle('frequencyHistory', {
        y: [runSimulationPipeline()]
    });
}

// Attach event tracking callbacks to target inputs
slider1.addEventListener('input', handleUpdate);
slider2.addEventListener('input', handleUpdate);

// --- 4. The Continuous Simulation Loop ---
function liveSimulationLoop() {
    // 1. Grab the current incoming signal sample
    const sampleIndex = timeStepCounter % rawChirpSamples.length;
    const currentInputSignal = rawChirpSamples[sampleIndex];

    // 2. Step the adaptive mathematical engine forward by exactly 1 sample
    const { ws } = resonator.update_neurons(currentInputSignal);
    // 3. Update rolling history timeline windows
    timeStepCounter++;
    timeHistory.push(timeStepCounter);
    
    // Store the frequency for our single oscillator
    for (let k=0; k < resonator.nfreq; k++){
        frequencyHistoryBuffers[k].push(ws[0][k] / (2 * Math.PI)); 
    }

    // 4. Wrap/Shift buffers when history limit is reached
    if (timeHistory.length > MAX_HISTORY_POINTS) {
        timeHistory.shift();
        // Trim EVERY oscillator buffer in the array
        for (let k = 0; k < resonator.nfreq; k++) {
            frequencyHistoryBuffers[k].shift();
        }
        
        // Slide the X-axis window forward dynamically
        layout_freq.xaxis.range = [timeStepCounter - MAX_HISTORY_POINTS, timeStepCounter];
    } else {
        // Before hitting the maximum, keep the window starting at 0
        layout_freq.xaxis.range = [0, MAX_HISTORY_POINTS];
    }

    // 5. Explicitly wrap your 1D arrays into arrays of arrays.
    // Plotly needs this structure to know which data belongs to Trace [0].
    const updateData = {
        x: [timeHistory],
        y: [frequencyHistoryBuffers[0]]
    };

    // 6. Tell Plotly exactly to update trace index 0
    Plotly.update('frequencyHistory', updateData, layout_freq, [0]);

    // 7. Loop frame tick
    requestAnimationFrame(liveSimulationLoop);
}
// Start the loop!
liveSimulationLoop();