import { FmcwRadar, NumPy, fmcw_base_config } from './radarEngine.js';

// --- 1. Simulation Setup Lifecycle ---
const radar = new FmcwRadar(fmcw_base_config);
const timeAxis = NumPy.linspace(0, radar.t_chirp, radar.n_samples, false);

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
    const matrixFrame = radar.generate_data();
    return matrixFrame[0][0];
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

// Attach event tracking callbacks to target inputs
slider1.addEventListener('input', handleUpdate);
slider2.addEventListener('input', handleUpdate);