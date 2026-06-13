// --- NumPy Equivalents for Complex Math and Matrix Utilities ---
const NumPy = {
    // Generates values linearly spaced between start and stop
    linspace: (start, stop, num, endpoint = true) => {
        const arr = [];
        if (num === 1) return [start]; // 👈 Guard clause for single oscillator
        const div = endpoint ? (num - 1) : num;
        const step = (stop - start) / div;
        for (let i = 0; i < num; i++) arr.push(start + step * i);
        return arr;
    },
    // Creates a 2D matrix filled with zeroes
    zeros2D: (rows, cols, isComplex = false) => {
        const matrix = [];
        for (let r = 0; r < rows; r++) {
            const row = [];
            for (let c = 0; c < cols; c++) {
                row.push(isComplex ? { re: 0, im: 0 } : 0);
            }
            matrix.push(row);
        }
        return matrix;
    },
    // Safe division handling to mimic np.nan_to_num (returns 0 if result is NaN)
    nanToNum: (val) => {
        return isNaN(val) || !isFinite(val) ? 0 : val;
    },
    // Safely flattens any nested JS array to pull out the first element
    flattenFirst: (arr) => {
        if (!Array.isArray(arr)) return arr;
        return NumPy.flattenFirst(arr[0]);
    },
    // Deep copy implementation mimicking np.copy
    copy: (obj) => JSON.parse(JSON.stringify(obj))
};

export class AdaptiveResonate {
    constructor(n_units, sim_time, t_res, k, feedback, w_scale, n_rxs = 1) {
        this.t_res = t_res;
        this.sim_time = sim_time;
        this.n_steps = Math.floor(sim_time / t_res);
        this.nfreq = n_units;
        this.n_units = n_units;
        this.n_rxs = n_rxs;
        this.k = k;
        this.feedback = feedback;
        this.w_scale = w_scale; // Expects a 2D Array matrix shape [n_rxs, n_units]

        this.default_w_scale = NumPy.flattenFirst(this.w_scale);
        this.total_neurons = this.n_rxs * this.nfreq;

        this.normalize_input = false;
        this.normalize_neuron = true;
        this.wdot_mode = false;
        this.spike_threshold = 0.0;

        // Initialize tracking matrices
        this.ws = NumPy.zeros2D(this.n_rxs, this.nfreq);
        this.vs = NumPy.zeros2D(this.n_rxs, this.n_units, true); // Complex objects

        // Populating base workspace frequencies
        const baseFreqs = NumPy.linspace(1, 10, this.nfreq);
        for (let rx = 0; rx < this.n_rxs; rx++) {
            for (let k = 0; k < this.nfreq; k++) {
                this.ws[rx][k] = baseFreqs[k] * 2 * Math.PI;
            }
        }

        this.starting_frequency = NumPy.copy(this.ws);
    }

    set_starting_frequency(new_starting_freqs) {
        // new_starting_freqs is expected to be a 1D array of length n_units
        for (let rx = 0; rx < this.n_rxs; rx++) {
            this.ws[rx] = [...new_starting_freqs];
        }
        this.starting_frequency = NumPy.copy(this.ws);
    }

    get_parameters() {
        return {
            t_res: this.t_res,
            n_units: this.nfreq,
            w_scale: this.w_scale,
            feedback: this.feedback,
            k: this.k,
            n_rxs: this.n_rxs,
            normalize_input: this.normalize_input,
            normalize_neuron: this.normalize_neuron,
            starting_frequencies: this.starting_frequency,
            spike_threshold: this.spike_threshold
        };
    }

    integrate_spikes(spikes = null) {
        const targetSpikes = spikes || this.spikes_out;
        const steps = targetSpikes.length; // Axis 0 length
        
        // Setup accumulation tracking matrix initialized with self.starting_frequency
        const integrated_ws = [];
        let current_accum = NumPy.copy(this.starting_frequency);

        for (let i = 0; i < steps; i++) {
            for (let r = 0; r < this.n_rxs; r++) {
                for (let f = 0; f < this.nfreq; f++) {
                    current_accum[r][f] += targetSpikes[i][r][f] * this.spike_threshold * 2 * Math.PI;
                }
            }
            integrated_ws.push(NumPy.copy(current_accum));
        }
        return integrated_ws;
    }

    update_neurons(input_currents) {
        for (let i=0; i<this.n_rxs; i++) {
            this.update_neurons_antenna(input_currents[i], i);
        }
    }

    update_neurons_antenna(input_current, antenna) {
        // Calculate dynamic feedback: summing along vs[0, :]
        let feedback_re = 0;
        let feedback_im = 0;
        for (let k = 0; k < this.nfreq; k++) {
            feedback_re += this.vs[antenna][k].re;
            feedback_im += this.vs[antenna][k].im;
        }

        // input_current can be a scalar real value or a complex object
        const input_re = typeof input_current === 'object' ? input_current.re : input_current;
        const input_im = typeof input_current === 'object' ? input_current.im : 0;

        const in_vals_re = (input_re - feedback_re) / this.nfreq;
        const in_vals_im = (input_im - feedback_im) / this.nfreq;

        let vs_inter = new Array(this.nfreq);

        for (let k = 0; k < this.nfreq; k++) {
            const v = this.vs[antenna][k];
            const magnitude = Math.sqrt(v.re * v.re + v.im * v.im);

            // Equivalent of vs_val = self.vs[0, k] / np.abs(self.vs[0, k])
            // Standardizing boundary cases where magnitude is 0
            const vs_val_re = magnitude === 0 ? 0 : v.re / magnitude;
            const vs_val_im = magnitude === 0 ? 0 : v.im / magnitude;

            // w_dot = vs_val.imag * in_vals.real - vs_val.real * in_vals.imag
            let w_dot = (vs_val_im * in_vals_re) - (vs_val_re * in_vals_im);
            w_dot = NumPy.nanToNum(w_dot);

            // Frequency adaptive step
            this.ws[antenna][k] -= (w_dot / this.w_scale[antenna][k]);

            // Add input current vector
            this.vs[antenna][k].re += in_vals_re;
            this.vs[antenna][k].im += in_vals_im;

            // Capture snapshot copy for intermediate history state tracks
            vs_inter[k] = { re: this.vs[antenna][k].re, im: this.vs[antenna][k].im };

            // Rotate through exponential angular integration: vs *= exp(1j * ws * t_res)
            // Euler's identity formula: exp(j*theta) = cos(theta) + j*sin(theta)
            const theta = this.ws[antenna][k] * this.t_res;
            const cos_t = Math.cos(theta);
            const sin_t = Math.sin(theta);

            const curr_re = this.vs[antenna][k].re;
            const curr_im = this.vs[antenna][k].im;

            // Complex multiplication
            this.vs[antenna][k].re = curr_re * cos_t - curr_im * sin_t;
            this.vs[antenna][k].im = curr_re * sin_t + curr_im * cos_t;
        }

        return { ws: this.ws, vs: this.vs, vs_inter: [vs_inter] };
    }

    update_neurons_frame(in_list, intermediate = false) {
        // in_list is expected to be a 2D structure [channels, time_steps]
        const time_steps = in_list[0].length; 

        // Initializing dimensional structural buffers for history mapping
        const vs_hist_len = intermediate ? time_steps * 2 : time_steps;
        const vs_hist = new Array(vs_hist_len);
        const ws_hist = new Array(time_steps);

        for (let i = 0; i < time_steps; i++) {
            // Unpacking input_current element across time column index
            const current_input = in_list[0][i];

            const { ws, vs, vs_inter } = this.update_neurons_antenna(current_input, 0);

            if (intermediate) {
                vs_hist[i * 2] = NumPy.copy(vs_inter);
                vs_hist[i * 2 + 1] = NumPy.copy(vs);
            } else {
                vs_hist[i] = NumPy.copy(vs);
            }

            ws_hist[i] = NumPy.copy(ws);
        }

        return { vs_hist, ws_hist };
    }
}