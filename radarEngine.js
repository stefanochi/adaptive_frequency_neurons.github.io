// fmcw_base_config Object
export const fmcw_base_config = {
    n_rx: 1, 
    fb: 60e9,
    B: 2e9,
    n_chirps: 64, 
    n_samples: 128,
    t_chirp: 6.4e-5,
    IQ: true, 
    noise_std: 0.00001,
};

// Internal custom helper functions mimicking NumPy
export const NumPy = {
    linspace: (start, stop, num, endpoint = true) => {
        const arr = [];
        const div = endpoint ? (num - 1) : num;
        const step = (stop - start) / div;
        for (let i = 0; i < num; i++) arr.push(start + step * i);
        return arr;
    },
    randomNormal: (mean = 0, std = 1) => {
        let u = 0, v = 0;
        while(u === 0) u = Math.random(); 
        while(v === 0) v = Math.random();
        return (Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)) * std + mean;
    },
    // Generates Hanning window coefficients
    hanning: (m) => {
        const window = [];
        for (let i = 0; i < m; i++) {
            window.push(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (m - 1)));
        }
        return window;
    }
};

// Core Physics Simulation Engine
export class FmcwRadar {
    static c = 3e8;

    constructor(config = fmcw_base_config) {
        Object.assign(this, config);
        this.d_rx = (FmcwRadar.c / this.fb) / 2;
        this.random_Ap = false; 
        this.targets_info = [];
        this.target_snr_db = 20;
        this.enable_hann = false;
    }

    set_targets(targets_info) {
        this.targets_info = Array.isArray(targets_info) ? targets_info : [targets_info];
    }

    init_empty_frame() {
        const frame = [];
        for (let r = 0; r < this.n_rx; r++) {
            const chirpArr = [];
            for (let c = 0; c < this.n_chirps; c++) {
                const sampleArr = [];
                for (let s = 0; s < this.n_samples; s++) {
                    // If IQ is true, save as complex object, otherwise scalar 0
                    sampleArr.push(this.IQ ? { re: 0, im: 0 } : 0);
                }
                chirpArr.push(sampleArr);
            }
            frame.push(chirpArr);
        }
        return frame;
    }

    target_add_random_phase_amplitude(target) {
        if (typeof target !== 'object') throw new Error("Target must be an object");

        const A_rand = Math.random() * 2.0 + 0.5;
        const p_rand = Math.random() * 2 * Math.PI;

        if (this.random_Ap) {
            target.A_rand = A_rand;
            target.p_rand = p_rand;
        } else {
            target.A_rand = 1;
            target.p_rand = 0.001;
        }
    }

    generate_data() {
        let frame = this.init_empty_frame();

        for (let target of this.targets_info) {
            this.target_add_random_phase_amplitude(target);
            const frame_target = this._generate_single_target(target);
            frame = this._add_frames(frame, frame_target);
        }

        const noise_std = this.noise_std !== null ? this.noise_std : 0.0;
        const noise = this._generate_noise_frame(noise_std);

        // Calculate power densities for SNR tracking
        let p_sign = 0, p_noise = 0, count = 0;
        this._loop_frame(frame, (val, r, c, s) => {
            if (!this.IQ) {
                p_sign += val * val;
                const nVal = noise[r][c][s];
                p_noise += nVal * nVal;
            } else {
                p_sign += (val.re * val.re + val.im * val.im);
                const nVal = noise[r][c][s];
                p_noise += (nVal.re * nVal.re + nVal.im * nVal.im);
            }
            count++;
        });
        p_sign /= count;
        p_noise /= count;

        this.snr = 10 * Math.log10(p_sign / p_noise);
        this.noise = noise;
        this.pure_signal = JSON.parse(JSON.stringify(frame)); // Deep copy

        return this._add_frames(frame, noise);
    }

    generate_data_snr() {
        let frame = this.init_empty_frame();

        for (let target of this.targets_info) {
            this.target_add_random_phase_amplitude(target);
            const frame_target = this._generate_single_target(target);
            frame = this._add_frames(frame, frame_target);
        }

        // Calculate Signal Power (Ps)
        let p_signal = 0, count = 0;
        this._loop_frame(frame, (val) => {
            p_signal += this.IQ ? (val.re * val.re + val.im * val.im) : (val * val);
            count++;
        });
        p_signal /= count;

        let noise_std = 0.0;
        if (this.target_snr_db !== null) {
            const p_noise = p_signal / Math.pow(10, this.target_snr_db / 10);
            noise_std = Math.sqrt(p_noise);
        } else {
            noise_std = this.noise_std !== null ? this.noise_std : 0.0;
        }

        // Generate Noise
        let noise;
        if (this.IQ) {
            const std_per_component = noise_std / Math.sqrt(2);
            noise = this._generate_noise_frame(std_per_component);
        } else {
            noise = this._generate_noise_frame(noise_std);
        }

        // Finalize stats
        let actual_p_noise = 0;
        this._loop_frame(noise, (val) => {
            actual_p_noise += this.IQ ? (val.re * val.re + val.im * val.im) : (val * val);
        });
        actual_p_noise /= count;

        this.snr = 10 * Math.log10(p_signal / actual_p_noise);
        this.pure_signal = JSON.parse(JSON.stringify(frame));

        frame = this._add_frames(frame, noise);

        if(this.enable_hann){
            return this.apply_hann(frame);
        }
        return frame;
    }

    _generate_single_target(target) {
        const frame = this.init_empty_frame();
        for (let chirp_id = 0; chirp_id < this.n_chirps; chirp_id++) {
            for (let rx_id = 0; rx_id < this.n_rx; rx_id++) {
                frame[rx_id][chirp_id] = this._generate_single_target_chirp(target, chirp_id, rx_id);
            }
        }
        return frame;
    }

    _generate_single_target_chirp(target, chirp_id, rx_id) {
        const t = NumPy.linspace(0, this.t_chirp, this.n_samples, false);
        const current_range = target.range + target.velocity * (chirp_id * this.t_chirp);

        const slope = this.B / this.t_chirp;
        const tau = (2 * current_range) / FmcwRadar.c;
        const fc = this.fb;

        const phase_angle = 2 * Math.PI * (this.d_rx * Math.sin(target.angle) / (FmcwRadar.c / fc)) * rx_id;
        const outChirp = [];

        for (let s = 0; s < this.n_samples; s++) {
            const beat_phase = 2 * Math.PI * (slope * tau * t[s] + fc * tau);
            const total_phase = beat_phase + phase_angle + target.p_rand;

            // Equivalent of Euler's formula: exp(1j * theta) = cos(theta) + j*sin(theta)
            const complex_signal = {
                re: target.A_rand * Math.cos(total_phase),
                im: target.A_rand * Math.sin(total_phase)
            };

            outChirp.push(this.IQ ? complex_signal : complex_signal.re);
        }

        return outChirp;
    }

    apply_hann(frame) {
        const new_frame = this.init_empty_frame();
        const hann_w = NumPy.hanning(this.n_samples);

        for (let r = 0; r < this.n_rx; r++) {
            for (let c = 0; c < this.n_chirps; c++) {
                // Compute mean along samples (DC component)
                let mean_re = 0, mean_im = 0;
                for (let s = 0; s < this.n_samples; s++) {
                    if (this.IQ) {
                        mean_re += frame[r][c][s].re;
                        mean_im += frame[r][c][s].im;
                    } else {
                        mean_re += frame[r][c][s];
                    }
                }
                mean_re /= this.n_samples;
                mean_im /= this.n_samples;

                // Apply Windowing
                for (let s = 0; s < this.n_samples; s++) {
                    if (this.IQ) {
                        const de_dc_re = frame[r][c][s].re - mean_re;
                        const de_dc_im = frame[r][c][s].im - mean_im;
                        new_frame[r][c][s] = {
                            re: de_dc_re * hann_w[s],
                            im: de_dc_im * hann_w[s]
                        };
                    } else {
                        const de_dc = frame[r][c][s] - mean_re;
                        new_frame[r][c][s] = de_dc * hann_w[s];
                    }
                }
            }
        }
        return new_frame;
    }

    // --- Private Structural Helpers ---
    _loop_frame(frame, callback) {
        for (let r = 0; r < this.n_rx; r++) {
            for (let c = 0; c < this.n_chirps; c++) {
                for (let s = 0; s < this.n_samples; s++) {
                    callback(frame[r][c][s], r, c, s);
                }
            }
        }
    }

    _add_frames(f1, f2) {
        const result = this.init_empty_frame();
        this._loop_frame(result, (_, r, c, s) => {
            if (this.IQ) {
                result[r][c][s] = {
                    re: f1[r][c][s].re + f2[r][c][s].re,
                    im: f1[r][c][s].im + f2[r][c][s].im
                };
            } else {
                result[r][c][s] = f1[r][c][s] + f2[r][c][s];
            }
        });
        return result;
    }

    _generate_noise_frame(std) {
        const noise = this.init_empty_frame();
        this._loop_frame(noise, (_, r, c, s) => {
            if (this.IQ) {
                noise[r][c][s] = {
                    re: NumPy.randomNormal(0, std),
                    im: NumPy.randomNormal(0, std)
                };
            } else {
                noise[r][c][s] = NumPy.randomNormal(0, std);
            }
        });
        return noise;
    }

    get_range_from_freq(freq, negative = false) {
        // Calculate max_range if not already explicitly defined on the class instance
        // max_range = (c * t_chirp * max_freq) / (2 * B), where max_freq = n_samples / t_chirp
        const maxRange = this.max_range || (FmcwRadar.c * this.n_samples) / (2 * this.B);

        // Helper to compute the core radar range formula
        const calcRange = (f) => (FmcwRadar.c * this.t_chirp * f) / (2 * this.B);

        // Helper to handle the modulo operation precisely (handling negative frequencies safely)
        const mod = (n, m) => ((n % m) + m) % m;

        if (Array.isArray(freq)) {
            return freq.map(f => {
                const rangeTmp = calcRange(f);
                return negative === false ? mod(rangeTmp, maxRange) : rangeTmp;
            });
        } else {
            const rangeTmp = calcRange(freq);
            return negative === false ? mod(rangeTmp, maxRange) : rangeTmp;
        }
    }

    get_freq_from_range(range) {
        const calcFreq = (r) => (r * 2 * this.B) / (FmcwRadar.c * this.t_chirp);

        if (Array.isArray(range)) {
            return range.map(r => calcFreq(r));
        } else {
            return calcFreq(range);
        }
    }

    get_velocity_from_doppler_frequency(frequency) {
        let lam = FmcwRadar.c / this.fb
        return (lam / 2.0) * frequency;
    }

    wrapVelocitySymmetric(v) {
        const minV = -20;
        const maxV = 20;
        const range = maxV - minV; // 40

        // Shift domain to [0, 40] for standard modulo arithmetic, 
        // force positive mapping, then shift back down to [-20, 20]
        let shifted = (v - minV) % range;
        if (shifted < 0) {
            shifted += range;
        }
        
        return shifted + minV;
    }
}