// fmcw_base_config Object
export const fmcw_base_config = {
    n_rx: 1, 
    fb: 2.4e9,
    B: 750e6,
    n_chirps: 1, 
    n_samples: 512,
    t_chirp: 4e-5,
    IQ: false, 
    noise_std: 0.05,
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
    }

    set_targets(targets_info) {
        this.targets_info = Array.isArray(targets_info) ? targets_info : [targets_info];
    }

    init_empty_frame() {
        const frame = [];
        for (let r = 0; r < this.n_rx; r++) {
            const chirpArr = [];
            for (let c = 0; c < this.n_chirps; c++) {
                const sampleArr = new Array(this.n_samples).fill(0);
                chirpArr.push(sampleArr);
            }
            // chirpArr.push(sampleArr);
            frame.push(chirpArr);
        }
        return frame;
    }

    generate_data() {
        let frame = this.init_empty_frame();
        for (let target of this.targets_info) {
            target.A_rand = 1; 
            target.p_rand = 0;
            const frame_target = this._generate_single_target(target);
            frame = this._add_frames(frame, frame_target);
        }
        const noise = this._generate_noise_frame(this.noise_std || 0);
        return this._add_frames(frame, noise);
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
        
        return t.map(ts => {
            const beat_phase = 2 * Math.PI * (slope * tau * ts + fc * tau);
            return target.A_rand * Math.cos(beat_phase + phase_angle + target.p_rand);
        });
    }

    _add_frames(f1, f2) {
        const result = this.init_empty_frame();
        for (let r = 0; r < this.n_rx; r++) {
            for (let c = 0; c < this.n_chirps; c++) {
                for (let s = 0; s < this.n_samples; s++) {
                    result[r][c][s] = f1[r][c][s] + f2[r][c][s];
                }
            }
        }
        return result;
    }

    _generate_noise_frame(std) {
        const noise = this.init_empty_frame();
        for (let r = 0; r < this.n_rx; r++) {
            for (let c = 0; c < this.n_chirps; c++) {
                for (let s = 0; s < this.n_samples; s++) {
                    noise[r][c][s] = NumPy.randomNormal(0, std);
                }
            }
        }
        return noise;
    }
}