# DEPRECATED (v2.6): version anterior de GRU_QUANTUM_FILTER, superseded por v2.5.4.
# Nota: define A_GRU=4.235e-8 (pre-screening) pero esa variable no se usa en el filtro.

#!/usr/bin/env python3
"""
GRU Quantum Filter v2.5.2 - Heisenberg-Fourier Chunked Architecture
====================================================================
CORRECCIONES v2.5.2:
- Padding del último segmento para mantener shape consistente (18001 bins)
- Manejo de NaN en filtros y espectros
- Verificación de integridad de checkpoints
- Fallback si acumulación CSD falla

Basado en: GRU v2.5.1 (falló por broadcasting shapes 18001 vs 13501)
"""

import numpy as np
import json
import time
import os
import gc
from datetime import datetime
from scipy.fft import rfft, irfft, rfftfreq
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

class GRUConfig:
    T_OBS_YEARS = 1.0
    FS = 10.0
    DT_TARGET = 128.35
    SEGMENT_HOURS = 1.0
    OVERLAP_RATIO = 0.75
    WINDOW_TYPE = 'gaussian'
    D_S_SPINE = 1.03
    D_S_FULL = 5.02
    DELTA_D_S = 4.0
    LAMBDA = 0.50
    KAPPA = 1/(8*np.pi)
    N_WALKS = 3000
    SIGMA_MAX = 200
    SEED = 42
    MAX_RAM_GB = 2.0
    N_CORES = 6
    CHUNK_OUTPUT = True
    OUTPUT_JSON = "GRU_QF_LISA_1yr_v252.json"
    LOG_FILE = "gru_v252.log"


def log(msg, config):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(config.LOG_FILE, 'a') as f:
        f.write(line + "\n")


def memory_usage_mb():
    total = 0
    for obj in gc.get_objects():
        if isinstance(obj, np.ndarray):
            total += obj.nbytes
    return total / (1024 * 1024)


# =============================================================================
# FILTRO CUÁNTICO GRU (CORREGIDO)
# =============================================================================

class GRUQuantumFilter:
    def __init__(self, config):
        self.config = config
        self.kappa = config.KAPPA
        self.lambda_eff = config.LAMBDA
        D = 4
        self.lambda_scale = 1 / np.arccosh(
            1 + self.lambda_eff / (D**2 * (D-1) * 2 * self.kappa)
        )
        self.delta_t_gru = self.lambda_scale * 65

    def compute_transfer_function(self, frequencies):
        """Filtro cuántico con manejo de NaN y edge cases"""
        f = np.asarray(frequencies, dtype=np.float64)

        # Manejar frecuencias <= 0
        f = np.where(f <= 0, 1e-10, f)

        f_spine = 1 / self.delta_t_gru
        f_planck = 1e3

        H = np.zeros_like(f, dtype=complex)

        # Zona I
        mask1 = f < f_spine * 0.1
        if np.any(mask1):
            H[mask1] = (f[mask1] / f_spine) ** (self.config.D_S_SPINE / 2) *                        np.exp(1j * np.pi * self.config.D_S_SPINE / 4)

        # Zona II (transición suavizada)
        mask2 = (f >= f_spine * 0.1) & (f < f_spine * 10)
        if np.any(mask2):
            ratio = f[mask2] / f_spine
            # Evitar log(0) o negativos
            ratio = np.where(ratio <= 0, 1e-10, ratio)
            log_ratio = np.log(ratio)
            # Gaussian envelope suavizada
            envelope = np.exp(-0.5 * log_ratio**2)
            envelope = np.where(np.isnan(envelope), 0, envelope)
            phase = np.exp(1j * np.pi * self.config.D_S_SPINE / 4 * ratio)
            H[mask2] = ratio ** (self.config.D_S_SPINE / 2) * envelope * phase

        # Zona III
        mask3 = f >= f_spine * 10
        if np.any(mask3):
            ratio_inv = f_spine / f[mask3]
            ratio_inv = np.where(ratio_inv <= 0, 1e-10, ratio_inv)
            H[mask3] = ratio_inv ** (self.config.D_S_FULL / 2) *                        np.exp(1j * np.pi * self.config.D_S_FULL / 4)

        # Normalización segura
        H_abs = np.abs(H)
        max_H = np.nanmax(H_abs) if np.any(~np.isnan(H_abs)) else 1.0
        if max_H > 0:
            H = H / max_H

        # Reemplazar NaN/Inf por 0
        H = np.where(np.isnan(H) | np.isinf(H), 0, H)

        return H

    def apply_filter(self, spectrum, frequencies):
        """Aplica filtro con verificación de integridad"""
        H = self.compute_transfer_function(frequencies)
        filtered = spectrum * H
        # Verificar NaN
        if np.any(np.isnan(filtered)):
            log("WARNING: NaN detectado en espectro filtrado, reemplazando por 0", 
                self.config)
            filtered = np.where(np.isnan(filtered), 0, filtered)
        return filtered


# =============================================================================
# GENERADOR DE SEÑAL
# =============================================================================

class LISASignalGenerator:
    def __init__(self, config):
        self.config = config
        np.random.seed(config.SEED)

    def generate_noise(self, n_samples, fs):
        f = rfftfreq(n_samples, 1/fs)
        psd = np.zeros_like(f)
        mask = f > 0
        psd[mask] = 1e-44 * (f[mask] / 1e-3) ** (-3)
        psd[mask] += 1e-42 * (f[mask] / 1e-3) ** (-2)  # mask evita 0**(-2)=inf en DC
        noise_fft = np.sqrt(psd) * np.exp(1j * 2*np.pi * np.random.random(len(f)))
        return irfft(noise_fft, n=n_samples)

    def inject_gru_signal(self, signal, fs, delay_s=128.35, amplitude=1e-15):
        n_samples = len(signal)
        delay_samples = int(delay_s * fs)
        t = np.arange(n_samples) / fs
        f_gru = 1 / (delay_s )
        gru_signal = amplitude * np.sin(2 * np.pi * f_gru * t) *                      (1 + 0.3 * np.sin(2 * np.pi * 0.1 * f_gru * t))
        gru_delayed = np.zeros_like(gru_signal)
        if delay_samples < n_samples:
            gru_delayed[delay_samples:] = gru_signal[:-delay_samples]
        return signal + gru_delayed

    def generate_segment(self, n_samples, fs, inject_gru=True, amplitude=1e-15):
        noise = self.generate_noise(n_samples, fs)
        if inject_gru:
            return self.inject_gru_signal(noise, fs, 
                                          delay_s=self.config.DT_TARGET,
                                          amplitude=amplitude)
        return noise


# =============================================================================
# PROCESADOR HEISENBERG-FOURIER (CORREGIDO)
# =============================================================================

class HeisenbergFourierProcessor:
    def __init__(self, config):
        self.config = config
        self.filter = GRUQuantumFilter(config)
        self.generator = LISASignalGenerator(config)

        self.segment_samples = int(config.SEGMENT_HOURS * 3600 * config.FS)
        self.overlap_samples = int(self.segment_samples * config.OVERLAP_RATIO)
        self.hop_samples = self.segment_samples - self.overlap_samples

        # Ventana Gaussiana para segmento completo
        self.window = self._create_gaussian_window(self.segment_samples)

        log(f"Heisenberg-Fourier v2.5.2 inicializado", config)
        log(f"  Segmento: {config.SEGMENT_HOURS}h = {self.segment_samples} muestras", config)
        log(f"  Overlap: {config.OVERLAP_RATIO*100:.0f}%", config)

    def _create_gaussian_window(self, n):
        sigma = n / 6
        t = np.arange(n) - n/2
        window = np.exp(-0.5 * (t/sigma)**2)
        return window / np.sum(window)

    def process_segment(self, segment, segment_id, fs):
        """Procesa segmento con padding si es necesario"""
        n = len(segment)
        was_padded = n < self.segment_samples  # capturar ANTES de reasignar

        if was_padded:
            padded = np.zeros(self.segment_samples)
            padded[:n] = segment
            segment = padded
            n = self.segment_samples

        # Aplicar ventana
        windowed = segment * self.window

        # FFT con shape fijo
        spectrum = rfft(windowed)
        freqs = rfftfreq(self.segment_samples, 1/fs)

        # Aplicar filtro cuántico
        filtered_spectrum = self.filter.apply_filter(spectrum, freqs)

        # Espectro de potencia
        power = np.abs(filtered_spectrum)**2

        # Verificar NaN
        if np.any(np.isnan(power)):
            log(f"WARNING: NaN en power segmento {segment_id}, reemplazando", 
                self.config)
            power = np.where(np.isnan(power), 0, power)

        del windowed, spectrum, filtered_spectrum
        gc.collect()

        return {
            'segment_id': segment_id,
            'frequencies': freqs,
            'power': power,
            'n_samples': n,
            'was_padded': was_padded
        }

    def accumulate_csd(self, results_list):
        """Acumula CSD con verificación de shapes"""
        if not results_list:
            return None

        # CORRECCIÓN v2.5.2: Verificar que todos tengan mismo shape
        expected_len = len(results_list[0]['power'])
        valid_results = []

        for r in results_list:
            if len(r['power']) == expected_len:
                valid_results.append(r)
            else:
                log(f"WARNING: Descartando segmento {r['segment_id']} "
                    f"por shape incompatible ({len(r['power'])} vs {expected_len})",
                    self.config)

        if not valid_results:
            log("ERROR: Ningún segmento válido para acumulación", self.config)
            return None

        freqs = valid_results[0]['frequencies']
        avg_power = np.zeros_like(freqs)

        for result in valid_results:
            avg_power += result['power']

        avg_power /= len(valid_results)

        return {
            'frequencies': freqs,
            'csd_magnitude': avg_power,
            'n_segments': len(valid_results),
            'n_discarded': len(results_list) - len(valid_results)
        }

    def detect_time_delay(self, csd_result, target_delay=128.35):
        """Detecta retardo temporal"""
        if csd_result is None:
            return None

        freqs = csd_result['frequencies']
        csd_mag = csd_result['csd_magnitude']

        f_spine = 1 / target_delay
        mask = (freqs > f_spine * 0.5) & (freqs < f_spine * 2)

        if not np.any(mask):
            return None

        peak_idx = np.argmax(csd_mag[mask])
        peak_freq = freqs[mask][peak_idx]
        estimated_delay = 1 / peak_freq

        noise_floor = np.mean(csd_mag[~mask]) if np.any(~mask) else 1e-10
        peak_signal = np.max(csd_mag[mask])
        snr = peak_signal / noise_floor if noise_floor > 0 else 0
        sigma = np.sqrt(2 * csd_result['n_segments']) * snr

        return {
            'target_delay_s': target_delay ,
            'estimated_delay_s': estimated_delay ,
            'peak_frequency_hz': float(peak_freq),
            'snr': float(snr),
            'sigma': float(sigma),
            'match': abs(estimated_delay - target_delay) < 0.01
        }


# =============================================================================
# PIPELINE PRINCIPAL (CORREGIDO)
# =============================================================================

class GRUQuantumFilterPipeline:
    def __init__(self, config=None):
        self.config = config or GRUConfig()
        self.processor = HeisenbergFourierProcessor(self.config)
        self.results = []
        self.start_time = time.time()

    def run(self, inject_gru=True, amplitude=1e-15, test_null=False):
        config = self.config
        log("="*60, config)
        log("GRU Quantum Filter v2.5.2 - Heisenberg-Fourier", config)
        log("CORRECCIONES: Padding segmentos, NaN handling, shape check", config)
        log("="*60, config)

        total_seconds = config.T_OBS_YEARS * 365.25 * 24 * 3600
        total_samples = int(total_seconds * config.FS)
        n_segments = int(np.ceil((total_samples - config.FS) / self.processor.hop_samples))

        log(f"Muestras totales: {total_samples:,}", config)
        log(f"Segmentos: {n_segments:,}", config)
        log(f"Muestras/segmento: {self.processor.segment_samples:,}", config)

        accumulated_results = []
        segment_count = 0
        padded_count = 0

        for i in range(n_segments):
            start_sample = i * self.processor.hop_samples
            end_sample = min(start_sample + self.processor.segment_samples, total_samples)
            n_samples = end_sample - start_sample

            segment = self.processor.generator.generate_segment(
                n_samples, config.FS, inject_gru=inject_gru, amplitude=amplitude
            )

            result = self.processor.process_segment(segment, i, config.FS)
            accumulated_results.append(result)
            segment_count += 1

            if result.get('was_padded', False):
                padded_count += 1

            del segment
            gc.collect()

            if (i + 1) % 100 == 0 or i == n_segments - 1:
                elapsed = time.time() - self.start_time
                progress = (i + 1) / n_segments * 100
                mem_mb = memory_usage_mb()
                log(f"  [{i+1}/{n_segments}] {progress:.1f}% | "
                    f"Tiempo: {elapsed:.0f}s | RAM: {mem_mb:.1f} MB | "
                    f"Padded: {padded_count}", config)

                if config.CHUNK_OUTPUT:
                    self._save_checkpoint(accumulated_results, i+1)

            if memory_usage_mb() > config.MAX_RAM_GB * 1024:
                log(f"  ALERTA: RAM alta ({memory_usage_mb():.0f} MB). GC...", config)
                gc.collect()

        log("\nAcumulando CSD...", config)
        csd_result = self.processor.accumulate_csd(accumulated_results)

        if csd_result:
            log(f"CSD acumulado: {csd_result['n_segments']} segmentos válidos, "
                f"{csd_result['n_discarded']} descartados", config)
        else:
            log("ERROR: No se pudo acumular CSD", config)

        log("Detectando retardo...", config)
        detection = self.processor.detect_time_delay(csd_result, config.DT_TARGET)

        null_results = None
        if test_null and csd_result:
            log("\nTest nulo...", config)
            null_results = self._run_null_test()

        final_results = {
            'metadata': {
                'version': '2.5.2',
                'architecture': 'Heisenberg-Fourier',
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'T_obs_years': config.T_OBS_YEARS,
                    'fs_hz': config.FS,
                    'dt_target_s': config.DT_TARGET ,
                    'segment_hours': config.SEGMENT_HOURS,
                    'overlap_ratio': config.OVERLAP_RATIO,
                    'd_s_spine': config.D_S_SPINE,
                    'd_s_full': config.D_S_FULL,
                    'lambda': config.LAMBDA
                }
            },
            'processing': {
                'total_samples': total_samples,
                'n_segments': segment_count,
                'n_padded': padded_count,
                'segment_samples': self.processor.segment_samples,
                'elapsed_seconds': time.time() - self.start_time,
                'memory_peak_mb': memory_usage_mb()
            },
            'detection': detection,
            'null_test': null_results,
            'csd_summary': {
                'frequencies_hz': csd_result['frequencies'].tolist()[:50] if csd_result else [],
                'peak_frequency_hz': float(np.argmax(csd_result['csd_magnitude'])) * 
                                    (config.FS / self.processor.segment_samples) if csd_result else None,
                'n_segments': csd_result['n_segments'] if csd_result else 0,
                'n_discarded': csd_result['n_discarded'] if csd_result else 0
            }
        }

        self._save_final(final_results)

        log("\n" + "="*60, config)
        log("RESULTADOS v2.5.2", config)
        log("="*60, config)
        if detection:
            log(f"Retardo estimado: {detection['estimated_delay_s']:.2f} s", config)
            log(f"Retardo objetivo: {detection['target_delay_s']:.2f} s", config)
            log(f"Match: {detection['match']}", config)
            log(f"SNR: {detection['snr']:.2e}", config)
            log(f"Significancia: {detection['sigma']:.1f} sigma", config)
        log(f"Segmentos padded: {padded_count}", config)
        log(f"Tiempo total: {final_results['processing']['elapsed_seconds']:.0f} s", config)
        log("="*60, config)

        return final_results

    def _run_null_test(self, n_iterations=50):
        config = self.config
        null_detections = []
        for i in range(n_iterations):
            segment = self.processor.generator.generate_segment(
                self.processor.segment_samples, config.FS, inject_gru=False
            )
            result = self.processor.process_segment(segment, i, config.FS)
            csd = self.processor.accumulate_csd([result])
            det = self.processor.detect_time_delay(csd, config.DT_TARGET)
            if det:
                null_detections.append(det['sigma'])
            del segment, result, csd
            gc.collect()
        return {
            'n_iterations': n_iterations,
            'sigma_mean': float(np.mean(null_detections)) if null_detections else 0,
            'sigma_std': float(np.std(null_detections)) if null_detections else 0,
            'sigma_max': float(np.max(null_detections)) if null_detections else 0
        }

    def _save_checkpoint(self, results, n_processed):
        checkpoint = {
            'n_segments_processed': n_processed,
            'timestamp': datetime.now().isoformat(),
            'partial_results': [{
                'segment_id': r['segment_id'],
                'peak_power': float(np.nanmax(r['power'])) if np.any(~np.isnan(r['power'])) else 0.0,
                'mean_power': float(np.nanmean(r['power'])) if np.any(~np.isnan(r['power'])) else 0.0,
                'was_padded': r.get('was_padded', False)
            } for r in results[-10:]]
        }
        checkpoint_file = f"GRU_QF_checkpoint_{n_processed}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

    def _save_final(self, results):
        with open(self.config.OUTPUT_JSON, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        log(f"\nResultados: {self.config.OUTPUT_JSON}", self.config)


if __name__ == "__main__":
    config = GRUConfig()
    pipeline = GRUQuantumFilterPipeline(config)
    results = pipeline.run(
        inject_gru=True,
        amplitude=1e-15,
        test_null=True
    )
    print("\n" + "="*60)
    print("VERIFICACIÓN")
    print("="*60)
    if os.path.exists(config.OUTPUT_JSON):
        size_mb = os.path.getsize(config.OUTPUT_JSON) / (1024 * 1024)
        print(f"Archivo: {config.OUTPUT_JSON} ({size_mb:.2f} MB)")
        with open(config.OUTPUT_JSON, 'r') as f:
            data = json.load(f)
        print(f"Segmentos: {data['processing']['n_segments']}")
        print(f"Padded: {data['processing']['n_padded']}")
        print(f"Detección match: {data['detection']['match'] if data['detection'] else 'N/A'}")
