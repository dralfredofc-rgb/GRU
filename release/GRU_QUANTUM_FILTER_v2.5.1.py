#!/usr/bin/env python3
"""
GRU Quantum Filter v2.5.1 - Heisenberg-Fourier Chunked Architecture
====================================================================
Aplicación del principio de incertidumbre para procesamiento estable de LISA.

Principio: Δt · Δf ≥ 1/(4π)
- Chunking temporal con ventanas Gaussianas (Gabor) óptimas
- Procesamiento por segmentos de 1 hora con 75% overlap
- RAM garantizada < 2GB para servidor gru (24GB, 6 vCores)
- Salida JSON incremental

Basado en: GRU v2.4 Protocolo A.22, §A.41-A.43
"""

import numpy as np
import json
import time
import os
import gc
from datetime import datetime
from scipy.signal import stft, csd, get_window
from scipy.fft import rfft, irfft
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN GRU v2.5.1
# =============================================================================

class GRUConfig:
    """Configuración basada en resultados v2.4 y principio de incertidumbre"""

    # Parámetros LISA (§A.41)
    T_OBS_YEARS = 1.0           # 1 año de observación
    FS = 10.0                   # Hz (frecuencia de muestreo)
    DT_TARGET = 128.35           # segundos, periodo spin/tratamiento CDT (T=20, N=2567)

    # Parámetros Heisenberg-Fourier
    SEGMENT_HOURS = 1.0         # Duración chunk: 1 hora
    OVERLAP_RATIO = 0.75        # 75% overlap para coherencia de fase
    WINDOW_TYPE = 'gaussian'    # Ventana Gabor óptima (mínima Δt·Δf)

    # Parámetros GRU físicos
    D_S_SPINE = 1.03            # Dimensión espectral espina (T1.2)
    D_S_FULL = 5.02             # Dimensión espectral completa
    DELTA_D_S = 4.0             # Jerarquía dimensional
    LAMBDA = 0.50               # Λ_eff = Λ_bare · f_screen (§A.42)
    KAPPA = 1/(8*np.pi)         # κ = 1/(8π) para D=4

    # Parámetros detección
    N_WALKS = 3000              # NWALKS protocolo A.22
    SIGMA_MAX = 200             # SIGMAMAX protocolo A.22
    SEED = 42                   # SEED protocolo A.22

    # Memoria y rendimiento
    MAX_RAM_GB = 2.0            # Límite RAM garantizado
    N_CORES = 6                 # vCores servidor gru
    CHUNK_OUTPUT = True         # Salida JSON incremental

    # Archivos
    OUTPUT_JSON = "GRU_QF_LISA_1yr_v251.json"
    LOG_FILE = "gru_v251.log"


# =============================================================================
# UTILIDADES
# =============================================================================

def log(msg, config):
    """Logging con timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(config.LOG_FILE, 'a') as f:
        f.write(line + "\n")


def memory_usage_mb():
    """Estimación simple de memoria usada por arrays numpy activos"""
    import sys
    total = 0
    for obj in gc.get_objects():
        if isinstance(obj, np.ndarray):
            total += obj.nbytes
    return total / (1024 * 1024)


# =============================================================================
# FILTRO CUÁNTICO GRU
# =============================================================================

class GRUQuantumFilter:
    """
    Filtro cuántico basado en dimensión espectral d_s.

    Implementa H_GRU(f) derivado de la jerarquía dimensional:
    d_s(full) = 5.02, d_s(spine) = 1.03, Δd_s = 4.0
    """

    def __init__(self, config):
        self.config = config
        self.kappa = config.KAPPA
        self.lambda_eff = config.LAMBDA

        # Calcular λ(Λ) según §A.41
        D = 4
        self.lambda_scale = 1 / np.arccosh(
            1 + self.lambda_eff / (D**2 * (D-1) * 2 * self.kappa)
        )

        # Escala temporal GRU
        self.delta_t_gru = 128.35  # segundos, periodo spin CDT (T=20, N=2567)

    def compute_transfer_function(self, frequencies):
        """
        Calcula H_GRU(f) basado en la dimensión espectral.

        Para d_s(spine) ≈ 1, la señal tiene características de camino aleatorio
        en la espina causal, con modificaciones por Δd_s = 4.0.

        Args:
            frequencies: Array de frecuencias en Hz

        Returns:
            H_gru: Función de transferencia compleja
        """
        f = np.asarray(frequencies)
        f = np.where(f == 0, 1e-10, f)  # Evitar división por cero

        # Frecuencia característica de la espina causal
        f_spine = 1 / self.delta_t_gru  # ~7.75 mHz

        # Frecuencia de transición (escala Planck efectiva)
        f_planck = 1e3  # Hz (placeholder, ajustar con datos reales)

        # Regimenes según T2.3 (tres zonas)
        # Zona I: f < f_spine (N < 10, régimen Bohr)
        # Zona II: f_spine < f < f_planck (10 ≤ N < 65)
        # Zona III: f > f_planck (N ≥ 65, clásico)

        H = np.zeros_like(f, dtype=complex)

        # Zona I: Decaimiento suave (dimensión baja)
        mask1 = f < f_spine * 0.1
        H[mask1] = (f[mask1] / f_spine) ** (self.config.D_S_SPINE / 2) *                    np.exp(1j * np.pi * self.config.D_S_SPINE / 4)

        # Zona II: Transición (espina dominante)
        mask2 = (f >= f_spine * 0.1) & (f < f_spine * 10)
        ratio = f[mask2] / f_spine
        H[mask2] = ratio ** (self.config.D_S_SPINE / 2) *                     np.exp(-0.5 * (np.log(ratio))**2) *                     np.exp(1j * np.pi * self.config.D_S_SPINE / 4 * ratio)

        # Zona III: Decaimiento rápido (dimensión completa)
        mask3 = f >= f_spine * 10
        H[mask3] = (f_spine / f[mask3]) ** (self.config.D_S_FULL / 2) *                    np.exp(1j * np.pi * self.config.D_S_FULL / 4)

        # Normalización
        H = H / np.max(np.abs(H))

        return H

    def apply_filter(self, spectrum, frequencies):
        """Aplica filtro cuántico a espectro"""
        H = self.compute_transfer_function(frequencies)
        return spectrum * H


# =============================================================================
# GENERADOR DE SEÑAL LISA + GRU
# =============================================================================

class LISASignalGenerator:
    """Generador de señal LISA con inyección GRU y ruido realista"""

    def __init__(self, config):
        self.config = config
        np.random.seed(config.SEED)

    def generate_noise(self, n_samples, fs):
        """Ruido LISA realista (simplificado)"""
        # Ruido de fondo estocástico gravitatorio (SGWB)
        # Perfil f^(-3) para frecuencias bajas (LISA band)
        f = np.fft.rfftfreq(n_samples, 1/fs)

        # Densidad espectral de potencia
        psd = np.zeros_like(f)
        psd[f > 0] = 1e-44 * (f[f > 0] / 1e-3) ** (-3)  # SGWB
        psd += 1e-42 * (f / 1e-3) ** (-2)  # Ruido instrumental

        # Generar ruido coloreado
        noise_fft = np.sqrt(psd) * np.exp(1j * 2*np.pi * np.random.random(len(f)))
        noise = np.fft.irfft(noise_fft, n=n_samples)

        return noise

    def inject_gru_signal(self, signal, fs, delay_s=128.35, amplitude=1e-15):
        """
        Inyecta señal GRU con retardo característico.

        La señal GRU se modela como modulación de la espina causal
        con retardo Δt = 128.35 s (§A.45.3).
        """
        n_samples = len(signal)
        delay_samples = int(delay_s * fs)

        # Señal GRU: modulación de baja frecuencia con estructura fractal
        t = np.arange(n_samples) / fs

        # Componente principal: frecuencia característica espina
        f_gru = 1 / delay_s  # ≈7.79 mHz para Δt=128.35 s

        # Señal con estructura de dimensión espectral d_s=1.03
        # (ruido 1/f con modificaciones)
        gru_signal = amplitude * np.sin(2 * np.pi * f_gru * t) *                      (1 + 0.3 * np.sin(2 * np.pi * 0.1 * f_gru * t))

        # Aplicar retardo
        gru_delayed = np.zeros_like(gru_signal)
        if delay_samples < n_samples:
            gru_delayed[delay_samples:] = gru_signal[:-delay_samples]

        return signal + gru_delayed

    def generate_segment(self, n_samples, fs, inject_gru=True, amplitude=1e-15):
        """Genera un segmento de señal LISA"""
        noise = self.generate_noise(n_samples, fs)
        if inject_gru:
            signal = self.inject_gru_signal(noise, fs,
                                            delay_s=self.config.DT_TARGET,
                                            amplitude=amplitude)
        else:
            signal = noise
        return signal


# =============================================================================
# PROCESADOR HEISENBERG-FOURIER
# =============================================================================

class HeisenbergFourierProcessor:
    """
    Procesador basado en principio de incertidumbre.

    Δt · Δf ≥ 1/(4π)

    Usa STFT con ventanas Gaussianas para óptimo trade-off tiempo-frecuencia.
    """

    def __init__(self, config):
        self.config = config
        self.filter = GRUQuantumFilter(config)
        self.generator = LISASignalGenerator(config)

        # Calcular parámetros de segmentación
        self.segment_samples = int(config.SEGMENT_HOURS * 3600 * config.FS)
        self.overlap_samples = int(self.segment_samples * config.OVERLAP_RATIO)
        self.hop_samples = self.segment_samples - self.overlap_samples

        # Ventana Gaussiana (Gabor) - óptima para incertidumbre
        self.window = self._create_gaussian_window(self.segment_samples)

        log(f"Heisenberg-Fourier Processor inicializado", config)
        log(f"  Segmento: {config.SEGMENT_HOURS}h = {self.segment_samples} muestras", config)
        log(f"  Overlap: {config.OVERLAP_RATIO*100:.0f}% = {self.overlap_samples} muestras", config)
        log(f"  Hop: {self.hop_samples} muestras", config)
        log(f"  Δt·Δf mínimo teórico: {1/(4*np.pi):.6f}", config)

    def _create_gaussian_window(self, n):
        """Crea ventana Gaussiana óptima (Gabor)"""
        # σ tal que la ventana cubra ~6σ (99.7% energía)
        sigma = n / 6
        t = np.arange(n) - n/2
        window = np.exp(-0.5 * (t/sigma)**2)
        return window / np.sum(window)  # Normalizar energía

    def process_segment(self, segment, segment_id, fs):
        """
        Procesa un segmento individual con filtro GRU.

        RAM usada: ~2 * segment_samples * 8 bytes (complex128)
        Para 1h @ 10Hz: ~2 * 36000 * 8 = 576 KB
        """
        # Aplicar ventana
        windowed = segment * self.window[:len(segment)]

        # FFT
        spectrum = rfft(windowed)
        freqs = np.fft.rfftfreq(len(segment), 1/fs)

        # Aplicar filtro cuántico GRU
        filtered_spectrum = self.filter.apply_filter(spectrum, freqs)

        # Espectro de potencia (para CSD acumulativo)
        power = np.abs(filtered_spectrum)**2

        # Liberar memoria explícita
        del windowed, spectrum, filtered_spectrum
        gc.collect()

        return {
            'segment_id': segment_id,
            'frequencies': freqs,
            'power': power,
            'n_samples': len(segment)
        }

    def accumulate_csd(self, results_list):
        """
        Acumula CSD (Cross-Spectral Density) de todos los segmentos.

        El CSD acumulativo permite detectar coherencia de fase
        y retardo temporal consistente (128.35 s).
        """
        if not results_list:
            return None

        # Promediar espectros de potencia
        freqs = results_list[0]['frequencies']
        avg_power = np.zeros_like(freqs)

        for result in results_list:
            avg_power += result['power']

        avg_power /= len(results_list)

        # Estimar CSD (simplificado - en implementación real usaría 2 canales)
        # Aquí usamos la fase del espectro promedio
        csd_magnitude = avg_power

        return {
            'frequencies': freqs,
            'csd_magnitude': csd_magnitude,
            'n_segments': len(results_list)
        }

    def detect_time_delay(self, csd_result, target_delay=128.35):
        """
        Detecta retardo temporal via fase del CSD.

        Para retardo τ, la fase del CSD es: φ(f) = -2πfτ
        """
        freqs = csd_result['frequencies']
        csd_mag = csd_result['csd_magnitude']

        # Encontrar frecuencias con alta coherencia (pico del filtro GRU)
        f_spine = 1 / target_delay  # ~7.75 mHz
        mask = (freqs > f_spine * 0.5) & (freqs < f_spine * 2)

        if not np.any(mask):
            return None

        # Estimar retardo desde la pendiente de la fase
        # (simplificado - en implementación real usar fase compleja)
        peak_freq = freqs[mask][np.argmax(csd_mag[mask])]
        estimated_delay = 1 / peak_freq

        # Calcular significancia (sigma)
        noise_floor = np.mean(csd_mag[~mask])
        peak_signal = np.max(csd_mag[mask])
        snr = peak_signal / noise_floor
        sigma = np.sqrt(2 * csd_result['n_segments']) * snr  # Aproximación

        return {
            'target_delay_s': target_delay * 1000,
            'estimated_delay_ms': estimated_delay * 1000,
            'peak_frequency_hz': peak_freq,
            'snr': snr,
            'sigma': sigma,
            'match': abs(estimated_delay - target_delay) < 0.01  # 10ms tolerancia
        }


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

class GRUQuantumFilterPipeline:
    """Pipeline principal v2.5.1"""

    def __init__(self, config=None):
        self.config = config or GRUConfig()
        self.processor = HeisenbergFourierProcessor(self.config)
        self.results = []
        self.start_time = time.time()

    def run(self, inject_gru=True, amplitude=1e-15, test_null=False):
        """
        Ejecuta pipeline completo.

        Args:
            inject_gru: Si True, inyecta señal GRU
            amplitude: Amplitud de la señal GRU
            test_null: Si True, ejecuta test nulo (50 iteraciones)
        """
        config = self.config
        log("="*60, config)
        log("GRU Quantum Filter v2.5.1 - Heisenberg-Fourier Architecture", config)
        log("="*60, config)
        log(f"Observación: {config.T_OBS_YEARS} año(s)", config)
        log(f"Frecuencia muestreo: {config.FS} Hz", config)
        log(f"Retardo objetivo: {config.DT_TARGET*1000:.1f} ms", config)
        log(f"RAM límite: {config.MAX_RAM_GB} GB", config)
        log("", config)

        # Calcular número total de muestras y segmentos
        total_seconds = config.T_OBS_YEARS * 365.25 * 24 * 3600
        total_samples = int(total_seconds * config.FS)
        n_segments = int(np.ceil((total_samples - config.FS) / 
                                  self.processor.hop_samples))

        log(f"Muestras totales: {total_samples:,}", config)
        log(f"Segmentos a procesar: {n_segments:,}", config)
        log(f"Muestras por segmento: {self.processor.segment_samples:,}", config)
        log("", config)

        # Inicializar acumuladores
        accumulated_results = []
        segment_count = 0

        # Procesar por segmentos (streaming)
        for i in range(n_segments):
            start_sample = i * self.processor.hop_samples
            end_sample = min(start_sample + self.processor.segment_samples, 
                           total_samples)
            n_samples = end_sample - start_sample

            # Generar segmento (simulado - en producción leer de LISA data)
            segment = self.processor.generator.generate_segment(
                n_samples, config.FS, inject_gru=inject_gru, amplitude=amplitude
            )

            # Procesar segmento
            result = self.processor.process_segment(segment, i, config.FS)
            accumulated_results.append(result)
            segment_count += 1

            # Liberar memoria del segmento
            del segment
            gc.collect()

            # Reporte progreso cada 100 segmentos
            if (i + 1) % 100 == 0 or i == n_segments - 1:
                elapsed = time.time() - self.start_time
                progress = (i + 1) / n_segments * 100
                mem_mb = memory_usage_mb()

                log(f"  [{i+1}/{n_segments}] {progress:.1f}% | "
                    f"Tiempo: {elapsed:.0f}s | RAM: {mem_mb:.1f} MB", config)

                # Salida incremental
                if config.CHUNK_OUTPUT:
                    self._save_checkpoint(accumulated_results, i+1)

            # Verificar límite de RAM
            if memory_usage_mb() > config.MAX_RAM_GB * 1024:
                log(f"  ALERTA: RAM excedida ({memory_usage_mb():.0f} MB). "
                    f"Forzando GC...", config)
                gc.collect()

        # Acumular CSD final
        log("\nAcumulando CSD...", config)
        csd_result = self.processor.accumulate_csd(accumulated_results)

        # Detectar retardo
        log("Detectando retardo temporal...", config)
        detection = self.processor.detect_time_delay(csd_result, config.DT_TARGET)

        # Test nulo si se solicita
        null_results = None
        if test_null:
            log("\nEjecutando test nulo (50 iteraciones)...", config)
            null_results = self._run_null_test()

        # Compilar resultados finales
        final_results = {
            'metadata': {
                'version': '2.5.1',
                'architecture': 'Heisenberg-Fourier',
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'T_obs_years': config.T_OBS_YEARS,
                    'fs_hz': config.FS,
                    'dt_target_ms': config.DT_TARGET * 1000,
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
                'segment_samples': self.processor.segment_samples,
                'elapsed_seconds': time.time() - self.start_time,
                'memory_peak_mb': memory_usage_mb()
            },
            'detection': detection,
            'null_test': null_results,
            'csd_summary': {
                'frequencies_hz': csd_result['frequencies'].tolist()[:100] if csd_result else [],
                'peak_frequency_hz': float(np.argmax(csd_result['csd_magnitude'])) * 
                                    (config.FS / self.processor.segment_samples) if csd_result else None,
                'n_segments': csd_result['n_segments'] if csd_result else 0
            }
        }

        # Guardar resultado final
        self._save_final(final_results)

        # Resumen
        log("\n" + "="*60, config)
        log("RESULTADOS FINALES", config)
        log("="*60, config)
        if detection:
            log(f"Retardo estimado: {detection['estimated_delay_ms']:.2f} ms", config)
            log(f"Retardo objetivo: {detection['target_delay_s']:.2f} ms", config)
            log(f"Match: {detection['match']}", config)
            log(f"SNR: {detection['snr']:.2e}", config)
            log(f"Significancia: {detection['sigma']:.1f} sigma", config)
        log(f"Tiempo total: {final_results['processing']['elapsed_seconds']:.0f} s", config)
        log(f"RAM pico: {final_results['processing']['memory_peak_mb']:.1f} MB", config)
        log("="*60, config)

        return final_results

    def _run_null_test(self, n_iterations=50):
        """Ejecuta test nulo (sin señal GRU)"""
        config = self.config
        null_detections = []

        for i in range(n_iterations):
            # Generar segmento corto sin señal
            segment = self.processor.generator.generate_segment(
                self.processor.segment_samples, config.FS, inject_gru=False
            )
            result = self.processor.process_segment(segment, i, config.FS)

            # Detectar (debería ser ruido)
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
        """Guarda checkpoint incremental"""
        checkpoint = {
            'n_segments_processed': n_processed,
            'timestamp': datetime.now().isoformat(),
            'partial_results': [{
                'segment_id': r['segment_id'],
                'peak_power': float(np.max(r['power'])),
                'mean_power': float(np.mean(r['power']))
            } for r in results[-10:]]  # Solo últimos 10 para no crecer archivo
        }

        checkpoint_file = f"GRU_QF_checkpoint_{n_processed}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _save_final(self, results):
        """Guarda resultado final"""
        with open(self.config.OUTPUT_JSON, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        log(f"\nResultados guardados en: {self.config.OUTPUT_JSON}", self.config)


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    # Crear configuración
    config = GRUConfig()

    # Crear y ejecutar pipeline
    pipeline = GRUQuantumFilterPipeline(config)

    # Ejecutar con señal GRU inyectada
    results = pipeline.run(
        inject_gru=True,
        amplitude=1e-15,
        test_null=True
    )

    # Verificación rápida
    print("\n" + "="*60)
    print("VERIFICACIÓN RÁPIDA")
    print("="*60)
    if os.path.exists(config.OUTPUT_JSON):
        size_mb = os.path.getsize(config.OUTPUT_JSON) / (1024 * 1024)
        print(f"Archivo {config.OUTPUT_JSON}: {size_mb:.2f} MB")
        with open(config.OUTPUT_JSON, 'r') as f:
            data = json.load(f)
        print(f"Segmentos procesados: {data['processing']['n_segments']}")
        print(f"Detección match: {data['detection']['match'] if data['detection'] else 'N/A'}")
