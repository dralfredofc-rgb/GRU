#!/usr/bin/env python3
"""
GRU_QUANTUM_FILTER.py v2.0
Filtro cuántico basado en la estructura del espine causal (d_s=1)
Conexión: GRU-Fourier → filtro cuántico → LISA

Autor: Protocolo GRU v2.5
Fecha: 2026-06-28
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq, rfftfreq, irfft
from scipy.signal import welch, csd
import json
import argparse

# =============================================================================
# SECCIÓN 1: PARÁMETROS FÍSICOS GRU
# =============================================================================

class GRUParameters:
    """Parámetros físicos del protocolo GRU v2.5"""
    
    # Dimensiones espectrales
    D_S_FULL = 5.02           # d_s(bulk) [memoria #55]
    D_S_SPINE = 1.03          # d_s(spine)
    DELTA_D_S = 4.0           # Diferencia dimensional
    
    # Frecuencias características
    F_SPINE = 7.877e-3        # mHz [memoria #57]
    F_PLANCK = 1.855e43       # Hz (c/L_Pl)
    
    # Retardo cuántico-geométrico
    DELTA_T_GRU = 128.35       # segundos [memoria #67]
    
    # Amplitudes
    A_GRU = 4.235e-8          # Amplitud característica [memoria #57]
    A_FINAL = 1.46e-16        # Valor oficial §C.2 [memoria #64]
    
    # Constantes
    C = 299792458.0           # m/s
    L_PLANCK = 1.616255e-35   # m
    T_PLANCK = 5.391247e-44   # s
    
    # Parámetros SU(2) / espín
    J_EFF = 2.0               # Espín efectivo del modo fundamental
    
    # LISA
    LISA_ARM = 2.5e9          # m (brazo LISA)
    LISA_F_STAR = 19.09e-3    # Hz (frecuencia de transferencia)


# =============================================================================
# SECCIÓN 2: FUNCIÓN DE TRANSFERENCIA CUÁNTICA H_GRU(f)
# =============================================================================

class GRUTransferFunction:
    """
    Función de transferencia del filtro cuántico GRU.
    
    H_GRU(f) = (f/f_0)^(Δd_s/2) * exp(i * Δφ(f))
    
    donde:
    - |H|^2 ~ (f/f_0)^Δd_s  : supresión de armónicos del bulk
    - Δφ(f) = 2πf·Δt_GRU + π/2·sgn(f) : retardo de fase cuántico-geométrico
    """
    
    def __init__(self, params=None):
        self.p = params or GRUParameters()
    
    def amplitude(self, f):
        """Amplitud |H_GRU(f)| = (f/f_Pl)^(Δd_s/2) para f < f_spine"""
        f = np.atleast_1d(f)
        f_eff = np.where(np.abs(f) > 1e-300, np.abs(f), 1e-300)
        f_eff = np.where(f_eff < self.p.F_SPINE, f_eff, self.p.F_SPINE)
        return (f_eff / self.p.F_PLANCK) ** (self.p.DELTA_D_S / 2)
    
    def phase(self, f):
        """Fase acumulada: φ(f) = 2πf·Δt_GRU + π/2·sgn(f)"""
        f = np.atleast_1d(f)
        return 2 * np.pi * f * self.p.DELTA_T_GRU + np.pi/2 * np.sign(f)
    
    def complex_response(self, f):
        """Respuesta completa H_GRU(f) = |H|·exp(iφ)"""
        return self.amplitude(f) * np.exp(1j * self.phase(f))
    
    def phase_only(self, f):
        """Solo la fase de H_GRU (amplitud = 1) para simulación detectable"""
        return np.exp(1j * self.phase(f))
    
    def power(self, f):
        """Potencia del filtro |H_GRU(f)|²"""
        return self.amplitude(f) ** 2
    
    def group_delay(self, f):
        """Retardo de grupo: τ_g = Δt_GRU (constante)"""
        return np.full_like(np.atleast_1d(f), self.p.DELTA_T_GRU)


# =============================================================================
# SECCIÓN 3: GENERADOR DE SGWB CON FIRMA GRU
# =============================================================================

class GRUSGWBGenerator:
    """
    Genera señales de fondo gravitatorio estocástico (SGWB) 
    con la firma cuántico-geométrica del espine causal.
    
    Modelo físico:
    - x(t) = s(t) + n_x(t): señal directa (bulk, sin retardo)
    - y(t) = IFFT[s̃(f) · exp(i·φ_GRU(f))] + n_y(t): señal con fase GRU
    
    La fase φ_GRU(f) = 2πf·Δt_GRU + π/2 impone el retardo cuántico-geométrico.
    """
    
    def __init__(self, fs=10.0, T_obs=1.0, params=None):
        """
        Args:
            fs: Frecuencia de muestreo en Hz
            T_obs: Tiempo de observación en años
            params: Instancia de GRUParameters
        """
        self.fs = fs
        self.T_obs_years = T_obs
        self.T_obs_seconds = T_obs * 365.25 * 24 * 3600
        self.N = int(self.T_obs_seconds * fs)
        self.p = params or GRUParameters()
        self.h_tf = GRUTransferFunction(self.p)
    
    def generate_white_noise(self, amplitude=1.0):
        """Genera ruido blanco gaussiano."""
        return amplitude * np.random.randn(self.N)
    
    def generate_colored_noise(self, alpha=0, amplitude=1.0):
        """
        Genera ruido coloreado con espectro ~ f^alpha.
        alpha=0: blanco, alpha=-1: rosa, alpha=-2: rojo/Brown.
        """
        white = np.random.randn(self.N)
        white_fft = fft(white)
        freqs = fftfreq(self.N, 1.0/self.fs)
        freqs[0] = freqs[1] if self.N > 1 else 1e-10
        factor = np.abs(freqs) ** (alpha / 2)
        factor[0] = 0
        colored_fft = white_fft * factor
        colored = np.real(ifft(colored_fft))
        if np.std(colored) > 0:
            colored = colored / np.std(colored) * amplitude
        return colored
    
    def generate_sgwb_gru(self, A_sgwb=1e-15, noise_type="white"):
        """
        Genera SGWB con firma GRU.
        
        x: señal directa (bulk)
        y: señal con fase GRU (espine causal)
        """
        if noise_type == "white":
            s_base = self.generate_white_noise(amplitude=A_sgwb)
        elif noise_type == "pink":
            s_base = self.generate_colored_noise(alpha=-1, amplitude=A_sgwb)
        elif noise_type == "red":
            s_base = self.generate_colored_noise(alpha=-2, amplitude=A_sgwb)
        else:
            raise ValueError(f"Tipo de ruido desconocido: {noise_type}")
        
        # FFT de la señal base
        s_fft = fft(s_base)
        freqs = fftfreq(self.N, 1.0/self.fs)
        
        # Fase de H_GRU (amplitud normalizada a 1 para detectabilidad)
        H_phase = self.h_tf.phase_only(freqs)
        
        # x: señal directa (bulk)
        x = s_base.copy()
        
        # y: señal con fase GRU (retardo cuántico-geométrico)
        y_fft = s_fft * H_phase
        y = np.real(ifft(y_fft))
        
        return {
            "signal_base": s_base,
            "x": x,
            "y": y,
            "A_sgwb": A_sgwb,
            "noise_type": noise_type
        }
    
    def generate_lisa_noise(self, psd_type="proposal"):
        """
        Genera ruido de LISA según curva de sensibilidad.
        
        psd_type: "proposal" (L3 proposal)
        """
        freqs = rfftfreq(self.N, 1.0/self.fs)
        
        if psd_type == "proposal":
            # Parámetros LISA proposal
            S_acc = 9e-30  # m²/s⁴/Hz
            S_oms = 4e-24  # m²/Hz
            L = self.p.LISA_ARM
            
            # PSD de desplazamiento
            S_n = (S_acc / (2 * np.pi * freqs)**4 + S_oms) / L**2
            
            # Transferencia de LISA (formación triangular)
            x = 2 * np.pi * freqs * L / self.p.C
            R = 0.5 * (1 + np.cos(x)**2) * np.sinc(x / np.pi)**2
            S_n *= R
        else:
            S_n = np.ones_like(freqs) * 1e-40
        
        # Evitar infinitos
        S_n = np.where(np.isfinite(S_n) & (S_n > 0), S_n, 1e-40)
        
        # Generar ruido en frecuencia
        white_real = np.random.randn(len(freqs))
        white_imag = np.random.randn(len(freqs))
        noise_fft = np.sqrt(S_n * self.fs / 2) * (white_real + 1j * white_imag)
        noise = irfft(noise_fft, n=self.N)
        
        return noise
    
    def generate_two_arm_signal(self, A_sgwb=1e-15, noise_type="white", 
                                 lisa_noise=True):
        """
        Genera señales para dos brazos de LISA con firma GRU.
        
        x: brazo X (señal directa + ruido)
        y: brazo Y (señal con fase GRU + ruido independiente)
        """
        sgwb = self.generate_sgwb_gru(A_sgwb=A_sgwb, noise_type=noise_type)
        
        # Ruido LISA
        if lisa_noise:
            n_x = self.generate_lisa_noise()
            n_y = self.generate_lisa_noise()
        else:
            n_x = np.zeros(self.N)
            n_y = np.zeros(self.N)
        
        x = sgwb["x"] + n_x
        y = sgwb["y"] + n_y
        
        return {
            "x": x,
            "y": y,
            "signal_only_x": sgwb["x"],
            "signal_only_y": sgwb["y"],
            "noise_x": n_x,
            "noise_y": n_y,
            "A_sgwb": A_sgwb
        }


# =============================================================================
# SECCIÓN 4: DETECTOR CSD (Cross-Spectral Density)
# =============================================================================

class GRUCSDDetector:
    """
    Detector de correlación cruzada espectral (CSD) para firma GRU.
    Implementa el método que detecta el retardo de fase cuántico-geométrico.
    """
    
    def __init__(self, fs=10.0, params=None):
        self.fs = fs
        self.p = params or GRUParameters()
        self.h_tf = GRUTransferFunction(self.p)
    
    def compute_csd(self, x, y, nperseg=None, noverlap=None, 
                     window="hann", detrend="linear"):
        """
        Calcula CSD entre dos señales.
        
        S_xy(f) = <x̃(f) · ỹ*(f)>
        """
        if nperseg is None:
            nperseg = min(int(self.fs / self.p.F_SPINE * 20), len(x)//4)
        if noverlap is None:
            noverlap = nperseg // 2
        
        f, Sxy = csd(x, y, fs=self.fs, nperseg=nperseg, 
                     noverlap=noverlap, window=window, 
                     detrend=detrend)
        
        return f, Sxy
    
    def extract_phase_delay(self, f, Sxy, f_band=None):
        """
        Extrae retardo de fase de CSD en banda de frecuencia.
        
        La fase del CSD es: φ(f) = 2πf·Δt_GRU
        Pendiente: dφ/df = 2π·Δt_GRU
        Retardo: Δt_est = slope/(2π)
        """
        if f_band is None:
            # Banda alrededor de f_spine
            f_center = self.p.F_SPINE
            f_width = f_center * 0.5
            f_band = (max(f_center - f_width, 1e-6), f_center + f_width)
        
        f_min, f_max = f_band
        mask = (f >= f_min) & (f <= f_max) & (f > 0)
        
        if np.sum(mask) < 2:
            return {
                "delay_estimated": None,
                "sigma": 0.0,
                "error": "Insufficient frequency points in band",
                "f_band": f_band
            }
        
        f_band_vals = f[mask]
        Sxy_band = Sxy[mask]
        
        # Fase del CSD
        phase = np.angle(Sxy_band)
        phase_unwrapped = np.unwrap(phase)
        
        # Ajuste lineal: φ(f) = 2π·Δt·f + φ₀
        A = np.vstack([f_band_vals, np.ones_like(f_band_vals)]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, phase_unwrapped, rcond=None)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Retardo estimado: Δt = slope / (2π)
        delay_est = slope / (2 * np.pi)
        
        # Error del ajuste
        phase_fit = slope * f_band_vals + intercept
        phase_residuals = phase_unwrapped - phase_fit
        
        # Varianza de fase
        var_phase = np.var(phase_residuals)
        if var_phase > 1e-300:
            df_mean = np.mean(np.diff(f_band_vals))
            n_points = len(f_band_vals)
            sigma_delay = np.sqrt(var_phase) / (2 * np.pi * df_mean * np.sqrt(max(n_points, 1)))
            sigma = abs(delay_est - self.p.DELTA_T_GRU) / max(sigma_delay, 1e-300)
        else:
            sigma_delay = 1e-300
            sigma = 0
        
        return {
            "delay_estimated": float(delay_est),
            "delay_expected": float(self.p.DELTA_T_GRU),
            "delay_match": abs(delay_est - self.p.DELTA_T_GRU) < 0.02,
            "sigma": float(min(sigma, 1e6)),
            "sigma_delay": float(sigma_delay),
            "phase_mean": float(np.mean(phase_unwrapped)),
            "phase_std": float(np.std(phase_unwrapped)),
            "slope": float(slope),
            "intercept": float(intercept),
            "f_band": f_band,
            "n_points": int(np.sum(mask)),
            "residuals_std": float(np.std(phase_residuals))
        }
    
    def test_null_hypothesis(self, x, y, n_iterations=100):
        """
        Test de hipótesis nula: permutar una señal y recalcular CSD.
        """
        # CSD original
        f, Sxy_orig = self.compute_csd(x, y)
        result_orig = self.extract_phase_delay(f, Sxy_orig)
        
        # Distribución nula
        delays_null = []
        sigmas_null = []
        
        for _ in range(n_iterations):
            y_perm = np.random.permutation(y)
            f_n, Sxy_n = self.compute_csd(x, y_perm)
            res_n = self.extract_phase_delay(f_n, Sxy_n)
            if res_n["delay_estimated"] is not None:
                delays_null.append(res_n["delay_estimated"])
                sigmas_null.append(res_n["sigma"])
        
        delays_null = np.array(delays_null)
        sigmas_null = np.array(sigmas_null)
        
        # p-value
        if len(delays_null) > 0:
            orig_delay = result_orig.get("delay_estimated", 0)
            p_value = np.mean(np.abs(delays_null - self.p.DELTA_T_GRU) < 
                             abs(orig_delay - self.p.DELTA_T_GRU))
        else:
            p_value = 1.0
        
        return {
            "original": result_orig,
            "null_delays_mean": float(np.mean(delays_null)) if len(delays_null) > 0 else None,
            "null_delays_std": float(np.std(delays_null)) if len(delays_null) > 0 else None,
            "null_sigmas_mean": float(np.mean(sigmas_null)) if len(sigmas_null) > 0 else None,
            "p_value": float(p_value),
            "reject_null": p_value < 0.01
        }
    
    def full_analysis(self, x, y, n_null=50):
        """Análisis completo: CSD + retardo + test nulo."""
        f, Sxy = self.compute_csd(x, y)
        
        # Espectros individuales
        nperseg = min(int(self.fs / self.p.F_SPINE * 20), len(x)//4)
        noverlap = nperseg // 2
        f_x, Pxx = welch(x, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
        f_y, Pyy = welch(y, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
        
        # Retardo
        delay_result = self.extract_phase_delay(f, Sxy)
        
        # Test nulo
        null_result = self.test_null_hypothesis(x, y, n_iterations=n_null)
        
        return {
            "frequency": f.tolist(),
            "csd_real": np.real(Sxy).tolist(),
            "csd_imag": np.imag(Sxy).tolist(),
            "csd_magnitude": np.abs(Sxy).tolist(),
            "csd_phase": np.angle(Sxy).tolist(),
            "power_x": Pxx.tolist(),
            "power_y": Pyy.tolist(),
            "delay_analysis": delay_result,
            "null_hypothesis": null_result,
            "detection": {
                "detected": delay_result.get("delay_match", False) and 
                           null_result.get("reject_null", False),
                "confidence_sigma": delay_result.get("sigma", 0.0)
            }
        }


# =============================================================================
# SECCIÓN 5: PIPELINE COMPLETO
# =============================================================================

class GRUQuantumFilterPipeline:
    """
    Pipeline completo: generación → filtrado → detección → exportación.
    """
    
    def __init__(self, fs=10.0, T_obs=1/365.25, params=None):
        self.fs = fs
        self.T_obs = T_obs
        self.p = params or GRUParameters()
        self.generator = GRUSGWBGenerator(fs, T_obs, self.p)
        self.detector = GRUCSDDetector(fs, self.p)
    
    def run(self, A_sgwb=1e-15, noise_type="white", lisa_noise=True, 
            n_null=50, export_json=True, output_file=None):
        """
        Ejecuta pipeline completo.
        
        Returns:
            dict con todos los resultados
        """
        print(f"="*70)
        print(f"GRU QUANTUM FILTER PIPELINE v2.0")
        print(f"="*70)
        print(f"Parámetros:")
        print(f"  fs = {self.fs} Hz")
        print(f"  T_obs = {self.T_obs*365.25:.2f} días")
        print(f"  N_samples = {self.generator.N}")
        print(f"  A_sgwb = {A_sgwb}")
        print(f"  f_spine = {self.p.F_SPINE} Hz")
        print(f"  Δt_GRU = {self.p.DELTA_T_GRU} s")
        print(f"  Δd_s = {self.p.DELTA_D_S}")
        print(f"-"*70)
        
        # 1. Generar señales
        print("[1/3] Generando señales SGWB con firma GRU...")
        signals = self.generator.generate_two_arm_signal(
            A_sgwb=A_sgwb, noise_type=noise_type, lisa_noise=lisa_noise
        )
        
        # 2. Análisis CSD
        print("[2/3] Calculando CSD y extrayendo retardo de fase...")
        analysis = self.detector.full_analysis(
            signals["x"], signals["y"], n_null=n_null
        )
        
        delay = analysis["delay_analysis"]
        print(f"  Retardo estimado: {delay.get('delay_estimated', 'N/A'):.4f} s")
        print(f"  Retardo esperado: {delay.get('delay_expected', 'N/A')} s")
        print(f"  Match: {delay.get('delay_match', False)}")
        print(f"  Significancia: {delay.get('sigma', 0):.2f} sigma")
        
        # 3. Test nulo
        print("[3/3] Test de hipótesis nula...")
        null = analysis["null_hypothesis"]
        print(f"  p-value: {null.get('p_value', 1.0):.4f}")
        print(f"  Rechaza nula: {null.get('reject_null', False)}")
        
        # Resumen
        detected = analysis["detection"]["detected"]
        confidence = analysis["detection"]["confidence_sigma"]
        print(f"-"*70)
        print(f"RESULTADO: {'DETECCIÓN GRU ✓' if detected else 'NO DETECTADO ✗'}")
        print(f"Confianza: {confidence:.2f} sigma")
        print(f"="*70)
        
        result = {
            "metadata": {
                "version": "2.0",
                "date": "2026-06-28",
                "fs": self.fs,
                "T_obs_years": self.T_obs,
                "N_samples": self.generator.N,
                "parameters": {
                    "f_spine": self.p.F_SPINE,
                    "delta_t_gru": self.p.DELTA_T_GRU,
                    "delta_d_s": self.p.DELTA_D_S,
                    "A_sgwb": A_sgwb,
                    "noise_type": noise_type,
                    "lisa_noise": lisa_noise
                }
            },
            "signals": {
                "signal_power_x": float(np.mean(signals["signal_only_x"]**2)),
                "signal_power_y": float(np.mean(signals["signal_only_y"]**2)),
                "noise_power_x": float(np.mean(signals["noise_x"]**2)),
                "noise_power_y": float(np.mean(signals["noise_y"]**2))
            },
            "analysis": analysis,
            "detection": analysis["detection"]
        }
        
        if export_json:
            filename = output_file or "GRU_QUANTUM_FILTER_result.json"
            with open(filename, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResultados guardados en: {filename}")
        
        return result


# =============================================================================
# SECCIÓN 6: MAIN / CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GRU Quantum Filter Pipeline")
    parser.add_argument("--fs", type=float, default=10.0, help="Frecuencia de muestreo (Hz)")
    parser.add_argument("--T-obs", type=float, default=1/365.25, help="Tiempo de observación (años)")
    parser.add_argument("--A-sgwb", type=float, default=1e-12, help="Amplitud SGWB")
    parser.add_argument("--noise-type", type=str, default="white", choices=["white", "pink", "red"])
    parser.add_argument("--lisa-noise", action="store_true", help="Incluir ruido LISA")
    parser.add_argument("--n-null", type=int, default=50, help="Iteraciones test nulo")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--output", type=str, default=None, help="Archivo de salida JSON")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    pipeline = GRUQuantumFilterPipeline(
        fs=args.fs,
        T_obs=args.T_obs
    )
    
    result = pipeline.run(
        A_sgwb=args.A_sgwb,
        noise_type=args.noise_type,
        lisa_noise=args.lisa_noise,
        n_null=args.n_null,
        export_json=True,
        output_file=args.output
    )
    
    return result


if __name__ == "__main__":
    result = main()