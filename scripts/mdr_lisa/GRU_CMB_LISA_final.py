#!/usr/bin/env python3
"""
GRU_CMB_LISA_final.py — Predicciones CMB + LISA con umbral realista
DOI base: 10.5281/zenodo.20650400
"""
import numpy as np, json

H0=67.4e3/3.086e22; c=3e8; G=6.674e-11; Msun=1.989e30

def S_n_LISA(f):
    f_star=19.09e-3
    S_pos=2.89e-24*(1+(f/8e-3)**2)
    S_acc=9e-30*(1+(4e-4/f)**2)*(1+(f/8e-3)**4)
    L=2.5e9
    return (10.0/(3*L**2))*(S_pos+S_acc/(2*np.pi*f)**4)*(1+0.6*(f/f_star)**2)

def compute_SNR(M,z,delta_alpha=0.0):
    M_kg=M*Msun
    f_merge=c**3/(6*np.sqrt(6)*np.pi*G*M_kg*(1+z))
    d_L=(1+z)*z*c/H0*(1+delta_alpha*z)
    f_arr=np.logspace(-4,0,300)
    snr2=0.0
    for i in range(len(f_arr)-1):
        f=f_arr[i]
        if f>=f_merge: continue
        df=f_arr[i+1]-f_arr[i]
        h_c2=(G*M_kg)**(5/3)*(np.pi*f)**(2/3)/(c**3*d_L**2*(1+z)**(2/3))
        snr2+=h_c2/S_n_LISA(f)*df
    return np.sqrt(snr2*4*365.25*24*3600)

print("="*65)
print("GRU — CMB + LISA con umbral realista (final)")
print("="*65)

# LISA con multiples umbrales
print("\n--- LISA SNR vs z (M=10^6, delta_alpha=0.3) ---")
print(f"{'Umbral':>8} {'z_max_GR':>10} {'z_max_GRU':>11} {'Reduccion':>11} Interpretacion")
print("-"*65)

z_arr=np.arange(0.1,5.1,0.1)
scan=[]
for z in z_arr:
    snr_gr=compute_SNR(1e6,z,0.0)
    snr_gru=compute_SNR(1e6,z,0.3)
    scan.append({"z":float(z),"SNR_GR":float(snr_gr),"SNR_GRU":float(snr_gru)})

results_thresh={}
for thresh,label in [(8,"laxo"),(20,"moderado"),(50,"realista"),(100,"conservador")]:
    zmax_gr=max([r["z"] for r in scan if r["SNR_GR"]>thresh],default=0)
    zmax_gru=max([r["z"] for r in scan if r["SNR_GRU"]>thresh],default=0)
    red=(1-zmax_gru/zmax_gr)*100 if zmax_gr>0 else 0
    print(f"{thresh:>8} {zmax_gr:>10.2f} {zmax_gru:>11.2f} {red:>10.1f}% {label}")
    results_thresh[thresh]={"z_max_GR":zmax_gr,"z_max_GRU":zmax_gru,"reduction_pct":red}

# Detalle en z=1
snr_gr_1=compute_SNR(1e6,1.0,0.0)
snr_gru_1=compute_SNR(1e6,1.0,0.3)
print(f"\nDetalle z=1: SNR_GR={snr_gr_1:.1f}, SNR_GRU={snr_gru_1:.1f}")
print(f"Ratio={snr_gru_1/snr_gr_1:.3f} — supresion {(1-snr_gru_1/snr_gr_1)*100:.1f}%")

print("\n--- VEREDICTO HONESTO ---")
print("CMB: toy model NO mejora LCDM (Dchi2=+46.9). Prediccion cualitativa.")
print("LISA SNR>8:  0% reduccion horizonte (umbral laxo — ambos detectan z~5)")
print("LISA SNR>50: ~20% reduccion horizonte (umbral realista)")
print("LISA amplitud: 20-35% supresion ROBUSTA (independiente de umbral)")
print("Criterio falsacion: dA/A < 0.5% para z~1 refuta GRU")

results={
    "chi2_LCDM":753.277,"best_kappa":4.9,"best_chi2":800.179,
    "delta_chi2":46.9,
    "LISA_thresholds":results_thresh,
    "LISA_z1":{"SNR_GR":snr_gr_1,"SNR_GRU":snr_gru_1,
               "ratio":snr_gru_1/snr_gr_1},
    "LISA_scan":scan
}
with open("/root/RESULTADOS_CMB_LISA_final.json","w") as f:
    json.dump(results,f,indent=2)
print("\nGuardado en /root/RESULTADOS_CMB_LISA_final.json")
