import numpy as np
import matplotlib.pyplot as plt
import time
import os
import logging

# Configuração para evitar erros de backend no Mac
import matplotlib
matplotlib.use('Agg') 
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class EmergentResearchFramework:
    def __init__(self, output_dir="research_results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def log_header(self, case_name):
        print(f"\n{'='*60}")
        print(f" EXPERIMENT: {case_name}")
        print(f"{'='*60}")

    def compute_2pt_correlation(self, phi):
        size = phi.shape[0]
        middle = size // 2
        ref_val = phi[middle, middle]
        correlations = []
        # Mede a correlação radial para verificar a estrutura do espaço-tempo
        for r in range(1, middle):
            avg_corr = np.mean(ref_val * phi[middle, middle + r])
            correlations.append(avg_corr)
        return np.array(correlations)

    def simulate_universe_case(self, lambda_val, beta, case_id, iterations=50):
        """Executa múltiplas iterações para gerar barras de erro (Stastical Reliability)"""
        self.log_header(f"Case: {case_id}")
        size = 100
        yp_samples = []
        
        print(f"Running {iterations} Monte Carlo iterations for statistical convergence...")
        
        for _ in range(iterations):
            phi = np.random.normal(0, 1/beta, (size, size))
            # Aplica acoplamento local (Física de Rede)
            phi = (phi + np.roll(phi, 1, 0) + np.roll(phi, 1, 1)) / 3
            
            energy_density = 0.5 * phi**2 + (lambda_val/4) * phi**4
            
            n_he = np.sum(energy_density > 0.8)
            n_h = np.sum((energy_density <= 0.8) & (energy_density > 0.1))
            yp = (4 * n_he) / (4 * n_he + n_h) if (4 * n_he + n_h) > 0 else 0
            yp_samples.append(yp)

        avg_yp = np.mean(yp_samples)
        std_yp = np.std(yp_samples)
        g_r = self.compute_2pt_correlation(phi) # Última amostra para o gráfico
        
        print(f"{'Metric':<25} | {'Value':<20}")
        print(f"{'-'*48}")
        print(f"{'Mean Y_p (Helium)':<25} | {avg_yp:<20.4f}")
        print(f"{'Stat. Error (sigma)':<25} | {std_yp:<20.4f}")
        print(f"{'Planck 2018 Offset':<25} | {abs(avg_yp-0.245):<20.4f}")
        
        return avg_yp, std_yp, g_r

    def run_multi_case_study(self):
        cases = [
            {"lambda": 0.01, "beta": 1.5, "desc": "Low_Entropy"},
            {"lambda": 0.1,  "beta": 1.0, "desc": "Standard_Model"},
            {"lambda": 0.5,  "beta": 0.5, "desc": "High_Fluctuation"}
        ]
        
        results = []
        plt.figure(figsize=(14, 6))
        
        for i, config in enumerate(cases):
            avg, std, g_r = self.simulate_universe_case(config['lambda'], config['beta'], config['desc'])
            results.append((avg, std))
            
            # SUBPLOT 1: CORRELAÇÃO (USANDO RAW STRINGS r"")
            plt.subplot(1, 2, 1)
            plt.plot(g_r, label=fr"Case {i+1} ($\lambda$={config['lambda']})")
            
        plt.title(r"Two-Point Correlation Function $G(r)$")
        plt.xlabel("Lattice Distance $r$")
        plt.ylabel(r"Correlation $\langle \phi(0)\phi(r) \rangle$")
        plt.legend()
        plt.grid(alpha=0.3)

        # SUBPLOT 2: BBN BENCHMARK COM BARRAS DE ERRO
        plt.subplot(1, 2, 2)
        names = [c['desc'] for c in cases]
        means = [r[0] for r in results]
        errors = [r[1] for r in results]
        
        plt.bar(names, means, yerr=errors, capsize=7, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
        plt.axhline(0.245, color='black', linestyle='--', label='Planck 2018 Data')
        plt.title(r"Helium Fraction $Y_p$ with Statistical Errors")
        plt.ylabel(r"Mass Fraction $Y_p$")
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "academic_analysis_v2.png")
        plt.savefig(plot_path, dpi=200)
        print(f"\n[SUCCESS] Statistical analysis complete. Figures saved to: {plot_path}")

if __name__ == "__main__":
    research = EmergentResearchFramework()
    research.run_multi_case_study()
