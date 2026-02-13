# src/models/jump_diffusion.py
"""
Modelo de Merton Jump Diffusion (1976)
dS/S = (μ - λκ)dt + σ dW + (e^J - 1)dN

Donde:
- μ: drift
- σ: volatilidad
- λ: intensidad de saltos (número de saltos por año)
- J ~ N(μ_jump, σ_jump²): tamaño del salto en log-rendimiento
- κ = E[e^J - 1] = exp(μ_jump + σ_jump²/2) - 1
- dN: proceso de Poisson con intensidad λ
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional, Union
import warnings


class MertonJumpDiffusion:
    """
    Implementación del modelo de Merton (1976) con saltos
    """

    def __init__(
            self,
            S0: float = 100.0,
            mu: float = 0.05,
            sigma: float = 0.20,
            lambda_jump: float = 1.0,
            mu_jump: float = 0.0,
            sigma_jump: float = 0.10,
            seed: Optional[int] = None
    ):
        """
        Inicializar modelo Jump Diffusion

        Args:
            S0: Precio inicial
            mu: Drift anual (rendimiento esperado)
            sigma: Volatilidad anual
            lambda_jump: Intensidad de saltos (número por año)
            mu_jump: Media del tamaño del salto (en log-rendimiento)
            sigma_jump: Desviación estándar del tamaño del salto
            seed: Semilla para reproducibilidad
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump

        # Calcular κ (compensación por saltos)
        self.kappa = np.exp(mu_jump + 0.5 * sigma_jump ** 2) - 1

        # Drift ajustado por compensación de saltos
        self.drift_adjusted = mu - lambda_jump * self.kappa

        if seed is not None:
            np.random.seed(seed)

    def simulate(
            self,
            T: float = 1.0,
            n_days: int = 252,
            n_simulations: int = 1000,
            return_paths: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simular trayectorias del modelo Jump Diffusion

        Args:
            T: Horizonte temporal en años
            n_days: Número de pasos de tiempo
            n_simulations: Número de simulaciones
            return_paths: Si True, retorna todas las trayectorias
                         Si False, solo retorna precios finales

        Returns:
            paths: Array de forma (n_days + 1, n_simulations) con las trayectorias
            o final_prices: Array de forma (n_simulations,) con precios finales
        """
        dt = T / n_days
        n_steps = n_days

        # Inicializar array de precios
        prices = np.zeros((n_steps + 1, n_simulations))
        prices[0, :] = self.S0

        # Pre-calcular componentes
        drift = (self.drift_adjusted - 0.5 * self.sigma ** 2) * dt
        diffusion_scale = self.sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            # 1. Componente difusión (normal)
            diffusion = diffusion_scale * np.random.randn(n_simulations)

            # 2. Componente de saltos (Poisson compuesto)
            n_jumps = np.random.poisson(self.lambda_jump * dt, n_simulations)
            jump_sizes = np.random.normal(self.mu_jump, self.sigma_jump, n_simulations)
            jump_component = jump_sizes * n_jumps

            # 3. Retorno total
            returns = drift + diffusion + jump_component

            # 4. Evolución del precio
            prices[t, :] = prices[t - 1, :] * np.exp(returns)

        if return_paths:
            return prices
        else:
            return prices[-1, :]

    def simulate_correlated(
            self,
            other_process: 'MertonJumpDiffusion',
            correlation: float,
            T: float = 1.0,
            n_days: int = 252,
            n_simulations: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simular dos procesos Jump Diffusion correlacionados

        Args:
            other_process: Segundo proceso Jump Diffusion
            correlation: Correlación entre los movimientos brownianos
            T: Horizonte temporal
            n_days: Número de pasos
            n_simulations: Número de simulaciones

        Returns:
            (paths_1, paths_2): Trayectorias de ambos procesos
        """
        dt = T / n_days
        n_steps = n_days

        # Inicializar arrays
        prices1 = np.zeros((n_steps + 1, n_simulations))
        prices2 = np.zeros((n_steps + 1, n_simulations))
        prices1[0, :] = self.S0
        prices2[0, :] = other_process.S0

        # Matriz de correlación
        corr_matrix = np.array([[1.0, correlation], [correlation, 1.0]])
        L = np.linalg.cholesky(corr_matrix)

        # Pre-calcular componentes
        drift1 = (self.drift_adjusted - 0.5 * self.sigma ** 2) * dt
        drift2 = (other_process.drift_adjusted - 0.5 * other_process.sigma ** 2) * dt

        diffusion_scale1 = self.sigma * np.sqrt(dt)
        diffusion_scale2 = other_process.sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            # Ruido correlacionado
            Z = np.random.randn(2, n_simulations)
            correlated_Z = L @ Z

            # Proceso 1
            diffusion1 = diffusion_scale1 * correlated_Z[0, :]
            n_jumps1 = np.random.poisson(self.lambda_jump * dt, n_simulations)
            jump_sizes1 = np.random.normal(self.mu_jump, self.sigma_jump, n_simulations)
            returns1 = drift1 + diffusion1 + jump_sizes1 * n_jumps1
            prices1[t, :] = prices1[t - 1, :] * np.exp(returns1)

            # Proceso 2
            diffusion2 = diffusion_scale2 * correlated_Z[1, :]
            n_jumps2 = np.random.poisson(other_process.lambda_jump * dt, n_simulations)
            jump_sizes2 = np.random.normal(other_process.mu_jump, other_process.sigma_jump, n_simulations)
            returns2 = drift2 + diffusion2 + jump_sizes2 * n_jumps2
            prices2[t, :] = prices2[t - 1, :] * np.exp(returns2)

        return prices1, prices2

    def expected_return(self, T: float = 1.0) -> float:
        """Retorno esperado en el horizonte T"""
        return self.S0 * np.exp(self.mu * T)

    def expected_variance(self, T: float = 1.0) -> float:
        """Varianza esperada (incluye saltos)"""
        diffusion_var = self.sigma ** 2 * T
        jump_var = self.lambda_jump * (self.mu_jump ** 2 + self.sigma_jump ** 2) * T
        return diffusion_var + jump_var

    def calibrate_from_returns(
            self,
            returns: np.ndarray,
            threshold: float = 2.5,
            method: str = 'threshold'
    ) -> Dict:
        """
        Calibrar parámetros a partir de retornos históricos

        Args:
            returns: Array de retornos logarítmicos diarios
            threshold: Número de desviaciones estándar para identificar saltos
            method: Método de calibración ('threshold' o 'mle')

        Returns:
            Diccionario con parámetros calibrados
        """
        returns = returns[~np.isnan(returns)]

        if method == 'threshold':
            # Método de umbral
            mean = returns.mean()
            std = returns.std()

            # Identificar saltos
            jump_mask = np.abs(returns - mean) > threshold * std
            jump_returns = returns[jump_mask]
            normal_returns = returns[~jump_mask]

            # Calcular parámetros
            n_days = len(returns)
            n_jumps = len(jump_returns)
            years = n_days / 252

            lambda_jump = n_jumps / years if years > 0 else 0

            if n_jumps > 1:
                mu_jump = jump_returns.mean()
                sigma_jump = jump_returns.std()
            else:
                mu_jump = 0.0
                sigma_jump = 0.0

            mu_diffusion = normal_returns.mean() * 252
            sigma_diffusion = normal_returns.std() * np.sqrt(252)

        elif method == 'mle':
            # Método de máxima verosimilitud (simplificado)
            # En una implementación completa usaríamos EM algorithm
            raise NotImplementedError("MLE calibration coming soon")

        else:
            raise ValueError(f"Método {method} no soportado")

        # Actualizar parámetros
        self.mu = mu_diffusion
        self.sigma = sigma_diffusion
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.kappa = np.exp(mu_jump + 0.5 * sigma_jump ** 2) - 1
        self.drift_adjusted = mu_diffusion - lambda_jump * self.kappa

        return {
            'mu_diffusion': float(mu_diffusion),
            'sigma_diffusion': float(sigma_diffusion),
            'lambda_jump': float(lambda_jump),
            'mu_jump': float(mu_jump),
            'sigma_jump': float(sigma_jump),
            'kappa': float(self.kappa),
            'n_jumps': int(n_jumps),
            'jump_percentage': float(n_jumps / n_days * 100),
            'method': method
        }

    def calculate_var_cvar(
            self,
            T: float = 1.0,
            confidence_levels: list = [0.95, 0.99],
            n_simulations: int = 100000
    ) -> Dict:
        """
        Calcular VaR y CVaR mediante simulación

        Args:
            T: Horizonte temporal
            confidence_levels: Lista de niveles de confianza
            n_simulations: Número de simulaciones

        Returns:
            Diccionario con VaR y CVaR para cada nivel
        """
        final_prices = self.simulate(T=T, n_days=252,
                                     n_simulations=n_simulations,
                                     return_paths=False)
        returns = (final_prices / self.S0) - 1

        results = {}
        for conf in confidence_levels:
            var = np.percentile(returns, (1 - conf) * 100)
            cvar = returns[returns <= var].mean()

            results[f'var_{int(conf * 100)}'] = float(var)
            results[f'cvar_{int(conf * 100)}'] = float(cvar)

        return results

    def __repr__(self) -> str:
        return (f"MertonJumpDiffusion(S0={self.S0:.2f}, "
                f"μ={self.mu:.4f}, σ={self.sigma:.4f}, "
                f"λ={self.lambda_jump:.2f}, μ_J={self.mu_jump:.4f}, σ_J={self.sigma_jump:.4f})")


class MultivariateJumpDiffusion:
    """
    Versión multivariada del modelo Jump Diffusion
    Permite correlación entre los movimientos brownianos
    Saltos independientes por activo
    """

    def __init__(
            self,
            symbols: list,
            initial_prices: list,
            mus: list,
            sigmas: list,
            lambda_jumps: list,
            mu_jumps: list,
            sigma_jumps: list,
            correlation_matrix: np.ndarray,
            seed: Optional[int] = None
    ):
        """
        Inicializar modelo multivariado

        Args:
            symbols: Lista de símbolos
            initial_prices: Lista de precios iniciales
            mus: Lista de drifts
            sigmas: Lista de volatilidades
            lambda_jumps: Lista de intensidades de salto
            mu_jumps: Lista de medias de saltos
            sigma_jumps: Lista de volatilidades de saltos
            correlation_matrix: Matriz de correlación
            seed: Semilla para reproducibilidad
        """
        self.symbols = symbols
        self.n_assets = len(symbols)

        self.S0 = np.array(initial_prices)
        self.mu = np.array(mus)
        self.sigma = np.array(sigmas)
        self.lambda_jump = np.array(lambda_jumps)
        self.mu_jump = np.array(mu_jumps)
        self.sigma_jump = np.array(sigma_jumps)

        # Calcular κ y drift ajustado para cada activo
        self.kappa = np.exp(self.mu_jump + 0.5 * self.sigma_jump ** 2) - 1
        self.drift_adjusted = self.mu - self.lambda_jump * self.kappa

        # Matriz de correlación
        self.correlation_matrix = correlation_matrix

        # Cholesky para simulación correlacionada
        try:
            self.L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # Añadir pequeño ruido para hacerla definida positiva
            n = correlation_matrix.shape[0]
            correlation_matrix = correlation_matrix + 0.001 * np.eye(n)
            self.L = np.linalg.cholesky(correlation_matrix)
            warnings.warn("Matriz de correlación ajustada para ser definida positiva")

        if seed is not None:
            np.random.seed(seed)

    def simulate(
            self,
            T: float = 1.0,
            n_days: int = 252,
            n_simulations: int = 1000
    ) -> np.ndarray:
        """
        Simular trayectorias correlacionadas

        Args:
            T: Horizonte temporal en años
            n_days: Número de pasos
            n_simulations: Número de simulaciones

        Returns:
            paths: Array de forma (n_assets, n_days + 1, n_simulations)
        """
        dt = T / n_days
        n_steps = n_days

        # Inicializar paths
        paths = np.zeros((self.n_assets, n_steps + 1, n_simulations))
        for i in range(self.n_assets):
            paths[i, 0, :] = self.S0[i]

        # Pre-calcular componentes
        drift = (self.drift_adjusted - 0.5 * self.sigma ** 2) * dt
        diffusion_scale = self.sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            # Ruido correlacionado
            Z = np.random.randn(self.n_assets, n_simulations)
            correlated_Z = self.L @ Z

            for i in range(self.n_assets):
                # Difusión correlacionada
                diffusion = diffusion_scale[i] * correlated_Z[i, :]

                # Saltos independientes
                n_jumps = np.random.poisson(self.lambda_jump[i] * dt, n_simulations)
                jump_sizes = np.random.normal(self.mu_jump[i], self.sigma_jump[i], n_simulations)
                jump_component = jump_sizes * n_jumps

                # Retorno total
                returns = drift[i] + diffusion + jump_component

                # Evolución
                paths[i, t, :] = paths[i, t - 1, :] * np.exp(returns)

        return paths

    def get_risk_metrics(
            self,
            paths: Optional[np.ndarray] = None,
            T: float = 1.0,
            n_simulations: int = 10000
    ) -> Dict:
        """
        Calcular métricas de riesgo para todos los activos

        Args:
            paths: Trayectorias pre-simuladas (opcional)
            T: Horizonte temporal
            n_simulations: Número de simulaciones si no se proveen paths

        Returns:
            Diccionario con métricas por activo
        """
        if paths is None:
            paths = self.simulate(T=T, n_days=252, n_simulations=n_simulations)

        metrics = {}
        final_prices = paths[:, -1, :]

        for i, symbol in enumerate(self.symbols):
            returns = (final_prices[i, :] / self.S0[i]) - 1

            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = returns[returns <= var_95].mean()

            metrics[symbol] = {
                'expected_return': float(np.mean(returns)),
                'volatility': float(np.std(returns)),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95),
                'prob_loss_20': float(np.mean(returns < -0.20)),
                'prob_loss_30': float(np.mean(returns < -0.30)),
                'prob_loss_50': float(np.mean(returns < -0.50)),
                'max_loss': float(returns.min()),
                'max_gain': float(returns.max()),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns))
            }

        return metrics
