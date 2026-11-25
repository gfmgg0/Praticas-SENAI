"""
Módulo de controle cinemático diferencial (reaproveitado de Teste_CCR_V7.ipynb).
Implementa o controlador de tarefa com inversa amortecida (DLS) e função de Lyapunov V(r).
"""

from typing import Tuple
import numpy as np
import uaibot as ub

from ..config.parametros import PARAM_K, PARAM_LAMBDA_DLS


def fun_G(_r: np.ndarray, _param_k: float = PARAM_K) -> np.ndarray:
    """Lei não-linear proporcional à raiz quadrada do erro de tarefa (Teste_CCR_V7)."""
    r = np.asarray(_r, dtype=float).reshape(-1, 1)
    return -_param_k * np.sign(r) * np.sqrt(np.abs(r))


def compute_u_nominal(
    _robot: ub.Robot,
    _q: np.ndarray,
    _htm_tg: np.ndarray,
    _lambda_dls: float = PARAM_LAMBDA_DLS,
    _param_k: float = PARAM_K
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Calcula a ação de controle articular nominal u_nominal via Damped Least Squares (DLS)
    e retorna (u_nominal, error_r, r, jac_r), preservando exatamente a formulação de Teste_CCR_V7.
    """
    _q = np.asarray(_q, dtype=float).reshape(-1, 1)

    r, jac_r = _robot.task_function(q=_q, htm_tg=np.matrix(_htm_tg))

    r = np.asarray(r, dtype=float).reshape(-1, 1)
    jac_r = np.asarray(jac_r, dtype=float)

    if jac_r.ndim == 1:
        jac_r = jac_r.reshape(1, -1)

    G = fun_G(r, _param_k)

    JJt = jac_r @ jac_r.T

    u_nominal = jac_r.T @ np.linalg.inv(
        JJt + (_lambda_dls ** 2) * np.eye(JJt.shape[0])
    ) @ G

    u_nominal = np.asarray(u_nominal, dtype=float).reshape(-1, 1)

    error_r = float(_param_k * np.sum(np.abs(r) ** (3.0 / 2.0)))

    return u_nominal, error_r, r, jac_r


def calcular_erros_pose(htm_atual: np.ndarray, htm_alvo: np.ndarray) -> Tuple[float, float]:
    """
    Calcula o erro posicional euclidiano [m] e o erro angular de orientação [rad]
    entre a pose atual do TCP e a pose desejada.
    """
    htm_a = np.asarray(htm_atual, dtype=float)
    htm_d = np.asarray(htm_alvo, dtype=float)

    erro_pos = float(np.linalg.norm(htm_a[0:3, 3] - htm_d[0:3, 3]))

    R_err = htm_d[0:3, 0:3].T @ htm_a[0:3, 0:3]
    traco = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    erro_ori = float(np.arccos(traco))

    return erro_pos, erro_ori

