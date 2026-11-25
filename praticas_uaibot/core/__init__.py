"""Módulos centrais reutilizáveis para as práticas didáticas no UAIBot."""

from .bancada import BancadaPadrao, inicializar_bancada_padrao, redefinir_bancada
from .controle import calcular_erros_pose, compute_u_nominal, fun_G
from .garra import CanetaFerramenta, FerramentaEfetuador, GarraSimulada
from .manipulador import ManipuladorUR3e, sanitize_joint_limits
from .objetos import FolhaPapel, ObjetoManipulavel, ObstaculoConfiguravel
from .seguranca import (
    ParametrosSegurancaQP,
    ResultadoFiltroQP,
    apply_constraints_dual,
    apply_constraints_single,
    compute_min_obs_distance_euclidean,
    compute_min_rob_to_rob_distance_euclidean,
    compute_min_self_distance_euclidean,
    extract_robot_col_objects,
)
from .trajetorias import PontoTrajetoria, gerar_trajetoria_escrita

