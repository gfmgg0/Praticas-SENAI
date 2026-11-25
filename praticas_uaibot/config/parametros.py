"""
Parâmetros geométricos, cinemáticos e de controle centralizados para as
3 práticas didáticas no UAIBot (baseado em Teste_CCR_V7.ipynb).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# =============================================================================
# 1. EXECUÇÃO E TEMPO DE AMOSTRAGEM
# =============================================================================
RUN_MODE: str = "simulation"          # "simulation" ou "real"
ROBOT_IP_A: str = "192.168.10.03"
ROBOT_IP_B: str = "192.168.10.04"

CONTROL_FREQUENCY: float = 100.0      # Hz
DT: float = 1.0 / CONTROL_FREQUENCY   # 0.01 s
ANIMATION_STEP: int = 1               # Passo de gravação de quadros no UAIBot

SHOW_COLLISION_PRIMITIVES: bool = False
ROBOT_MESH_OPACITY_WITH_COL: float = 0.25
ROBOT_MESH_OPACITY_DEFAULT: float = 1.0

# Diretório padrão para salvar simulações HTML do UAIBot
OUTPUT_DIR: str = "outputs_simulacao"

# =============================================================================
# 2. PARÂMETROS DE DISTÂNCIA (UAIBot)
# =============================================================================
# Distância diferenciável (usada na malha de controle / QP / CBF)
PARAM_H_DIST_CTRL: float = 1e-6
PARAM_EPS_DIST_CTRL: float = 1e-6
PARAM_TOL_DIST_CTRL: float = 1e-4
PARAM_NO_ITER_MAX_DIST_CTRL: int = 50

# Distância euclidiana exata (h=0, eps=0 - usada para registro e avaliação)
PARAM_H_DIST_PLOT: float = 0.0
PARAM_EPS_DIST_PLOT: float = 0.0
PARAM_TOL_DIST_PLOT: float = 1e-4
PARAM_NO_ITER_MAX_DIST_PLOT: int = 50

# =============================================================================
# 3. PARÂMETROS DO CONTROLADOR DIFERENCIAL E FILTRO QP / CBF
# =============================================================================
PARAM_K: float = 0.30                 # Ganho de convergência da função de tarefa
PARAM_LAMBDA_DLS: float = 0.08        # Amortecimento Damped Least Squares (DLS)
PARAM_TOL_ERROR: float = 0.002        # Tolerância de convergência da função de tarefa V(r)

# Ganhos das Control Barrier Functions (CBF)
PARAM_ETA_JOINT: float = 0.25         # Limites articulares
PARAM_ETA_AUTO: float = 0.05          # Autocolisão
PARAM_ETA_OBS: float = 0.20           # Obstáculos estáticos / bancada
PARAM_ETA_ROB: float = 0.20           # Colisão entre Robô A e Robô B

# Margens de segurança (delta) e raios de ativação
PARAM_JOINT_DELTA: float = float(5.0 * np.pi / 180.0)

PARAM_AUTO_DELTA: float = 0.005       # 5 mm
PARAM_AUTO_ACTIVATION: float = 0.025  # 25 mm

PARAM_OBS_DELTA: float = 0.015        # 15 mm
PARAM_OBS_ACTIVATION: float = 0.055   # 55 mm
PARAM_OBS_CONTACT_TOL: float = 1e-6
PARAM_MAX_DIST_OBS: float = 0.10      # 100 mm (cutoff de busca de colisão)

PARAM_ROB_DELTA: float = 0.015        # 15 mm
PARAM_ROB_ACTIVATION: float = 0.060   # 60 mm
PARAM_MAX_DIST_ROB: float = 0.16      # 160 mm

# Limites de velocidade articular do UR3e [rad/s]
MAX_VEL_MODULO: np.ndarray = np.array(
    [np.pi, np.pi, np.pi, 2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi],
    dtype=float
).reshape(-1, 1)

ACC_DEFAULT: float = 1.0

# =============================================================================
# 4. GEOMETRIA DA BANCADA PADRÃO E DO MANIPULADOR UR3e
# =============================================================================
H_MESA: float = 0.81                  # Altura da superfície da bancada [m]
D_MESA: float = 1.00                  # Comprimento da bancada no eixo X [m]
W_MESA: float = 0.655                 # Largura da bancada no eixo Y [m]
X_CENTRO_MESA: float = (0.66 - 0.25) / 2.0  # 0.205 m (alinhado com Teste_CCR_V7)

# Configuração articular inicial confortável (cotovelo a 90°, efetuador voltado p/ baixo)
Q0_PADRAO: np.ndarray = np.array(
    [3.0 * np.pi / 4.0, -np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0, 0.0],
    dtype=float
).reshape(-1, 1)

# Rotação padrão da base do UR3e sobre a bancada
ROT_Z_BASE_PADRAO: float = float(-3.0 * np.pi / 4.0)

# TCP padrão do UR3e no UAIBot (flange + garra RG6 = 0.20 m no eixo Z do último elo)
TCP_GARRA_OFFSET: Tuple[float, float, float] = (0.0, 0.0, 0.20)
# TCP da caneta acoplada ao efetuador (ponta 4 cm além do TCP padrão da garra)
TCP_CANETA_OFFSET: Tuple[float, float, float] = (0.0, 0.0, 0.24)


@dataclass
class ConfigBancada:
    """Parâmetros configuráveis para instanciar uma bancada didática padrão."""
    sufixo: str = "1"
    origem_bancada: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_bancada: float = 0.0
    h_mesa: float = H_MESA
    d_mesa: float = D_MESA
    w_mesa: float = W_MESA
    incluir_acessorios_padrao: bool = True
    incluir_gaiola_seguranca: bool = False
    # Posições relativas à base do robô na superfície da bancada (x, y, z_acima_da_mesa)
    pos_origem_rel: Tuple[float, float, float] = (0.34, 0.16, 0.0)
    pos_destino_rel: Tuple[float, float, float] = (0.34, -0.16, 0.0)
    pos_home_rel: Tuple[float, float, float] = (0.25, 0.02, 0.28)
    cor_mesa: str = "white"


@dataclass
class ConfigPratica1:
    """Parâmetros que os alunos podem modificar na Prática 1 (Pick-and-Place Seguro)."""
    pos_origem_rel: Tuple[float, float, float] = (0.32, 0.16, 0.0)
    pos_destino_rel: Tuple[float, float, float] = (0.32, -0.16, 0.0)
    pos_obstaculo_rel: Tuple[float, float, float] = (0.335, 0.00, 0.075)
    dim_obstaculo: Tuple[float, float, float] = (0.045, 0.045, 0.15)  # (largura/diam, prof, altura)
    dim_objeto: Tuple[float, float, float] = (0.04, 0.04, 0.10)
    altura_aproximacao: float = 0.24
    altura_transporte: float = 0.14
    usar_filtro_qp: bool = True
    executar_comparacao_sem_filtro: bool = True
    max_iter_por_etapa: int = 450
    tol_erro: float = 0.0002
    eta_obs: float = 0.25
    obs_delta: float = 0.015
    obs_activation: float = 0.045


@dataclass
class ConfigPratica2:
    """Parâmetros que os alunos podem modificar na Prática 2 (Escrita Robótica)."""
    tipo_trajetoria: str = "palavra"      # Opções: "linha", "quadrado", "circulo", "palavra", "combo"
    texto_palavra: str = "UAI"            # Palavra curta para escrita
    centro_folha_rel: Tuple[float, float, float] = (0.30, 0.00, 0.002)
    dim_folha: Tuple[float, float, float] = (0.22, 0.30, 0.003)     # A4 aproximado sobre a mesa
    escala_desenho: float = 0.06          # Tamanho característico da figura/letra [m]
    altura_escrita_rel: float = 0.005     # Altura da ponta da caneta durante o traço [m]
    altura_retracao_rel: float = 0.045    # Altura de elevação da caneta entre segmentos [m]
    pontos_por_segmento: int = 25         # Resolução da trajetória
    passos_por_ponto: int = 10            # Velocidade de rastreamento (menor = mais rápido)
    inclinacao_caneta_rad: float = 0.0    # Inclinação opcional da ferramenta em relação à vertical
    usar_filtro_qp: bool = True
    incluir_obstaculo_borda: bool = True
    pos_obstaculo_rel: Tuple[float, float, float] = (0.18, 0.16, 0.08)


@dataclass
class ConfigPratica3:
    """Parâmetros que os alunos podem modificar na Prática 3 (Colaboração entre 2 Bancadas)."""
    dy_bancada_b: float = -0.66           # Distância lateral Y entre Bancada A (0.0) e Bancada B (-0.66)
    pos_coleta_a_rel: Tuple[float, float, float] = (0.35, 0.15, 0.0)
    pos_transferencia_mundo: Tuple[float, float, float] = (0.25, -0.33, H_MESA + 0.22)
    pos_destino_b_rel: Tuple[float, float, float] = (0.28, 0.08, 0.0)
    dim_objeto: Tuple[float, float, float] = (0.04, 0.04, 0.14)
    usar_estacao_intermediaria: bool = False  # False = handover direto entre garras; True = mesa de apoio
    tol_sincronizacao_m: float = 0.025        # Tolerância espacial p/ validar handover seguro [m]
    tol_erro_tarefa: float = 0.0002
    max_iter_total: int = 3200
    max_iter_por_estado: int = 450
    usar_filtro_qp: bool = True
