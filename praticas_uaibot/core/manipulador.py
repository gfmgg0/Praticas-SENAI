"""
Módulo do manipulador robótico UR3e para simulação UAIBot e preparação
de interface para robô físico (RTDE).
"""

from typing import Optional, Tuple
import numpy as np
import uaibot as ub

from ..config.parametros import (
    MAX_VEL_MODULO,
    Q0_PADRAO,
    ROBOT_MESH_OPACITY_DEFAULT,
    ROBOT_MESH_OPACITY_WITH_COL,
    ROT_Z_BASE_PADRAO,
    SHOW_COLLISION_PRIMITIVES,
)
from .garra import FerramentaEfetuador, GarraSimulada

_URL_CACHE = {}
_ORIG_IS_URL_AVAILABLE = ub.Utils.is_url_available


def _cached_is_url_available(url: str, types: list) -> str:
    """Evita falhas intermitentes de rede ao validar repetidamente URLs de malha 3D do UAIBot."""
    if not isinstance(url, str):
        return " is not a valid url"
    ind = url.rfind(".")
    filetype = url[ind + 1:] if ind != -1 else ""
    if filetype not in types:
        return " must be an file of the types: '" + str(types) + "'"
    if "cdn.jsdelivr.net/gh/UAIbot/uaibot_data" in url:
        return "ok!"
    key = (url, tuple(types))
    if key not in _URL_CACHE:
        _URL_CACHE[key] = _ORIG_IS_URL_AVAILABLE(url, types)
    return _URL_CACHE[key]


ub.Utils.is_url_available = staticmethod(_cached_is_url_available)


def sanitize_joint_limits(joint_limit: np.ndarray) -> np.ndarray:
    """Garante limites articulares finitos (reaproveitado de Teste_CCR_V7.ipynb)."""
    jl = np.asarray(joint_limit, dtype=float).copy()
    for j in range(jl.shape[0]):
        if not np.isfinite(jl[j, 0]):
            jl[j, 0] = -1e6
        if not np.isfinite(jl[j, 1]):
            jl[j, 1] = 1e6
    return jl


class ManipuladorUR3e:
    """
    Encapsula o manipulador UR3e do UAIBot mantendo os comandos de movimentação
    desacoplados da interface gráfica do simulador (facilitando integração RTDE futura).
    """

    def __init__(
        self,
        nome: str = "robot_1",
        htm_base_mundo: Optional[np.ndarray] = None,
        q0: Optional[np.ndarray] = None,
        ferramenta: Optional[FerramentaEfetuador] = None,
        mostrar_primitivas_colisao: bool = SHOW_COLLISION_PRIMITIVES,
        opacidade: Optional[float] = None
    ):
        self.nome = nome
        if htm_base_mundo is None:
            htm_base_mundo = np.asarray(
                ub.Utils.trn([0.0, 0.0, 0.81]) @ ub.Utils.rotz(ROT_Z_BASE_PADRAO),
                dtype=float
            )
        self.htm_base_mundo = np.asarray(htm_base_mundo, dtype=float).copy()

        self.q0 = (
            np.asarray(q0, dtype=float).reshape(-1, 1).copy()
            if q0 is not None
            else Q0_PADRAO.copy()
        )
        self.q = self.q0.copy()

        self.mostrar_primitivas_colisao = mostrar_primitivas_colisao
        if opacidade is None:
            opacidade = (
                ROBOT_MESH_OPACITY_WITH_COL
                if mostrar_primitivas_colisao
                else ROBOT_MESH_OPACITY_DEFAULT
            )
        self.opacidade = opacidade

        self.robot = ub.Robot.create_ur_ur3e(
            htm=self.htm_base_mundo,
            name=self.nome,
            opacity=self.opacidade
        )

        self.limites_juntas = sanitize_joint_limits(self.robot.joint_limit)
        self.limites_velocidade = MAX_VEL_MODULO.copy()

        self.ferramenta: FerramentaEfetuador = (
            ferramenta if ferramenta is not None else GarraSimulada(nome=f"garra_{self.nome}")
        )
        self.aplicar_ferramenta(self.ferramenta)

        # Inicializa frame t=0
        self.robot.add_ani_frame(time=0.0, q=self.q0)
        self.ferramenta.resetar(self.robot, self.q0, tempo_sim=0.0)

    def aplicar_ferramenta(self, ferramenta: FerramentaEfetuador) -> None:
        """Acopla uma ferramenta (Garra ou Caneta) e atualiza o TCP (htm_n_eef) do robô."""
        from uaibot.robot.robot import _to_cpp

        self.ferramenta = ferramenta
        self.robot.set_htm_to_eef(np.matrix(ferramenta.htm_n_eef))
        if getattr(ub.os.environ, "get", lambda k: "0")("CPP_SO_FOUND") == "1":
            self.robot._cpp_robot = _to_cpp(self.robot)

    def ler_juntas(self) -> np.ndarray:
        """Retorna o vetor coluna (6, 1) da configuração articular atual."""
        return self.q.copy()

    def pose_efetuador(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Retorna a matriz de transformação homogênea (4, 4) do TCP atual."""
        q_use = self.q if q is None else np.asarray(q, dtype=float).reshape(-1, 1)
        _, htm_eef = self.robot.jac_geo(q=q_use)
        return np.asarray(htm_eef, dtype=float)

    def posicao_efetuador(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Retorna a posição cartesiana (3,) do TCP atual."""
        htm_eef = self.pose_efetuador(q=q)
        return htm_eef[0:3, 3].reshape(3)

    def aplicar_velocidade_articular(
        self,
        u: np.ndarray,
        dt: float,
        tempo_sim: Optional[float] = None,
        registrar_animacao: bool = True
    ) -> np.ndarray:
        """
        Integra a velocidade articular u com saturação de segurança e
        atualiza a animação do manipulador e da ferramenta acoplada.
        """
        u_vec = np.asarray(u, dtype=float).reshape(-1, 1)
        u_sat = np.clip(u_vec, -self.limites_velocidade, self.limites_velocidade)
        self.q = self.q + u_sat * float(dt)

        if registrar_animacao and tempo_sim is not None:
            self.robot.add_ani_frame(time=float(tempo_sim), q=self.q)
            self.ferramenta.atualizar_animacao(self.robot, self.q, tempo_sim=float(tempo_sim))

        return self.q.copy()

    def resetar(self, tempo_sim: float = 0.0) -> None:
        """Restaura o manipulador e sua ferramenta para a configuração inicial q0."""
        self.q = self.q0.copy()
        self.robot._frames = []
        self.robot._max_time = 0
        self.robot.add_ani_frame(time=float(tempo_sim), q=self.q0)
        self.ferramenta.resetar(self.robot, self.q0, tempo_sim=float(tempo_sim))
