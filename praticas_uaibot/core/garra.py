"""
Módulo de garras e ferramentas acopladas ao efetuador (Garra RG6 simulada e Caneta).
Separa a interface lógica de acionamento (preparada para hardware físico) da
atualização visual e do acoplamento de primitivas de colisão no UAIBot.
"""

from typing import List, Optional, Set, Tuple
import numpy as np
import uaibot as ub

from ..config.parametros import TCP_CANETA_OFFSET, TCP_GARRA_OFFSET
from .objetos import ObjetoManipulavel


class FerramentaEfetuador:
    """Classe base para ferramentas acopladas ao flange/efetuador do manipulador."""

    def __init__(
        self,
        nome: str,
        tcp_offset_xyz: Tuple[float, float, float] = TCP_GARRA_OFFSET,
        tcp_rot_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.nome = nome
        self.tcp_offset_xyz = tcp_offset_xyz
        self.tcp_rot_rpy = tcp_rot_rpy
        self.htm_n_eef = self._calcular_htm_tcp(tcp_offset_xyz, tcp_rot_rpy)

    @staticmethod
    def _calcular_htm_tcp(
        offset_xyz: Tuple[float, float, float],
        rot_rpy: Tuple[float, float, float]
    ) -> np.ndarray:
        rx, ry, rz = rot_rpy
        htm = (
            ub.Utils.trn(list(offset_xyz))
            @ ub.Utils.rotz(rz)
            @ ub.Utils.roty(ry)
            @ ub.Utils.rotx(rx)
        )
        return np.asarray(htm, dtype=float)

    def configurar_tcp(
        self,
        offset_xyz: Tuple[float, float, float],
        rot_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """Atualiza o TCP da ferramenta."""
        self.tcp_offset_xyz = offset_xyz
        self.tcp_rot_rpy = rot_rpy
        self.htm_n_eef = self._calcular_htm_tcp(offset_xyz, rot_rpy)
        return self.htm_n_eef

    def obter_objetos_visuais(self) -> List[ub.SimObject]:
        return []

    def atualizar_animacao(self, robot: ub.Robot, q: np.ndarray, tempo_sim: float) -> None:
        pass

    def resetar(self, robot: ub.Robot, q0: np.ndarray, tempo_sim: float = 0.0) -> None:
        pass


class GarraSimulada(FerramentaEfetuador):
    """
    Interface de acionamento de garra independente da implementação visual.
    Diferencia:
    1. Estado lógico da garra (aberta/fechada, abertura em mm, força em N).
    2. Condição efetiva de apreensão simulada (objeto apreendido + primitiva de colisão acoplada ao elo 5).
    3. Movimento visual do objeto apreendido e do indicador luminoso da garra.
    """

    def __init__(
        self,
        nome: str = "garra_rg6",
        tcp_offset_xyz: Tuple[float, float, float] = TCP_GARRA_OFFSET,
        abertura_max_mm: float = 110.0,
        forca_padrao_n: float = 40.0,
        tolerancia_pega_m: float = 0.08
    ):
        super().__init__(nome=nome, tcp_offset_xyz=tcp_offset_xyz)
        self.aberta: bool = True
        self.abertura_atual_mm: float = abertura_max_mm
        self.abertura_max_mm: float = abertura_max_mm
        self.forca_n: float = forca_padrao_n
        self.tolerancia_pega_m: float = tolerancia_pega_m

        self.objeto_apreendido: Optional[ObjetoManipulavel] = None
        self.nome_col_acoplado: Optional[str] = None

        # Indicador visual da garra no TCP (verde = fechada com peça, ciano = aberta)
        self.indicador_tcp = ub.Ball(
            htm=ub.Utils.trn([0.0, 0.0, 0.0]),
            radius=0.011,
            name=f"ind_{self.nome}",
            color="cyan",
            opacity=0.85
        )

    def abrir(
        self,
        robot: ub.Robot,
        q: np.ndarray,
        tempo_sim: Optional[float] = None,
        obstaculos_ignorados: Optional[Set[str]] = None
    ) -> Optional[ObjetoManipulavel]:
        """
        Aciona a abertura da garra e libera qualquer objeto apreendido,
        removendo sua primitiva de colisão do elo final do robô.
        """
        self.aberta = True
        self.abertura_atual_mm = self.abertura_max_mm
        obj_liberado = self.objeto_apreendido

        if obj_liberado is not None:
            _, htm_eef = robot.jac_geo(q=q)
            htm_eef_np = np.asarray(htm_eef, dtype=float)
            if obj_liberado.htm_rel_eef is not None:
                htm_final_obj = htm_eef_np @ obj_liberado.htm_rel_eef
                obj_liberado.atualizar_pose(htm_final_obj, tempo_sim=tempo_sim)

            obj_liberado.preso_por = None
            obj_liberado.htm_rel_eef = None

            # Remove primitiva de colisão acoplada ao elo 5 (mesmo padrão do Teste_CCR_V7.ipynb)
            self._remover_colisao_acoplada(robot)

        self.objeto_apreendido = None
        return obj_liberado

    def fechar(
        self,
        robot: ub.Robot,
        q: np.ndarray,
        objeto_candidato: Optional[ObjetoManipulavel] = None,
        tempo_sim: Optional[float] = None,
        obstaculos_ignorados: Optional[Set[str]] = None,
        forcar_apreensao: bool = False
    ) -> bool:
        """
        Aciona o fechamento da garra. Caso haja um objeto candidato dentro da
        tolerância de pega (ou forcar_apreensao=True), realiza a apreensão efetiva:
        - Registra transformação relativa efetuador -> objeto.
        - Acopla primitiva de colisão do objeto ao link 5 do robô para o filtro QP/CBF.
        - Ignora o objeto estático original na lista de obstáculos para evitar autocolisão fictícia.
        """
        self.aberta = False
        self.abertura_atual_mm = 20.0

        if objeto_candidato is None:
            return False

        _, htm_eef = robot.jac_geo(q=q)
        htm_eef_np = np.asarray(htm_eef, dtype=float)
        p_eef = htm_eef_np[0:3, 3].reshape(3)
        p_obj = objeto_candidato.obter_posicao()
        dist_pega = float(np.linalg.norm(p_eef - p_obj))

        if (not forcar_apreensao) and (dist_pega > self.tolerancia_pega_m):
            return False

        self.objeto_apreendido = objeto_candidato
        objeto_candidato.preso_por = robot.name
        objeto_candidato.htm_rel_eef = np.linalg.inv(htm_eef_np) @ objeto_candidato.htm_atual

        # Incorpora a caixa/objeto ao modelo de colisão do elo 5 do robô
        self._remover_colisao_acoplada(robot)
        htm_dh = robot.fkm(q=q, axis="dh")
        htm_link5 = np.asarray(htm_dh[5], dtype=float)
        htm_local_link5 = np.linalg.inv(htm_link5) @ objeto_candidato.htm_atual

        col_attached = objeto_candidato.criar_primitiva_colisao_acoplada(sufixo=robot.name)
        self.nome_col_acoplado = col_attached.name
        robot.links[5].col_objects.append([col_attached, np.matrix(htm_local_link5)])

        if obstaculos_ignorados is not None:
            obstaculos_ignorados.add(objeto_candidato.nome)

        if tempo_sim is not None:
            objeto_candidato.atualizar_pose(objeto_candidato.htm_atual, tempo_sim=tempo_sim)

        return True

    def _remover_colisao_acoplada(self, robot: ub.Robot) -> None:
        if self.nome_col_acoplado is None:
            return
        for idx_c, itm in enumerate(list(robot.links[5].col_objects)):
            if itm[0].name == self.nome_col_acoplado:
                robot.links[5].col_objects.pop(idx_c)
                break
        self.nome_col_acoplado = None

    def obter_objetos_visuais(self) -> List[ub.SimObject]:
        return [self.indicador_tcp]

    def atualizar_animacao(self, robot: ub.Robot, q: np.ndarray, tempo_sim: float) -> None:
        """Atualiza o indicador no TCP e movimenta o objeto apreendido junto ao efetuador."""
        _, htm_eef = robot.jac_geo(q=q)
        htm_eef_np = np.asarray(htm_eef, dtype=float)
        self.indicador_tcp.add_ani_frame(time=float(tempo_sim), htm=htm_eef_np)

        if (
            self.objeto_apreendido is not None
            and self.objeto_apreendido.preso_por == robot.name
            and self.objeto_apreendido.htm_rel_eef is not None
        ):
            htm_obj_novo = htm_eef_np @ self.objeto_apreendido.htm_rel_eef
            self.objeto_apreendido.atualizar_pose(htm_obj_novo, tempo_sim=tempo_sim)

    def resetar(self, robot: ub.Robot, q0: np.ndarray, tempo_sim: float = 0.0) -> None:
        self._remover_colisao_acoplada(robot)
        self.aberta = True
        self.abertura_atual_mm = self.abertura_max_mm
        self.objeto_apreendido = None
        _, htm_eef = robot.jac_geo(q=q0)
        self.indicador_tcp._frames = []
        self.indicador_tcp._max_time = 0
        self.indicador_tcp._htm = htm_eef
        self.indicador_tcp.add_ani_frame(time=float(tempo_sim), htm=htm_eef)


class CanetaFerramenta(FerramentaEfetuador):
    """
    Ferramenta de escrita simulada acoplada ao efetuador (Prática 2).
    Possui corpo cilíndrico, ponta esférica no TCP e relação geométrica explícita
    com o flange do manipulador, facilitando uso futuro de suporte físico de caneta.
    """

    def __init__(
        self,
        nome: str = "caneta_rob",
        tcp_offset_xyz: Tuple[float, float, float] = TCP_CANETA_OFFSET,
        comprimento_corpo: float = 0.08,
        raio_corpo: float = 0.007,
        cor_corpo: str = "#1E90FF",
        cor_ponta: str = "#00008B"
    ):
        super().__init__(nome=nome, tcp_offset_xyz=tcp_offset_xyz)
        self.comprimento_corpo = comprimento_corpo
        self.raio_corpo = raio_corpo

        self.corpo_caneta = ub.Cylinder(
            htm=ub.Utils.trn([0.0, 0.0, 0.0]),
            radius=raio_corpo,
            height=comprimento_corpo,
            name=f"corpo_{self.nome}",
            color=cor_corpo,
            opacity=0.95
        )
        self.ponta_caneta = ub.Ball(
            htm=ub.Utils.trn([0.0, 0.0, 0.0]),
            radius=raio_corpo * 0.9,
            name=f"ponta_{self.nome}",
            color=cor_ponta,
            opacity=1.0
        )

    def obter_objetos_visuais(self) -> List[ub.SimObject]:
        return [self.corpo_caneta, self.ponta_caneta]

    def _calcular_poses_visuais(self, htm_eef: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dado o HTM do TCP (ponta da caneta), retorna o HTM do corpo cilíndrico
        (recuado em -Z do TCP) e o HTM da ponta esférica.
        """
        htm_ponta = np.asarray(htm_eef, dtype=float)
        htm_corpo = htm_ponta @ np.asarray(ub.Utils.trn([0.0, 0.0, -self.comprimento_corpo / 2.0]), dtype=float)
        return htm_corpo, htm_ponta

    def atualizar_animacao(self, robot: ub.Robot, q: np.ndarray, tempo_sim: float) -> None:
        _, htm_eef = robot.jac_geo(q=q)
        htm_corpo, htm_ponta = self._calcular_poses_visuais(htm_eef)
        self.corpo_caneta.add_ani_frame(time=float(tempo_sim), htm=htm_corpo)
        self.ponta_caneta.add_ani_frame(time=float(tempo_sim), htm=htm_ponta)

    def resetar(self, robot: ub.Robot, q0: np.ndarray, tempo_sim: float = 0.0) -> None:
        _, htm_eef = robot.jac_geo(q=q0)
        htm_corpo, htm_ponta = self._calcular_poses_visuais(htm_eef)
        for obj, htm_ini in ((self.corpo_caneta, htm_corpo), (self.ponta_caneta, htm_ponta)):
            obj._frames = []
            obj._max_time = 0
            obj._htm = np.matrix(htm_ini)
            obj.add_ani_frame(time=float(tempo_sim), htm=htm_ini)

