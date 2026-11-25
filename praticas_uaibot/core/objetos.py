"""
Módulo de objetos manipuláveis, obstáculos configuráveis e folha de escrita
para a bancada didática no UAIBot.
"""

from typing import List, Optional, Tuple
import numpy as np
import uaibot as ub


class ObjetoManipulavel:
    """
    Representa um objeto manipulável na bancada, diferenciando:
    - Estado visual na simulação UAIBot (frames de animação).
    - Pose física/lógica atual no mundo (htm_atual).
    - Condição de apreensão (preso_por, htm_rel_eef) e primitiva de colisão acoplada.
    """

    def __init__(
        self,
        nome: str,
        htm_inicial: np.ndarray,
        dimensoes: Tuple[float, float, float] = (0.04, 0.04, 0.10),
        cor: str = "#D2691E",
        opacidade: float = 0.9,
        formato: str = "box"
    ):
        self.nome = nome
        self.htm_inicial = np.asarray(htm_inicial, dtype=float).copy()
        self.htm_atual = np.asarray(htm_inicial, dtype=float).copy()
        self.dimensoes = dimensoes  # (width/x, depth/y, height/z)
        self.cor = cor
        self.opacidade = opacidade
        self.formato = formato.lower()

        self.preso_por: Optional[str] = None
        self.htm_rel_eef: Optional[np.ndarray] = None

        if self.formato == "cylinder":
            raio = dimensoes[0] / 2.0
            altura = dimensoes[2]
            self.obj_sim = ub.Cylinder(
                htm=self.htm_inicial,
                radius=raio,
                height=altura,
                name=self.nome,
                color=self.cor,
                opacity=self.opacidade
            )
        else:
            self.obj_sim = ub.Box(
                htm=self.htm_inicial,
                width=dimensoes[0],
                depth=dimensoes[1],
                height=dimensoes[2],
                name=self.nome,
                color=self.cor,
                opacity=self.opacidade
            )

    def atualizar_pose(self, novo_htm: np.ndarray, tempo_sim: Optional[float] = None) -> None:
        """Atualiza a pose lógica e opcionalmente registra quadro de animação no UAIBot."""
        self.htm_atual = np.asarray(novo_htm, dtype=float).copy()
        self.obj_sim._htm = np.matrix(self.htm_atual)
        if tempo_sim is not None:
            self.obj_sim.add_ani_frame(time=float(tempo_sim), htm=self.htm_atual)

    def obter_posicao(self) -> np.ndarray:
        """Retorna a posição cartesiana (3,) do centro do objeto."""
        return np.asarray(self.htm_atual[0:3, 3], dtype=float).reshape(3)

    def criar_primitiva_colisao_acoplada(self, sufixo: str) -> ub.SimObject:
        """Cria uma primitiva geométrica equivalente para acoplar ao elo final do robô."""
        nome_col = f"{self.nome}_acoplado_{sufixo}"
        if self.formato == "cylinder":
            return ub.Cylinder(
                radius=self.dimensoes[0] / 2.0,
                height=self.dimensoes[2],
                name=nome_col,
                color=self.cor
            )
        return ub.Box(
            width=self.dimensoes[0],
            depth=self.dimensoes[1],
            height=self.dimensoes[2],
            name=nome_col,
            color=self.cor
        )

    def resetar(self, tempo_sim: float = 0.0) -> None:
        """Restaura o objeto para sua posição inicial e libera estados de apreensão."""
        self.preso_por = None
        self.htm_rel_eef = None
        self.htm_atual = self.htm_inicial.copy()
        self.obj_sim._frames = []
        self.obj_sim._max_time = 0
        self.obj_sim._htm = np.matrix(self.htm_inicial)
        self.obj_sim.add_ani_frame(time=float(tempo_sim), htm=self.htm_inicial)


class ObstaculoConfiguravel:
    """Obstáculo parametrizável que pode ter sua posição alterada pelos alunos."""

    def __init__(
        self,
        nome: str,
        htm_inicial: np.ndarray,
        dimensoes: Tuple[float, float, float] = (0.04, 0.04, 0.22),
        cor: str = "#DC143C",
        opacidade: float = 0.85,
        formato: str = "box"
    ):
        self.nome = nome
        self.htm_inicial = np.asarray(htm_inicial, dtype=float).copy()
        self.htm_atual = np.asarray(htm_inicial, dtype=float).copy()
        self.dimensoes = dimensoes
        self.cor = cor
        self.opacidade = opacidade
        self.formato = formato.lower()

        if self.formato == "cylinder":
            self.obj_sim = ub.Cylinder(
                htm=self.htm_inicial,
                radius=dimensoes[0] / 2.0,
                height=dimensoes[2],
                name=self.nome,
                color=self.cor,
                opacity=self.opacidade
            )
        elif self.formato == "ball":
            self.obj_sim = ub.Ball(
                htm=self.htm_inicial,
                radius=dimensoes[0] / 2.0,
                name=self.nome,
                color=self.cor,
                opacity=self.opacidade
            )
        else:
            self.obj_sim = ub.Box(
                htm=self.htm_inicial,
                width=dimensoes[0],
                depth=dimensoes[1],
                height=dimensoes[2],
                name=self.nome,
                color=self.cor,
                opacity=self.opacidade
            )

    def alterar_posicao(self, novo_htm: np.ndarray, tempo_sim: float = 0.0) -> None:
        """Altera a posição do obstáculo no cenário."""
        self.htm_atual = np.asarray(novo_htm, dtype=float).copy()
        self.obj_sim._htm = np.matrix(self.htm_atual)
        self.obj_sim.add_ani_frame(time=float(tempo_sim), htm=self.htm_atual)

    def resetar(self, tempo_sim: float = 0.0) -> None:
        """Restaura o obstáculo à sua pose inicial."""
        self.htm_atual = self.htm_inicial.copy()
        self.obj_sim._frames = []
        self.obj_sim._max_time = 0
        self.obj_sim._htm = np.matrix(self.htm_inicial)
        self.obj_sim.add_ani_frame(time=float(tempo_sim), htm=self.htm_inicial)


class FolhaPapel:
    """
    Representa uma folha posicionada sobre a bancada e gerencia o traço
    produzido pela ferramenta de escrita ao longo do tempo usando PointCloud do UAIBot.
    """

    def __init__(
        self,
        nome: str,
        htm_centro: np.ndarray,
        dimensoes: Tuple[float, float, float] = (0.22, 0.30, 0.003),
        cor_folha: str = "#FFFDF5",
        cor_traco: str = "#0033AA",
        espessura_traco: float = 0.006
    ):
        self.nome = nome
        self.htm_inicial = np.asarray(htm_centro, dtype=float).copy()
        self.htm_atual = np.asarray(htm_centro, dtype=float).copy()
        self.dimensoes = dimensoes
        self.cor_folha = cor_folha
        self.cor_traco = cor_traco
        self.espessura_traco = espessura_traco

        self.obj_sim = ub.Box(
            htm=self.htm_inicial,
            width=dimensoes[0],
            depth=dimensoes[1],
            height=dimensoes[2],
            name=self.nome,
            color=self.cor_folha,
            opacity=1.0
        )

        # Histórico de pontos (3,) registrados quando a caneta toca a folha
        self.pontos_traco: List[np.ndarray] = []
        # Histórico de (tempo, qtd_pontos_acumulados) para animar o PointCloud
        self.historico_frames_traco: List[Tuple[float, int]] = [(0.0, 0)]

    @property
    def z_superficie(self) -> float:
        """Coordenada Z do topo da folha de papel."""
        return float(self.htm_atual[2, 3]) + self.dimensoes[2] / 2.0

    def registrar_ponto_traco(self, ponto_xyz: np.ndarray, tempo_sim: float, caneta_ativa: bool) -> None:
        """
        Registra um ponto no traço caso a caneta esteja em contato com a folha,
        atualizando o índice final do PointCloud para animação sincronizada.
        """
        if caneta_ativa:
            p = np.asarray(ponto_xyz, dtype=float).reshape(3).copy()
            # Projeta suavemente ligeiramente acima da superfície da folha para visibilidade
            p[2] = max(p[2], self.z_superficie + 0.0015)
            # Evita pontos duplicados idênticos
            if len(self.pontos_traco) == 0 or np.linalg.norm(p - self.pontos_traco[-1]) > 0.0005:
                self.pontos_traco.append(p)
        self.historico_frames_traco.append((float(tempo_sim), len(self.pontos_traco)))

    def gerar_pointcloud_animado(self, nome_pc: str = "traco_escrita") -> ub.PointCloud:
        """
        Constrói um objeto ub.PointCloud com todos os quadros de animação sincronizados
        para exibir o desenho sendo formado progressivamente no UAIBot.
        """
        if len(self.pontos_traco) == 0:
            # Ponto dummy invisível fora de vista caso nada tenha sido desenhado
            pts_mat = np.matrix([[0.0], [0.0], [-10.0]])
            pc = ub.PointCloud(name=nome_pc, points=pts_mat, size=self.espessura_traco, color=self.cor_traco)
            pc._frames = [[0.0, 0, 0]]
            return pc

        pts_array = np.column_stack(self.pontos_traco)  # 3 x N
        pts_mat = np.matrix(pts_array)
        pc = ub.PointCloud(name=nome_pc, points=pts_mat, size=self.espessura_traco, color=self.cor_traco)
        pc._frames = []
        pc._max_time = 0.0

        n_max = pts_mat.shape[1]
        for t_sim, count in self.historico_frames_traco:
            idx_fim = min(int(count), n_max)
            pc.add_ani_frame(time=float(t_sim), initial_ind=0, final_ind=idx_fim)

        return pc

    def resetar(self, tempo_sim: float = 0.0) -> None:
        """Limpa os traços registrados e reposiciona a folha."""
        self.pontos_traco.clear()
        self.historico_frames_traco = [(float(tempo_sim), 0)]
        self.htm_atual = self.htm_inicial.copy()
        self.obj_sim._frames = []
        self.obj_sim._max_time = 0
        self.obj_sim._htm = np.matrix(self.htm_inicial)
        self.obj_sim.add_ani_frame(time=float(tempo_sim), htm=self.htm_inicial)

