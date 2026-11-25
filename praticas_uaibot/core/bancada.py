"""
Módulo da Bancada Didática Padrão e Modular para todas as práticas no UAIBot.
Reaproveita a geometria real da bancada de Teste_CCR_V7.ipynb e fornece
transformações consistentes de referenciais, objetos, obstáculos, regiões de
operação/segurança e funções de inicialização e redefinição (reset).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import uaibot as ub

from ..config.parametros import (
    ConfigBancada,
    H_MESA,
    Q0_PADRAO,
    ROT_Z_BASE_PADRAO,
    SHOW_COLLISION_PRIMITIVES,
    X_CENTRO_MESA,
)
from .garra import CanetaFerramenta, FerramentaEfetuador, GarraSimulada
from .manipulador import ManipuladorUR3e
from .objetos import FolhaPapel, ObjetoManipulavel, ObstaculoConfiguravel


class BancadaPadrao:
    """
    Representa uma bancada didática completa e parametrizável contendo:
    - Manipulador robótico UR3e + Garra ou Caneta acoplada.
    - Superfície da bancada e acessórios padrão (rampa, suportes cilíndricos).
    - Referenciais de coordenadas (base, origem de coleta, destino e alvo ativo).
    - Regiões de operação e segurança.
    - Objetos manipuláveis, obstáculos configuráveis e folha de escrita.
    """

    def __init__(
        self,
        config: Optional[ConfigBancada] = None,
        ferramenta: Optional[FerramentaEfetuador] = None,
        q0: Optional[np.ndarray] = None,
        mostrar_primitivas_colisao: bool = SHOW_COLLISION_PRIMITIVES
    ):
        self.config = config if config is not None else ConfigBancada()
        self.sufixo = self.config.sufixo

        ox, oy, oz = self.config.origem_bancada
        self.yaw = float(self.config.yaw_bancada)
        self.z_superficie = float(oz + self.config.h_mesa)

        # Referencial da superfície da bancada sob a base do robô (x=ox, y=oy, z=z_superficie)
        self.htm_referencial_bancada = np.asarray(
            ub.Utils.trn([ox, oy, self.z_superficie]) @ ub.Utils.rotz(self.yaw),
            dtype=float
        )

        # Base do UR3e rotacionada conforme modelo digital de referência (-3*pi/4)
        self.htm_base_robo = np.asarray(
            self.htm_referencial_bancada @ ub.Utils.rotz(ROT_Z_BASE_PADRAO),
            dtype=float
        )

        # 1. Superfície da bancada e acessórios fixos (idênticos ao Teste_CCR_V7.ipynb)
        self.objetos_fixos_bancada: List[ub.SimObject] = self._criar_geometria_bancada()

        # 2. Paredes/região de segurança opcional
        self.paredes_seguranca: List[ub.SimObject] = (
            self._criar_gaiola_seguranca() if self.config.incluir_gaiola_seguranca else []
        )

        # 3. Manipulador UR3e e ferramenta
        self.manipulador = ManipuladorUR3e(
            nome=f"robot_{self.sufixo}",
            htm_base_mundo=self.htm_base_robo,
            q0=q0 if q0 is not None else Q0_PADRAO.copy(),
            ferramenta=ferramenta,
            mostrar_primitivas_colisao=mostrar_primitivas_colisao
        )

        # 4. Poses padrão de Home, Origem e Destino no referencial da bancada
        self.htm_home = self.pose_rel_para_mundo(
            self.config.pos_home_rel,
            rot_local=np.asarray(ub.Utils.rotx(np.pi), dtype=float)
        )
        self.htm_origem = self.pose_rel_para_mundo(
            self.config.pos_origem_rel,
            rot_local=np.asarray(ub.Utils.rotx(np.pi), dtype=float)
        )
        self.htm_destino = self.pose_rel_para_mundo(
            self.config.pos_destino_rel,
            rot_local=np.asarray(ub.Utils.rotx(np.pi), dtype=float)
        )

        # 5. Referenciais de coordenadas e marcadores visuais
        self.frame_bancada = ub.Frame(htm=self.htm_referencial_bancada, size=0.10)
        self.frame_alvo = ub.Frame(htm=self.htm_home, size=0.09)
        self.marcador_origem = ub.Cylinder(
            htm=self.pose_rel_para_mundo((self.config.pos_origem_rel[0], self.config.pos_origem_rel[1], 0.001)),
            radius=0.032,
            height=0.002,
            name=f"regiao_origem_{self.sufixo}",
            color="#32CD32",
            opacity=0.65
        )
        self.marcador_destino = ub.Cylinder(
            htm=self.pose_rel_para_mundo((self.config.pos_destino_rel[0], self.config.pos_destino_rel[1], 0.001)),
            radius=0.032,
            height=0.002,
            name=f"regiao_destino_{self.sufixo}",
            color="#FF8C00",
            opacity=0.65
        )
        self.bola_alvo = ub.Ball(
            htm=self.htm_home,
            radius=0.016,
            name=f"alvo_{self.sufixo}",
            color="purple",
            opacity=0.8
        )

        # 6. Coleções dinâmicas do cenário
        self.objetos_manipulaveis: Dict[str, ObjetoManipulavel] = {}
        self.obstaculos_configuraveis: Dict[str, ObstaculoConfiguravel] = {}
        self.folhas_papel: Dict[str, FolhaPapel] = {}

        # Região operacional nominal (em coordenadas relativas à base na bancada)
        self.limites_regiao_operacao = {
            "x_min": -0.20,
            "x_max": 0.48,
            "y_min": -0.30,
            "y_max": 0.30,
            "z_min": 0.00,
            "z_max": 0.45,
        }

    def ponto_rel_para_mundo(self, xyz_rel: Tuple[float, float, float]) -> np.ndarray:
        """
        Converte uma coordenada relativa à superfície da bancada (x, y, z_acima_da_mesa)
        para coordenadas globais (3,) no mundo do UAIBot.
        """
        p_local = np.array([xyz_rel[0], xyz_rel[1], xyz_rel[2], 1.0], dtype=float).reshape(4, 1)
        p_mundo = self.htm_referencial_bancada @ p_local
        return p_mundo[0:3, 0].reshape(3)

    def pose_rel_para_mundo(
        self,
        xyz_rel: Tuple[float, float, float],
        rot_local: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Converte posição relativa (x, y, z_acima_da_mesa) e orientação local (4x4)
        para uma matriz de transformação homogênea 4x4 no mundo.
        """
        htm_trn = np.asarray(ub.Utils.trn(list(xyz_rel)), dtype=float)
        if rot_local is None:
            rot_local = np.eye(4, dtype=float)
        return np.asarray(self.htm_referencial_bancada @ htm_trn @ rot_local, dtype=float)

    def verificar_ponto_em_regiao_operacao(self, xyz_rel: Tuple[float, float, float]) -> bool:
        """Verifica se um ponto relativo está dentro da região segura de operação da bancada."""
        x, y, z = xyz_rel
        lim = self.limites_regiao_operacao
        return (
            lim["x_min"] <= x <= lim["x_max"]
            and lim["y_min"] <= y <= lim["y_max"]
            and lim["z_min"] <= z <= lim["z_max"]
        )

    def _criar_geometria_bancada(self) -> List[ub.SimObject]:
        """Cria a mesa e os componentes padrão de bancada (Teste_CCR_V7.ipynb)."""
        ox, oy, oz = self.config.origem_bancada
        h_mesa = self.config.h_mesa

        htm_m = self.htm_referencial_bancada @ np.asarray(
            ub.Utils.trn([X_CENTRO_MESA, 0.0, -h_mesa / 2.0]), dtype=float
        )
        mesa_b = ub.Box(
            htm=htm_m,
            width=self.config.d_mesa,
            depth=self.config.w_mesa,
            height=h_mesa,
            name=f"mesa_{self.sufixo}",
            color=self.config.cor_mesa,
            opacity=1.0
        )
        objs = [mesa_b]

        if not self.config.incluir_acessorios_padrao:
            return objs

        # Rampa lateral (reaproveitada de Teste_CCR_V7.ipynb)
        x_rampa, y_rampa = 0.2745, -0.1775
        largura_rampa = 0.055
        altura_rampa = 0.114
        base_rampa = 0.185
        hipotenusa = 0.213
        theta_rampa = np.arctan2(altura_rampa, base_rampa)

        htm_rampa = (
            self.htm_referencial_bancada
            @ np.asarray(ub.Utils.trn([x_rampa, y_rampa, 0.0]), dtype=float)
            @ np.asarray(ub.Utils.rotx(-theta_rampa), dtype=float)
        )
        rampa_b = ub.Box(
            htm=htm_rampa,
            width=largura_rampa,
            depth=hipotenusa,
            height=altura_rampa,
            name=f"rampa_solida_{self.sufixo}",
            color="#228B22",
            opacity=1.0
        )

        # Suporte 1 traseiro
        x_s1, y_s1 = -0.153, 0.0
        r_s1, h_s1 = 0.03, 0.014
        base_s1_b = ub.Cylinder(
            htm=self.pose_rel_para_mundo((x_s1, y_s1, h_s1 / 2.0)),
            radius=r_s1,
            height=h_s1,
            name=f"base_s1_{self.sufixo}",
            color="black",
            opacity=1.0
        )
        topo_s1_b = ub.Ball(
            htm=self.pose_rel_para_mundo((x_s1, y_s1, h_s1)),
            radius=r_s1,
            name=f"topo_s1_{self.sufixo}",
            color="gray",
            opacity=1.0
        )

        # Suporte 2 traseiro
        x_s2, y_s2 = -0.15, 0.169
        r_base_s2, h_base_s2 = 0.014, 0.007
        r_pino_s2, h_pino_s2 = 0.008, 0.03
        base_s2_b = ub.Cylinder(
            htm=self.pose_rel_para_mundo((x_s2, y_s2, h_base_s2 / 2.0)),
            radius=r_base_s2,
            height=h_base_s2,
            name=f"base_s2_{self.sufixo}",
            color="#404040",
            opacity=1.0
        )
        pino_s2_b = ub.Cylinder(
            htm=self.pose_rel_para_mundo((x_s2, y_s2, h_base_s2 + h_pino_s2 / 2.0)),
            radius=r_pino_s2,
            height=h_pino_s2,
            name=f"pino_s2_{self.sufixo}",
            color="blue",
            opacity=1.0
        )

        objs.extend([rampa_b, base_s1_b, topo_s1_b, base_s2_b, pino_s2_b])
        return objs

    def _criar_gaiola_seguranca(self) -> List[ub.SimObject]:
        """Cria paredes virtuais de confinamento na periferia da bancada."""
        z_rel = 0.50
        y_esq = 0.335
        y_dir = -0.335
        depth_g = y_esq - y_dir
        return [
            ub.Box(
                htm=self.pose_rel_para_mundo((-0.26, 0.0, z_rel)),
                width=0.02,
                depth=depth_g,
                height=1.0,
                name=f"parede_tras_{self.sufixo}",
                color="red",
                opacity=0.20
            ),
            ub.Box(
                htm=self.pose_rel_para_mundo((0.67, 0.0, z_rel)),
                width=0.02,
                depth=depth_g,
                height=1.0,
                name=f"parede_frente_{self.sufixo}",
                color="red",
                opacity=0.20
            ),
        ]

    def adicionar_objeto_manipulavel(
        self,
        nome: str,
        pos_rel: Tuple[float, float, float],
        dimensoes: Tuple[float, float, float] = (0.04, 0.04, 0.10),
        cor: str = "#D2691E",
        formato: str = "box"
    ) -> ObjetoManipulavel:
        """Cria e posiciona um objeto manipulável apoiado sobre a bancada."""
        z_centro = pos_rel[2] + dimensoes[2] / 2.0
        htm_ini = self.pose_rel_para_mundo((pos_rel[0], pos_rel[1], z_centro))
        obj = ObjetoManipulavel(
            nome=nome,
            htm_inicial=htm_ini,
            dimensoes=dimensoes,
            cor=cor,
            formato=formato
        )
        self.objetos_manipulaveis[nome] = obj
        return obj

    def adicionar_obstaculo_configuravel(
        self,
        nome: str,
        pos_rel: Tuple[float, float, float],
        dimensoes: Tuple[float, float, float] = (0.04, 0.04, 0.22),
        cor: str = "#DC143C",
        opacidade: float = 0.85,
        formato: str = "box"
    ) -> ObstaculoConfiguravel:
        """Cria um obstáculo configurável sobre a bancada."""
        htm_ini = self.pose_rel_para_mundo(pos_rel)
        obs = ObstaculoConfiguravel(
            nome=nome,
            htm_inicial=htm_ini,
            dimensoes=dimensoes,
            cor=cor,
            opacidade=opacidade,
            formato=formato
        )
        self.obstaculos_configuraveis[nome] = obs
        return obs

    def adicionar_folha_papel(
        self,
        nome: str = "folha_a4",
        centro_rel: Tuple[float, float, float] = (0.30, 0.00, 0.002),
        dimensoes: Tuple[float, float, float] = (0.22, 0.30, 0.003),
        cor_traco: str = "#0033AA"
    ) -> FolhaPapel:
        """Posiciona uma folha de papel sobre a bancada para a Prática 2."""
        htm_centro = self.pose_rel_para_mundo(centro_rel)
        folha = FolhaPapel(
            nome=nome,
            htm_centro=htm_centro,
            dimensoes=dimensoes,
            cor_traco=cor_traco
        )
        self.folhas_papel[nome] = folha
        return folha

    def atualizar_alvo_visual(self, htm_alvo: np.ndarray, tempo_sim: float) -> None:
        """Atualiza o referencial e a esfera indicadora do alvo atual na simulação."""
        self.frame_alvo.add_ani_frame(time=float(tempo_sim), htm=htm_alvo)
        self.bola_alvo.add_ani_frame(time=float(tempo_sim), htm=htm_alvo)

    def obter_obstaculos_colisao(self, incluir_objetos_manipulaveis: bool = True) -> List[ub.SimObject]:
        """Retorna a lista de todos os objetos de colisão pertencentes a esta bancada."""
        lista: List[ub.SimObject] = list(self.objetos_fixos_bancada) + list(self.paredes_seguranca)
        for obs in self.obstaculos_configuraveis.values():
            lista.append(obs.obj_sim)
        if incluir_objetos_manipulaveis:
            for obj_m in self.objetos_manipulaveis.values():
                lista.append(obj_m.obj_sim)
        return lista

    def obter_objetos_simulacao(self) -> List[ub.SimObject]:
        """Retorna todos os objetos visuais e físicos da bancada para adicionar ao ub.Simulation."""
        elementos: List[ub.SimObject] = [
            self.manipulador.robot,
            self.frame_bancada,
            self.frame_alvo,
            self.marcador_origem,
            self.marcador_destino,
            self.bola_alvo,
        ]
        elementos.extend(self.manipulador.ferramenta.obter_objetos_visuais())
        elementos.extend(self.objetos_fixos_bancada)
        elementos.extend(self.paredes_seguranca)
        for obs in self.obstaculos_configuraveis.values():
            elementos.append(obs.obj_sim)
        for obj_m in self.objetos_manipulaveis.values():
            elementos.append(obj_m.obj_sim)
        for folha in self.folhas_papel.values():
            elementos.append(folha.obj_sim)
        return elementos

    def resetar(self, tempo_sim: float = 0.0) -> None:
        """Redefine o manipulador, a ferramenta, os objetos e os obstáculos ao estado inicial."""
        self.manipulador.resetar(tempo_sim=tempo_sim)
        for obj_m in self.objetos_manipulaveis.values():
            obj_m.resetar(tempo_sim=tempo_sim)
        for obs in self.obstaculos_configuraveis.values():
            obs.resetar(tempo_sim=tempo_sim)
        for folha in self.folhas_papel.values():
            folha.resetar(tempo_sim=tempo_sim)
        self.frame_alvo._frames = []
        self.frame_alvo._max_time = 0
        self.frame_alvo.add_ani_frame(time=float(tempo_sim), htm=self.htm_home)
        self.bola_alvo._frames = []
        self.bola_alvo._max_time = 0
        self.bola_alvo.add_ani_frame(time=float(tempo_sim), htm=self.htm_home)


def inicializar_bancada_padrao(
    config: Optional[ConfigBancada] = None,
    tipo_ferramenta: str = "garra",
    mostrar_primitivas_colisao: bool = SHOW_COLLISION_PRIMITIVES
) -> BancadaPadrao:
    """
    Inicializa uma bancada didática em sua configuração padrão com a ferramenta escolhida
    ('garra' ou 'caneta').
    """
    cfg = config if config is not None else ConfigBancada()
    if tipo_ferramenta.lower() == "caneta":
        ferramenta: FerramentaEfetuador = CanetaFerramenta(nome=f"caneta_{cfg.sufixo}")
    else:
        ferramenta = GarraSimulada(nome=f"garra_{cfg.sufixo}")

    return BancadaPadrao(
        config=cfg,
        ferramenta=ferramenta,
        mostrar_primitivas_colisao=mostrar_primitivas_colisao
    )


def redefinir_bancada(bancada: BancadaPadrao, tempo_sim: float = 0.0) -> None:
    """Redefine todos os elementos da bancada padrão ao seu estado inicial."""
    bancada.resetar(tempo_sim=tempo_sim)

