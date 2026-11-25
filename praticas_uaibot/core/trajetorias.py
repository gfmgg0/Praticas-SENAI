"""
Gerador de trajetórias cartesianas parametrizadas para a Prática 2 (Escrita Robótica).
Suporta linhas, quadrados, círculos e escrita de palavras curtas com controle de
elevação da caneta entre segmentos.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import uaibot as ub


@dataclass
class PontoTrajetoria:
    """Representa um waypoint cartesiano da trajetória de escrita."""
    htm_alvo: np.ndarray
    posicao_xyz: np.ndarray
    caneta_abaixada: bool
    segmento_id: int
    descricao: str = ""


# Fonte vetorial simplificada de traços normalizados em [0, 1] x [0, 1]
# Cada caractere é uma lista de polilinhas [(x0, y0), (x1, y1), ...]
FONTE_VETORIAL: Dict[str, List[List[Tuple[float, float]]]] = {
    "U": [[(0.0, 1.0), (0.0, 0.15), (0.2, 0.0), (0.8, 0.0), (1.0, 0.15), (1.0, 1.0)]],
    "A": [
        [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
        [(0.2, 0.45), (0.8, 0.45)],
    ],
    "I": [
        [(0.2, 1.0), (0.8, 1.0)],
        [(0.5, 1.0), (0.5, 0.0)],
        [(0.2, 0.0), (0.8, 0.0)],
    ],
    "B": [
        [(0.0, 0.0), (0.0, 1.0), (0.75, 1.0), (0.95, 0.75), (0.75, 0.5), (0.0, 0.5)],
        [(0.75, 0.5), (0.95, 0.25), (0.75, 0.0), (0.0, 0.0)],
    ],
    "O": [[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]],
    "T": [
        [(0.0, 1.0), (1.0, 1.0)],
        [(0.5, 1.0), (0.5, 0.0)],
    ],
    "S": [[(1.0, 0.85), (0.75, 1.0), (0.15, 1.0), (0.0, 0.75), (1.0, 0.25), (0.85, 0.0), (0.15, 0.0), (0.0, 0.15)]],
    "E": [
        [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)],
        [(0.0, 0.5), (0.75, 0.5)],
    ],
    "N": [[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]],
    "R": [
        [(0.0, 0.0), (0.0, 1.0), (0.8, 1.0), (1.0, 0.75), (0.8, 0.5), (0.0, 0.5)],
        [(0.5, 0.5), (1.0, 0.0)],
    ],
    "C": [[(1.0, 0.85), (0.75, 1.0), (0.2, 1.0), (0.0, 0.75), (0.0, 0.25), (0.2, 0.0), (0.75, 0.0), (1.0, 0.15)]],
    "L": [[(0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]],
    "P": [[(0.0, 0.0), (0.0, 1.0), (0.85, 1.0), (1.0, 0.75), (0.85, 0.5), (0.0, 0.5)]],
    "3": [
        [(0.0, 0.85), (0.25, 1.0), (0.85, 1.0), (1.0, 0.75), (0.5, 0.5)],
        [(0.5, 0.5), (1.0, 0.25), (0.85, 0.0), (0.25, 0.0), (0.0, 0.15)],
    ],
}


def construir_orientacao_caneta(inclinacao_rad: float = 0.0) -> np.ndarray:
    """
    Retorna a matriz de rotação 4x4 para manter a caneta voltada para a superfície
    (-Z do mundo), com inclinação opcional no plano XZ.
    """
    return np.asarray(ub.Utils.rotx(np.pi) @ ub.Utils.roty(float(inclinacao_rad)), dtype=float)


def _interpolar_polilinha_2d(
    vertices_xy: List[Tuple[float, float]],
    pontos_por_trecho: int
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    n_pts = max(2, int(pontos_por_trecho))
    for k in range(len(vertices_xy) - 1):
        x0, y0 = vertices_xy[k]
        x1, y1 = vertices_xy[k + 1]
        alphas = np.linspace(0.0, 1.0, n_pts, endpoint=(k == len(vertices_xy) - 2))
        for a in alphas:
            pts.append((float((1.0 - a) * x0 + a * x1), float((1.0 - a) * y0 + a * y1)))
    return pts


def empacotar_segmentos_2d_em_trajetoria_3d(
    segmentos_2d: List[List[Tuple[float, float]]],
    htm_referencial_bancada: np.ndarray,
    z_escrita_rel: float,
    z_retracao_rel: float,
    inclinacao_caneta_rad: float = 0.0,
    pontos_transicao_vertical: int = 5
) -> List[PontoTrajetoria]:
    """
    Converte uma lista de segmentos 2D (no plano da folha sobre a bancada) em uma
    trajetória 3D completa com aproximação, descida da caneta, escrita contínua e
    elevação da caneta nas mudanças de segmento.
    """
    rot_caneta = construir_orientacao_caneta(inclinacao_caneta_rad)
    trajetoria: List[PontoTrajetoria] = []

    def criar_ponto(x_rel: float, y_rel: float, z_rel: float, abaixada: bool, seg_id: int, desc: str) -> PontoTrajetoria:
        htm_local = np.asarray(ub.Utils.trn([x_rel, y_rel, z_rel]), dtype=float) @ rot_caneta
        htm_mundo = np.asarray(htm_referencial_bancada @ htm_local, dtype=float)
        pos_xyz = htm_mundo[0:3, 3].reshape(3).copy()
        return PontoTrajetoria(
            htm_alvo=htm_mundo,
            posicao_xyz=pos_xyz,
            caneta_abaixada=abaixada,
            segmento_id=seg_id,
            descricao=desc
        )

    n_vert = max(2, int(pontos_transicao_vertical))

    for seg_idx, seg in enumerate(segmentos_2d):
        if len(seg) == 0:
            continue
        x_ini, y_ini = seg[0]
        x_fim, y_fim = seg[-1]

        # 1. Posicionamento aéreo sobre o início do segmento (caneta levantada)
        trajetoria.append(
            criar_ponto(x_ini, y_ini, z_retracao_rel, False, seg_idx, f"Aproximacao_Seg_{seg_idx + 1}")
        )

        # 2. Descida suave da caneta até a folha
        for z_val in np.linspace(z_retracao_rel, z_escrita_rel, n_vert):
            trajetoria.append(
                criar_ponto(x_ini, y_ini, float(z_val), False, seg_idx, f"Descida_Seg_{seg_idx + 1}")
            )

        # 3. Execução do traço sobre a folha (caneta abaixada)
        for x_p, y_p in seg:
            trajetoria.append(
                criar_ponto(x_p, y_p, z_escrita_rel, True, seg_idx, f"Escrita_Seg_{seg_idx + 1}")
            )

        # 4. Elevação da caneta ao final do segmento
        for z_val in np.linspace(z_escrita_rel, z_retracao_rel, n_vert):
            trajetoria.append(
                criar_ponto(x_fim, y_fim, float(z_val), False, seg_idx, f"Subida_Seg_{seg_idx + 1}")
            )

    return trajetoria


def gerar_segmentos_linha(
    centro_xy: Tuple[float, float],
    comprimento: float = 0.10,
    angulo_rad: float = 0.0,
    n_pontos: int = 30
) -> List[List[Tuple[float, float]]]:
    cx, cy = centro_xy
    dx = 0.5 * comprimento * np.cos(angulo_rad)
    dy = 0.5 * comprimento * np.sin(angulo_rad)
    verts = [(cx - dx, cy - dy), (cx + dx, cy + dy)]
    return [_interpolar_polilinha_2d(verts, n_pontos)]


def gerar_segmentos_quadrado(
    centro_xy: Tuple[float, float],
    lado: float = 0.08,
    pontos_por_lado: int = 18
) -> List[List[Tuple[float, float]]]:
    cx, cy = centro_xy
    h = 0.5 * lado
    verts = [
        (cx - h, cy - h),
        (cx + h, cy - h),
        (cx + h, cy + h),
        (cx - h, cy + h),
        (cx - h, cy - h),
    ]
    return [_interpolar_polilinha_2d(verts, pontos_por_lado)]


def gerar_segmentos_circulo(
    centro_xy: Tuple[float, float],
    raio: float = 0.045,
    n_pontos: int = 60
) -> List[List[Tuple[float, float]]]:
    cx, cy = centro_xy
    thetas = np.linspace(0.0, 2.0 * np.pi, max(12, int(n_pontos)))
    seg = [(float(cx + raio * np.cos(t)), float(cy + raio * np.sin(t))) for t in thetas]
    return [seg]


def gerar_segmentos_palavra(
    texto: str,
    centro_xy: Tuple[float, float],
    altura_letra: float = 0.055,
    pontos_por_trecho: int = 10
) -> List[List[Tuple[float, float]]]:
    """
    Gera os segmentos 2D para escrever uma palavra curta sobre a folha.
    O texto é orientado ao longo do eixo Y da folha (da esquerda y>0 para a direita y<0)
    com o topo das letras apontando para +X da bancada.
    """
    palavra = texto.strip().upper()
    if not palavra:
        palavra = "UAI"

    largura_letra = 0.65 * altura_letra
    espacamento = 0.28 * altura_letra
    largura_total = len(palavra) * largura_letra + max(0, len(palavra) - 1) * espacamento

    cx, cy = centro_xy
    x_base = cx - 0.5 * altura_letra
    y_inicio = cy + 0.5 * largura_total

    segmentos: List[List[Tuple[float, float]]] = []

    for idx_char, ch in enumerate(palavra):
        tracos_norm = FONTE_VETORIAL.get(ch, FONTE_VETORIAL["O"])
        y_char_esq = y_inicio - idx_char * (largura_letra + espacamento)

        for polilinha in tracos_norm:
            verts_mundo_2d: List[Tuple[float, float]] = []
            for u_norm, v_norm in polilinha:
                # v_norm (vertical da letra [0..1]) -> eixo +X da bancada
                # u_norm (horizontal da letra [0..1]) -> eixo -Y da bancada
                x_pt = x_base + v_norm * altura_letra
                y_pt = y_char_esq - u_norm * largura_letra
                verts_mundo_2d.append((float(x_pt), float(y_pt)))
            segmentos.append(_interpolar_polilinha_2d(verts_mundo_2d, pontos_por_trecho))

    return segmentos


def gerar_trajetoria_escrita(
    tipo: str,
    htm_referencial_bancada: np.ndarray,
    centro_folha_xy: Tuple[float, float] = (0.30, 0.00),
    z_escrita_rel: float = 0.005,
    z_retracao_rel: float = 0.045,
    escala: float = 0.06,
    resolucao_pontos: int = 20,
    texto_palavra: str = "UAI",
    inclinacao_caneta_rad: float = 0.0
) -> List[PontoTrajetoria]:
    """
    Função principal para gerar a trajetória cartesiana parametrizada da Prática 2.
    Tipos suportados: 'linha', 'quadrado', 'circulo', 'palavra', 'combo'.
    """
    t = tipo.strip().lower()
    if t == "linha":
        segs = gerar_segmentos_linha(centro_folha_xy, comprimento=1.5 * escala, n_pontos=resolucao_pontos)
    elif t == "quadrado":
        segs = gerar_segmentos_quadrado(centro_folha_xy, lado=escala, pontos_por_lado=max(8, resolucao_pontos // 2))
    elif t == "circulo":
        segs = gerar_segmentos_circulo(centro_folha_xy, raio=0.55 * escala, n_pontos=max(24, resolucao_pontos * 2))
    elif t == "combo":
        # Desenha uma moldura quadrada + círculo + linha central
        segs = (
            gerar_segmentos_quadrado((centro_folha_xy[0], centro_folha_xy[1] + 0.06), lado=0.8 * escala, pontos_por_lado=10)
            + gerar_segmentos_circulo((centro_folha_xy[0], centro_folha_xy[1] - 0.06), raio=0.4 * escala, n_pontos=28)
        )
    else:
        segs = gerar_segmentos_palavra(
            texto=texto_palavra,
            centro_xy=centro_folha_xy,
            altura_letra=escala,
            pontos_por_trecho=max(6, resolucao_pontos // 2)
        )

    return empacotar_segmentos_2d_em_trajetoria_3d(
        segmentos_2d=segs,
        htm_referencial_bancada=htm_referencial_bancada,
        z_escrita_rel=z_escrita_rel,
        z_retracao_rel=z_retracao_rel,
        inclinacao_caneta_rad=inclinacao_caneta_rad
    )

