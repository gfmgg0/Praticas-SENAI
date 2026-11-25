"""
Prática 2 — Rastreamento de trajetória cartesiana com escrita robótica no UAIBot.

Sequência operacional:
1. Inicializar a bancada padrão.
2. Substituir a garra pela ferramenta de escrita simulada (CanetaFerramenta com TCP próprio).
3. Posicionar uma folha sobre a bancada (FolhaPapel).
4. Gerar trajetória cartesiana parametrizada (linha, quadrado, círculo, palavra curta ou combo).
5. Movimentar o efetuador até o início da trajetória.
6. Aproximar a caneta da superfície da folha.
7. Executar o rastreamento contínuo da trajetória controlando posição e orientação.
8. Elevar a caneta nas mudanças de segmento.
9. Retornar à posição segura inicial.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import uaibot as ub

from ..config.parametros import (
    ANIMATION_STEP,
    ConfigBancada,
    ConfigPratica2,
    DT,
    OUTPUT_DIR,
    PARAM_LAMBDA_DLS,
    SHOW_COLLISION_PRIMITIVES,
)
from ..core.bancada import BancadaPadrao, inicializar_bancada_padrao
from ..core.controle import calcular_erros_pose, compute_u_nominal
from ..core.garra import CanetaFerramenta
from ..core.seguranca import (
    ParametrosSegurancaQP,
    apply_constraints_single,
    compute_min_obs_distance_euclidean,
    compute_min_self_distance_euclidean,
)
from ..core.trajetorias import PontoTrajetoria, construir_orientacao_caneta, gerar_trajetoria_escrita


@dataclass
class MetricasPratica2:
    """Métricas de rastreamento e segurança registradas na Prática 2."""
    tipo_trajetoria: str
    texto_palavra: str
    num_waypoints: int
    num_pontos_escritos: int
    iteracoes_totais: int
    tempo_total_s: float
    erro_pos_medio_escrita_m: float
    erro_pos_rmse_escrita_m: float
    erro_pos_max_escrita_m: float
    erro_ori_medio_escrita_rad: float
    min_dist_obs_ctrl_m: float
    min_dist_self_ctrl_m: float
    total_ativacoes_qp_obs: int
    falhas_qp: int
    caminho_html: Optional[str] = None
    hist_t: List[float] = field(default_factory=list)
    hist_erro_pos: List[float] = field(default_factory=list)
    hist_erro_ori: List[float] = field(default_factory=list)
    hist_caneta_ativa: List[bool] = field(default_factory=list)
    hist_dist_obs_ctrl: List[float] = field(default_factory=list)


def construir_cenario_pratica_2(
    config_p2: Optional[ConfigPratica2] = None,
    mostrar_primitivas_colisao: bool = SHOW_COLLISION_PRIMITIVES
) -> BancadaPadrao:
    """Inicializa a bancada padrão equipada com CanetaFerramenta e FolhaPapel."""
    cfg = config_p2 if config_p2 is not None else ConfigPratica2()
    cfg_bancada = ConfigBancada(
        sufixo="1",
        origem_bancada=(0.0, 0.0, 0.0),
        incluir_acessorios_padrao=True,
        incluir_gaiola_seguranca=False,
    )
    # Inicializa bancada já substituindo a garra pela ferramenta de escrita (Caneta)
    bancada = inicializar_bancada_padrao(
        config=cfg_bancada,
        tipo_ferramenta="caneta",
        mostrar_primitivas_colisao=mostrar_primitivas_colisao
    )

    # Posiciona a folha sobre a bancada
    bancada.adicionar_folha_papel(
        nome="folha_a4",
        centro_rel=cfg.centro_folha_rel,
        dimensoes=cfg.dim_folha,
        cor_traco="#0033AA"
    )

    # Obstáculo opcional próximo à borda da folha para demonstrar proteção do filtro QP
    if cfg.incluir_obstaculo_borda:
        bancada.adicionar_obstaculo_configuravel(
            nome="suporte_borda_p2",
            pos_rel=cfg.pos_obstaculo_rel,
            dimensoes=(0.035, 0.035, 0.14),
            cor="#B22222",
            opacidade=0.80,
            formato="cylinder"
        )

    return bancada


def executar_pratica_2(
    config_p2: Optional[ConfigPratica2] = None,
    tipo_trajetoria: Optional[str] = None,
    texto_palavra: Optional[str] = None,
    salvar_html: bool = True,
    executar_animacao_browser: bool = False,
    nome_arquivo_html: Optional[str] = None,
    verbose: bool = True
) -> Tuple[BancadaPadrao, ub.Simulation, MetricasPratica2]:
    """
    Executa a simulação completa da Prática 2 (Escrita Robótica) e retorna
    (bancada, sim, metricas).
    """
    cfg = config_p2 if config_p2 is not None else ConfigPratica2()
    if tipo_trajetoria is not None:
        cfg.tipo_trajetoria = tipo_trajetoria
    if texto_palavra is not None:
        cfg.texto_palavra = texto_palavra

    bancada = construir_cenario_pratica_2(cfg)
    robot = bancada.manipulador.robot
    caneta = bancada.manipulador.ferramenta
    assert isinstance(caneta, CanetaFerramenta)
    folha = bancada.folhas_papel["folha_a4"]

    # Gera os waypoints da trajetória cartesiana de escrita
    z_topo_folha_rel = cfg.centro_folha_rel[2] + cfg.dim_folha[2] / 2.0
    waypoints: List[PontoTrajetoria] = gerar_trajetoria_escrita(
        tipo=cfg.tipo_trajetoria,
        htm_referencial_bancada=bancada.htm_referencial_bancada,
        centro_folha_xy=(cfg.centro_folha_rel[0], cfg.centro_folha_rel[1]),
        z_escrita_rel=z_topo_folha_rel + cfg.altura_escrita_rel,
        z_retracao_rel=z_topo_folha_rel + cfg.altura_retracao_rel,
        escala=cfg.escala_desenho,
        resolucao_pontos=cfg.pontos_por_segmento,
        texto_palavra=cfg.texto_palavra,
        inclinacao_caneta_rad=cfg.inclinacao_caneta_rad
    )

    rot_caneta = construir_orientacao_caneta(cfg.inclinacao_caneta_rad)
    htm_home_segura = bancada.pose_rel_para_mundo((0.26, 0.02, 0.22), rot_local=rot_caneta)

    # Adiciona abordagem inicial partindo de Home e retorno final à posição segura
    ponto_home_ini = PontoTrajetoria(
        htm_alvo=htm_home_segura,
        posicao_xyz=htm_home_segura[0:3, 3].reshape(3),
        caneta_abaixada=False,
        segmento_id=-1,
        descricao="Home_Inicial"
    )
    ponto_home_fim = PontoTrajetoria(
        htm_alvo=htm_home_segura,
        posicao_xyz=htm_home_segura[0:3, 3].reshape(3),
        caneta_abaixada=False,
        segmento_id=999,
        descricao="Retorno_Home_Segura"
    )
    sequencia_completa = [ponto_home_ini] + waypoints + [ponto_home_fim]

    # Ignora a mesa e a rampa lateral para permitir aproximação milimétrica da ponta da caneta à folha
    obstaculos_cenario = bancada.obter_obstaculos_colisao(incluir_objetos_manipulaveis=False)
    ignored_obs = {f"mesa_{bancada.sufixo}", f"rampa_solida_{bancada.sufixo}"}

    params_qp = ParametrosSegurancaQP(
        eta_obs=0.22,
        obs_delta=0.012,
        obs_activation=0.045,
        max_dist_obs=0.09
    )

    # Ganho mais alto para rastreamento fino de trajetória cartesiana
    ganho_k_rastreamento = 0.55

    i = 0
    falhas_qp = 0
    hist_t: List[float] = []
    hist_erro_pos: List[float] = []
    hist_erro_ori: List[float] = []
    hist_caneta_ativa: List[bool] = []
    hist_dist_obs_ctrl: List[float] = []
    hist_dist_self_ctrl: List[float] = []
    hist_n_obs_ativas: List[int] = []

    if verbose:
        print(
            f"\n=== Iniciando Prática 2 — Escrita Robótica ("
            f"tipo='{cfg.tipo_trajetoria}', palavra='{cfg.texto_palavra}', "
            f"waypoints={len(waypoints)}) ==="
        )

    for idx_wp, wp in enumerate(sequencia_completa):
        # Na chegada ao primeiro ponto ou retorno ao Home, permite mais passos de convergência;
        # durante os pontos densos do traço, usa passos_por_ponto configurável.
        if idx_wp in (0, 1, len(sequencia_completa) - 1):
            max_passos_wp = 180
            tol_pos_wp = 0.0025
        elif not wp.caneta_abaixada:
            max_passos_wp = max(18, cfg.passos_por_ponto * 3)
            tol_pos_wp = 0.0020
        else:
            max_passos_wp = max(3, cfg.passos_por_ponto)
            tol_pos_wp = 0.0012

        for passo_local in range(max_passos_wp):
            i += 1
            t_atual = i * DT
            q_atual = bancada.manipulador.ler_juntas()
            bancada.atualizar_alvo_visual(wp.htm_alvo, tempo_sim=t_atual)

            u_nom, erro_r, _, _ = compute_u_nominal(
                _robot=robot,
                _q=q_atual,
                _htm_tg=wp.htm_alvo,
                _lambda_dls=PARAM_LAMBDA_DLS,
                _param_k=ganho_k_rastreamento
            )

            if cfg.usar_filtro_qp:
                res_qp = apply_constraints_single(
                    _robot=robot,
                    _q=q_atual,
                    _u_nom=u_nom,
                    _all_obstacles=obstaculos_cenario,
                    ignored_link_for_table=0,
                    ignored_obs_names=ignored_obs,
                    params=params_qp
                )
                u_cmd = res_qp.u1
                d_obs_ctrl = res_qp.min_obs_dist_ctrl
                d_self_ctrl = res_qp.min_self_dist_ctrl
                n_obs_act = res_qp.n_obs_constraints
                if not res_qp.qp_viavel:
                    falhas_qp += 1
            else:
                u_cmd = u_nom.copy()
                d_self_ctrl = compute_min_self_distance_euclidean(robot, q_atual)
                d_obs_ctrl = compute_min_obs_distance_euclidean(
                    robot, q_atual, obstaculos_cenario,
                    ignored_link_for_table=0,
                    ignored_obs_names=ignored_obs
                )
                n_obs_act = 0

            gravar_frame = (i % ANIMATION_STEP == 0)
            q_novo = bancada.manipulador.aplicar_velocidade_articular(
                u=u_cmd,
                dt=DT,
                tempo_sim=t_atual,
                registrar_animacao=gravar_frame
            )

            htm_eef = bancada.manipulador.pose_efetuador(q_novo)
            p_ponta = htm_eef[0:3, 3].reshape(3)
            erro_pos, erro_ori = calcular_erros_pose(htm_eef, wp.htm_alvo)

            if gravar_frame:
                folha.registrar_ponto_traco(
                    ponto_xyz=p_ponta,
                    tempo_sim=t_atual,
                    caneta_ativa=wp.caneta_abaixada
                )

            hist_t.append(t_atual)
            hist_erro_pos.append(erro_pos)
            hist_erro_ori.append(erro_ori)
            hist_caneta_ativa.append(wp.caneta_abaixada)
            hist_dist_obs_ctrl.append(d_obs_ctrl)
            hist_dist_self_ctrl.append(d_self_ctrl)
            hist_n_obs_ativas.append(n_obs_act)

            if erro_pos <= tol_pos_wp and passo_local >= 1:
                break

    # Gera o PointCloud animado com o traço desenhado sobre a folha
    pc_traco = folha.gerar_pointcloud_animado(nome_pc="traco_escrita_p2")
    sim = ub.Simulation(bancada.obter_objetos_simulacao() + [pc_traco])
    if SHOW_COLLISION_PRIMITIVES:
        robot.add_col_object(sim)

    caminho_salvo = None
    if salvar_html:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if nome_arquivo_html is None:
            nome_arquivo_html = f"pratica_2_{cfg.tipo_trajetoria}"
        sim.save(OUTPUT_DIR, file_name=nome_arquivo_html)
        caminho_salvo = os.path.join(OUTPUT_DIR, f"{nome_arquivo_html}.html")

    if executar_animacao_browser:
        sim.run()

    # Filtra erros apenas nos instantes em que a caneta está escrevendo na superfície
    erros_pos_escrita = [e for e, a in zip(hist_erro_pos, hist_caneta_ativa) if a]
    erros_ori_escrita = [o for o, a in zip(hist_erro_ori, hist_caneta_ativa) if a]
    if not erros_pos_escrita:
        erros_pos_escrita = hist_erro_pos
        erros_ori_escrita = hist_erro_ori

    arr_pos = np.asarray(erros_pos_escrita, dtype=float)
    arr_ori = np.asarray(erros_ori_escrita, dtype=float)

    metricas = MetricasPratica2(
        tipo_trajetoria=cfg.tipo_trajetoria,
        texto_palavra=cfg.texto_palavra,
        num_waypoints=len(waypoints),
        num_pontos_escritos=len(folha.pontos_traco),
        iteracoes_totais=len(hist_t),
        tempo_total_s=hist_t[-1] if hist_t else 0.0,
        erro_pos_medio_escrita_m=float(np.mean(arr_pos)),
        erro_pos_rmse_escrita_m=float(np.sqrt(np.mean(arr_pos ** 2))),
        erro_pos_max_escrita_m=float(np.max(arr_pos)),
        erro_ori_medio_escrita_rad=float(np.mean(arr_ori)),
        min_dist_obs_ctrl_m=float(np.min(hist_dist_obs_ctrl)) if hist_dist_obs_ctrl else np.inf,
        min_dist_self_ctrl_m=float(np.min(hist_dist_self_ctrl)) if hist_dist_self_ctrl else np.inf,
        total_ativacoes_qp_obs=int(np.sum(np.array(hist_n_obs_ativas) > 0)) if hist_n_obs_ativas else 0,
        falhas_qp=falhas_qp,
        caminho_html=caminho_salvo,
        hist_t=hist_t,
        hist_erro_pos=hist_erro_pos,
        hist_erro_ori=hist_erro_ori,
        hist_caneta_ativa=hist_caneta_ativa,
        hist_dist_obs_ctrl=hist_dist_obs_ctrl,
    )

    if verbose:
        print(
            f"=== Resumo Prática 2 ({cfg.tipo_trajetoria.upper()}) ===\n"
            f"  Iterações: {metricas.iteracoes_totais} ({metricas.tempo_total_s:.2f} s)\n"
            f"  Pontos gravados no traço (PointCloud): {metricas.num_pontos_escritos}\n"
            f"  Erro médio de rastreamento (escrita): {metricas.erro_pos_medio_escrita_m*1000:.2f} mm "
            f"(RMSE={metricas.erro_pos_rmse_escrita_m*1000:.2f} mm, Máx={metricas.erro_pos_max_escrita_m*1000:.2f} mm)\n"
            f"  Erro médio de orientação da caneta: { np.degrees(metricas.erro_ori_medio_escrita_rad):.2f} deg\n"
            f"  Distância mín. autocolisão: {metricas.min_dist_self_ctrl_m*1000:.2f} mm\n"
            f"  Arquivo HTML: {metricas.caminho_html}"
        )

    return bancada, sim, metricas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prática 2 — Rastreamento de trajetória com escrita robótica no UAIBot."
    )
    parser.add_argument(
        "--tipo",
        choices=["palavra", "linha", "quadrado", "circulo", "combo"],
        default="palavra",
        help="Tipo de trajetória a desenhar sobre a folha (padrão: palavra)."
    )
    parser.add_argument(
        "--palavra",
        type=str,
        default="UAI",
        help="Palavra curta a ser escrita quando --tipo=palavra (padrão: UAI)."
    )
    parser.add_argument(
        "--escala",
        type=float,
        default=0.06,
        help="Escala da figura/letra em metros (padrão: 0.06)."
    )
    args = parser.parse_args()

    cfg = ConfigPratica2(
        tipo_trajetoria=args.tipo,
        texto_palavra=args.palavra,
        escala_desenho=args.escala
    )
    executar_pratica_2(config_p2=cfg, salvar_html=True, verbose=True)


if __name__ == "__main__":
    main()

