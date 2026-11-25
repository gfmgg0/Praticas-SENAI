"""
Prática 1 — Manipulação simples (Pick-and-Place) com filtro de segurança QP/CBF.

Sequência operacional:
1. Inicializar o manipulador em posição segura (Home).
2. Posicionar objeto manipulável e obstáculo configurável na bancada.
3. Movimentar o efetuador até a região de coleta (aproximação aérea + descida de pega).
4. Acionar a garra simulada (apreensão + acoplamento da primitiva de colisão).
5. Transportar o objeto até a região de destino desviando do obstáculo via QP/CBF.
6. Liberar o objeto na região de destino.
7. Retornar à configuração segura inicial.

Permite comparar o comportamento com e sem filtro de segurança em simulação.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import uaibot as ub

from ..config.parametros import (
    ANIMATION_STEP,
    ConfigBancada,
    ConfigPratica1,
    DT,
    OUTPUT_DIR,
    PARAM_K,
    PARAM_LAMBDA_DLS,
    SHOW_COLLISION_PRIMITIVES,
)
from ..core.bancada import BancadaPadrao, inicializar_bancada_padrao
from ..core.controle import calcular_erros_pose, compute_u_nominal
from ..core.garra import GarraSimulada
from ..core.seguranca import (
    ParametrosSegurancaQP,
    apply_constraints_single,
    compute_min_obs_distance_euclidean,
    compute_min_self_distance_euclidean,
)


@dataclass
class MetricasPratica1:
    """Métricas registradas durante a execução da Prática 1."""
    usou_filtro: bool
    iteracoes_totais: int
    tempo_total_s: float
    sucesso_coleta: bool
    sucesso_entrega: bool
    erro_final_tarefa: float
    erro_pos_final_m: float
    erro_pos_entrega_m: float
    min_dist_obs_ctrl_m: float
    min_dist_obs_eucl_m: float
    min_dist_self_ctrl_m: float
    max_restricoes_obs_ativas: int
    total_ativacoes_qp_obs: int
    falhas_qp: int
    caminho_html: Optional[str] = None
    hist_t: List[float] = field(default_factory=list)
    hist_erro_tarefa: List[float] = field(default_factory=list)
    hist_erro_pos: List[float] = field(default_factory=list)
    hist_dist_obs_ctrl: List[float] = field(default_factory=list)
    hist_dist_obs_eucl: List[float] = field(default_factory=list)
    hist_dist_self_ctrl: List[float] = field(default_factory=list)
    hist_n_obs_ativas: List[int] = field(default_factory=list)
    hist_etapa: List[str] = field(default_factory=list)


def construir_cenario_pratica_1(
    config_p1: Optional[ConfigPratica1] = None,
    mostrar_primitivas_colisao: bool = SHOW_COLLISION_PRIMITIVES
) -> BancadaPadrao:
    """Cria a bancada padrão configurada para a Prática 1."""
    cfg = config_p1 if config_p1 is not None else ConfigPratica1()
    cfg_bancada = ConfigBancada(
        sufixo="1",
        origem_bancada=(0.0, 0.0, 0.0),
        pos_origem_rel=cfg.pos_origem_rel,
        pos_destino_rel=cfg.pos_destino_rel,
        incluir_acessorios_padrao=True,
        incluir_gaiola_seguranca=False,
    )
    bancada = inicializar_bancada_padrao(
        config=cfg_bancada,
        tipo_ferramenta="garra",
        mostrar_primitivas_colisao=mostrar_primitivas_colisao
    )

    # 1. Objeto manipulável na região de coleta
    bancada.adicionar_objeto_manipulavel(
        nome="peca_p1",
        pos_rel=cfg.pos_origem_rel,
        dimensoes=cfg.dim_objeto,
        cor="#D2691E",
        formato="box"
    )

    # 2. Obstáculo configurável próximo ao caminho entre origem e destino
    bancada.adicionar_obstaculo_configuravel(
        nome="obstaculo_p1",
        pos_rel=cfg.pos_obstaculo_rel,
        dimensoes=cfg.dim_obstaculo,
        cor="#DC143C",
        opacidade=0.85,
        formato="cylinder"
    )

    return bancada


def executar_pratica_1(
    config_p1: Optional[ConfigPratica1] = None,
    usar_filtro_qp: Optional[bool] = None,
    pos_obstaculo_custom_rel: Optional[Tuple[float, float, float]] = None,
    salvar_html: bool = True,
    executar_animacao_browser: bool = False,
    nome_arquivo_html: Optional[str] = None,
    verbose: bool = True
) -> Tuple[BancadaPadrao, ub.Simulation, MetricasPratica1]:
    """
    Executa a simulação completa da Prática 1 (com ou sem filtro QP/CBF) e
    retorna (bancada, sim, metricas).
    """
    cfg = config_p1 if config_p1 is not None else ConfigPratica1()
    if usar_filtro_qp is not None:
        cfg.usar_filtro_qp = bool(usar_filtro_qp)
    if pos_obstaculo_custom_rel is not None:
        cfg.pos_obstaculo_rel = pos_obstaculo_custom_rel

    bancada = construir_cenario_pratica_1(cfg)
    robot = bancada.manipulador.robot
    garra = bancada.manipulador.ferramenta
    assert isinstance(garra, GarraSimulada)

    peca = bancada.objetos_manipulaveis["peca_p1"]
    rot_down = np.asarray(ub.Utils.rotx(np.pi), dtype=float)

    ox, oy, oz = cfg.pos_origem_rel
    dx, dy, dz = cfg.pos_destino_rel
    h_pega = oz + cfg.dim_objeto[2] / 2.0
    h_solta = dz + cfg.dim_objeto[2] / 2.0

    # Definição das etapas da tarefa de Pick-and-Place
    etapas = [
        ("1_Home_Segura", bancada.pose_rel_para_mundo((0.25, 0.08, cfg.altura_aproximacao), rot_down)),
        ("2_Aprox_Coleta", bancada.pose_rel_para_mundo((ox, oy, cfg.altura_aproximacao), rot_down)),
        ("3_Descida_Coleta", bancada.pose_rel_para_mundo((ox, oy, h_pega), rot_down)),
        ("4_Elevacao_Peca", bancada.pose_rel_para_mundo((ox, oy, cfg.altura_transporte), rot_down)),
        ("5_Transporte_Destino", bancada.pose_rel_para_mundo((dx, dy, cfg.altura_transporte), rot_down)),
        ("6_Descida_Entrega", bancada.pose_rel_para_mundo((dx, dy, h_solta), rot_down)),
        ("7_Recuo_Pos_Entrega", bancada.pose_rel_para_mundo((dx, dy, cfg.altura_aproximacao), rot_down)),
        ("8_Retorno_Home", bancada.pose_rel_para_mundo((0.25, 0.02, cfg.altura_aproximacao), rot_down)),
    ]

    params_qp = ParametrosSegurancaQP(
        eta_obs=cfg.eta_obs,
        obs_delta=cfg.obs_delta,
        obs_activation=cfg.obs_activation,
        max_dist_obs=max(0.12, cfg.obs_activation + 0.04),
    )

    obstaculos_cenario = bancada.obter_obstaculos_colisao(incluir_objetos_manipulaveis=True)
    # Ignora a rampa lateral fixa caso o destino esteja próximo a ela para focar no obstáculo didático central
    ignored_obs: set = {"rampa_solida_1"}

    # Rastro visual do TCP
    p_eef_0 = bancada.manipulador.posicao_efetuador()
    cor_rastro = "cyan" if cfg.usar_filtro_qp else "magenta"
    bola_rastro = ub.Ball(
        htm=ub.Utils.trn(p_eef_0),
        radius=0.008,
        name="rastro_tcp_p1",
        color=cor_rastro
    )

    sim = ub.Simulation(bancada.obter_objetos_simulacao() + [bola_rastro])
    if SHOW_COLLISION_PRIMITIVES:
        robot.add_col_object(sim)

    idx_etapa = 0
    iter_inicio_etapa = 0
    i = 0
    max_iter_global = len(etapas) * cfg.max_iter_por_etapa

    sucesso_coleta = False
    sucesso_entrega = False
    erro_pos_entrega = np.inf
    falhas_qp = 0

    hist_t: List[float] = []
    hist_erro_tarefa: List[float] = []
    hist_erro_pos: List[float] = []
    hist_dist_obs_ctrl: List[float] = []
    hist_dist_obs_eucl: List[float] = []
    hist_dist_self_ctrl: List[float] = []
    hist_n_obs_ativas: List[int] = []
    hist_etapa: List[str] = []

    modo_str = "COM FILTRO QP/CBF" if cfg.usar_filtro_qp else "SEM FILTRO (NOMINAL)"
    if verbose:
        print(f"\n=== Iniciando Prática 1 ({modo_str}) ===")

    while idx_etapa < len(etapas) and i < max_iter_global:
        i += 1
        t_atual = i * DT
        iter_local = i - iter_inicio_etapa
        nome_etapa, htm_alvo = etapas[idx_etapa]

        q_atual = bancada.manipulador.ler_juntas()
        bancada.atualizar_alvo_visual(htm_alvo, tempo_sim=t_atual)

        # Durante a descida de coleta/entrega, desconsidera a própria peça estática como obstáculo
        if idx_etapa in (1, 2):
            ignored_obs.add(peca.nome)

        u_nom, erro_r, _, _ = compute_u_nominal(
            _robot=robot,
            _q=q_atual,
            _htm_tg=htm_alvo,
            _lambda_dls=PARAM_LAMBDA_DLS,
            _param_k=PARAM_K
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
                ignored_obs_names=ignored_obs,
                max_dist=params_qp.max_dist_obs
            )
            n_obs_act = 0

        # Integra velocidade articular e atualiza animação
        gravar_frame = (i % ANIMATION_STEP == 0)
        q_novo = bancada.manipulador.aplicar_velocidade_articular(
            u=u_cmd,
            dt=DT,
            tempo_sim=t_atual,
            registrar_animacao=gravar_frame
        )

        htm_eef = bancada.manipulador.pose_efetuador(q_novo)
        p_eef = htm_eef[0:3, 3].reshape(3)
        if gravar_frame:
            bola_rastro.add_ani_frame(time=t_atual, htm=ub.Utils.trn(p_eef))
            if peca.preso_por is None:
                peca.atualizar_pose(peca.htm_atual, tempo_sim=t_atual)

        erro_pos, erro_ori = calcular_erros_pose(htm_eef, htm_alvo)
        d_obs_eucl = compute_min_obs_distance_euclidean(
            robot, q_novo, obstaculos_cenario,
            ignored_link_for_table=0,
            ignored_obs_names=ignored_obs,
            max_dist=params_qp.max_dist_obs
        )

        hist_t.append(t_atual)
        hist_erro_tarefa.append(erro_r)
        hist_erro_pos.append(erro_pos)
        hist_dist_obs_ctrl.append(d_obs_ctrl)
        hist_dist_obs_eucl.append(d_obs_eucl)
        hist_dist_self_ctrl.append(d_self_ctrl)
        hist_n_obs_ativas.append(n_obs_act)
        hist_etapa.append(nome_etapa)

        # Verifica transição de etapa com precisão posicional e angular
        atingiu_alvo = (erro_pos <= 0.0035 and erro_ori <= 0.04) or (erro_r <= cfg.tol_erro)
        estourou_tempo = iter_local >= cfg.max_iter_por_etapa

        if atingiu_alvo or estourou_tempo:
            if verbose:
                print(
                    f"  [Etapa {idx_etapa + 1}/{len(etapas)}] {nome_etapa:<22} | "
                    f"iter={iter_local:03d} | V(r)={erro_r:.4f} | erro_pos={erro_pos*1000:.2f} mm | "
                    f"d_obs_min={min(d_obs_ctrl, d_obs_eucl)*1000:.1f} mm"
                )

            # Ação de fechamento da garra ao concluir a descida de coleta (etapa idx 2)
            if idx_etapa == 2:
                sucesso_coleta = garra.fechar(
                    robot=robot,
                    q=q_novo,
                    objeto_candidato=peca,
                    tempo_sim=t_atual,
                    obstaculos_ignorados=ignored_obs,
                    forcar_apreensao=False
                )
                if verbose:
                    print(f"    -> Garra acionada (FECHAR): sucesso_coleta={sucesso_coleta}")

            # Ação de abertura da garra ao concluir a descida de entrega (etapa idx 5)
            elif idx_etapa == 5:
                garra.abrir(
                    robot=robot,
                    q=q_novo,
                    tempo_sim=t_atual,
                    obstaculos_ignorados=ignored_obs
                )
                p_peca_final = peca.obter_posicao()
                p_dest_esp = bancada.ponto_rel_para_mundo((dx, dy, h_solta))
                erro_pos_entrega = float(np.linalg.norm(p_peca_final - p_dest_esp))
                sucesso_entrega = erro_pos_entrega <= 0.025
                if verbose:
                    print(
                        f"    -> Garra acionada (ABRIR): peça liberada | "
                        f"erro_entrega={erro_pos_entrega*1000:.2f} mm"
                    )

            idx_etapa += 1
            iter_inicio_etapa = i

    caminho_salvo = None
    if salvar_html:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if nome_arquivo_html is None:
            suf = "com_filtro" if cfg.usar_filtro_qp else "sem_filtro"
            nome_arquivo_html = f"pratica_1_{suf}"
        sim.save(OUTPUT_DIR, file_name=nome_arquivo_html)
        caminho_salvo = os.path.join(OUTPUT_DIR, f"{nome_arquivo_html}.html")

    if executar_animacao_browser:
        sim.run()

    metricas = MetricasPratica1(
        usou_filtro=cfg.usar_filtro_qp,
        iteracoes_totais=len(hist_t),
        tempo_total_s=hist_t[-1] if hist_t else 0.0,
        sucesso_coleta=sucesso_coleta,
        sucesso_entrega=sucesso_entrega,
        erro_final_tarefa=hist_erro_tarefa[-1] if hist_erro_tarefa else np.inf,
        erro_pos_final_m=hist_erro_pos[-1] if hist_erro_pos else np.inf,
        erro_pos_entrega_m=erro_pos_entrega,
        min_dist_obs_ctrl_m=float(np.min(hist_dist_obs_ctrl)) if hist_dist_obs_ctrl else np.inf,
        min_dist_obs_eucl_m=float(np.min(hist_dist_obs_eucl)) if hist_dist_obs_eucl else np.inf,
        min_dist_self_ctrl_m=float(np.min(hist_dist_self_ctrl)) if hist_dist_self_ctrl else np.inf,
        max_restricoes_obs_ativas=int(np.max(hist_n_obs_ativas)) if hist_n_obs_ativas else 0,
        total_ativacoes_qp_obs=int(np.sum(np.array(hist_n_obs_ativas) > 0)) if hist_n_obs_ativas else 0,
        falhas_qp=falhas_qp,
        caminho_html=caminho_salvo,
        hist_t=hist_t,
        hist_erro_tarefa=hist_erro_tarefa,
        hist_erro_pos=hist_erro_pos,
        hist_dist_obs_ctrl=hist_dist_obs_ctrl,
        hist_dist_obs_eucl=hist_dist_obs_eucl,
        hist_dist_self_ctrl=hist_dist_self_ctrl,
        hist_n_obs_ativas=hist_n_obs_ativas,
        hist_etapa=hist_etapa,
    )

    if verbose:
        print(
            f"=== Resumo Prática 1 ({modo_str}) ===\n"
            f"  Iterações: {metricas.iteracoes_totais} ({metricas.tempo_total_s:.2f} s)\n"
            f"  Coleta OK: {metricas.sucesso_coleta} | Entrega OK: {metricas.sucesso_entrega} "
            f"(erro={metricas.erro_pos_entrega_m*1000:.2f} mm)\n"
            f"  Distância mín. obstáculo (euclidiana): {metricas.min_dist_obs_eucl_m*1000:.2f} mm\n"
            f"  Distância mín. autocolisão: {metricas.min_dist_self_ctrl_m*1000:.2f} mm\n"
            f"  Máx. restrições de obstáculo ativas: {metricas.max_restricoes_obs_ativas} "
            f"({metricas.total_ativacoes_qp_obs} passos ativos)\n"
            f"  Arquivo HTML: {metricas.caminho_html}"
        )

    return bancada, sim, metricas


def comparar_com_e_sem_filtro(
    config_p1: Optional[ConfigPratica1] = None,
    salvar_html: bool = True,
    verbose: bool = True
) -> Dict[str, MetricasPratica1]:
    """
    Executa a Prática 1 em simulação COM filtro QP/CBF e SEM filtro (nominal),
    comparando distâncias mínimas ao obstáculo, ativação das restrições e precisão.
    """
    cfg_com = config_p1 if config_p1 is not None else ConfigPratica1()
    _, _, met_com = executar_pratica_1(
        config_p1=cfg_com,
        usar_filtro_qp=True,
        salvar_html=salvar_html,
        nome_arquivo_html="pratica_1_com_filtro",
        verbose=verbose
    )

    cfg_sem = ConfigPratica1(
        pos_origem_rel=cfg_com.pos_origem_rel,
        pos_destino_rel=cfg_com.pos_destino_rel,
        pos_obstaculo_rel=cfg_com.pos_obstaculo_rel,
        dim_obstaculo=cfg_com.dim_obstaculo,
        dim_objeto=cfg_com.dim_objeto,
        altura_aproximacao=cfg_com.altura_aproximacao,
        altura_transporte=cfg_com.altura_transporte,
        usar_filtro_qp=False,
        max_iter_por_etapa=cfg_com.max_iter_por_etapa,
        tol_erro=cfg_com.tol_erro,
    )
    _, _, met_sem = executar_pratica_1(
        config_p1=cfg_sem,
        usar_filtro_qp=False,
        salvar_html=salvar_html,
        nome_arquivo_html="pratica_1_sem_filtro",
        verbose=verbose
    )

    if verbose:
        print("\n" + "=" * 68)
        print("COMPARATIVO PRÁTICA 1: COM FILTRO QP/CBF vs SEM FILTRO (NOMINAL)")
        print("=" * 68)
        print(
            f"  Dist. Mín. Obstáculo (COM filtro): {met_com.min_dist_obs_eucl_m*1000:6.2f} mm "
            f"(Margem δ_obs = {cfg_com.obs_delta*1000:.1f} mm)"
        )
        print(
            f"  Dist. Mín. Obstáculo (SEM filtro): {met_sem.min_dist_obs_eucl_m*1000:6.2f} mm "
            f"{'-> COLISÃO / VIOLAÇÃO DE MARGEM!' if met_sem.min_dist_obs_eucl_m < cfg_com.obs_delta else ''}"
        )
        print(
            f"  Passos com CBF de obstáculo ativa: {met_com.total_ativacoes_qp_obs} passos "
            f"(máx {met_com.max_restricoes_obs_ativas} restrições simultâneas)"
        )
        print(
            f"  Erro posicional de entrega:        "
            f"COM={met_com.erro_pos_entrega_m*1000:.2f} mm | SEM={met_sem.erro_pos_entrega_m*1000:.2f} mm"
        )
        print("=" * 68)

    return {"com_filtro": met_com, "sem_filtro": met_sem}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prática 1 — Manipulação simples (Pick-and-Place) com filtro QP/CBF no UAIBot."
    )
    parser.add_argument(
        "--modo",
        choices=["comparar", "com_filtro", "sem_filtro"],
        default="comparar",
        help="Executa apenas com filtro, apenas sem filtro ou comparativo completo (padrão: comparar)."
    )
    parser.add_argument(
        "--obs-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Posição relativa customizada do obstáculo sobre a bancada [m], ex: 0.30 0.01 0.11"
    )
    args = parser.parse_args()

    cfg = ConfigPratica1()
    if args.obs_xyz is not None:
        cfg.pos_obstaculo_rel = (float(args.obs_xyz[0]), float(args.obs_xyz[1]), float(args.obs_xyz[2]))

    if args.modo == "comparar":
        comparar_com_e_sem_filtro(config_p1=cfg, salvar_html=True, verbose=True)
    elif args.modo == "com_filtro":
        executar_pratica_1(config_p1=cfg, usar_filtro_qp=True, salvar_html=True, verbose=True)
    else:
        executar_pratica_1(config_p1=cfg, usar_filtro_qp=False, salvar_html=True, verbose=True)


if __name__ == "__main__":
    main()
