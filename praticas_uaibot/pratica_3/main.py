"""
Prática 3 — Colaboração entre duas bancadas com dois manipuladores UR3e no UAIBot.

Máquina de Estados de Cooperação (EstadoCooperacao):
1. FASE_1_APROXIMACAO_COLETA_A: Robô A vai para o ponto aéreo sobre o objeto; Robô B aguarda em Home segura.
2. FASE_2_DESCIDA_E_PEGA_A: Robô A desce até o objeto e aciona sua garra para coletá-lo.
3. FASE_3_TRANSPORTE_ATE_ZONA_TRANSFERENCIA: Robô A eleva e leva o objeto até a zona compartilhada entre as bancadas.
4. FASE_4_APROXIMACAO_ROBO_B: Com o Robô A estabilizado na região de entrega, o Robô B aproxima seu efetuador em orientação perpendicular.
5. FASE_5_TRANSFERENCIA_SINCRONIZADA: Valida condição de sincronização espacial; Robô B fecha a garra primeiro (dupla apreensão segura) e só então o Robô A abre sua garra.
6. FASE_6_RECUO_A_E_TRANSPORTE_B: Robô A recua para sua posição segura enquanto Robô B transporta o objeto até a região de destino na Bancada B.
7. FASE_7_ENTREGA_DESTINO_B: Robô B posiciona o objeto no destino da Bancada B e abre sua garra.
8. FASE_8_RETORNO_SEGURO_AMBOS: Ambos os manipuladores retornam às configurações seguras finais.
- ESTADO_SEGURO_FALHA: Acionado em caso de falha de sincronização ou inviabilidade persistente do QP (freio seguro u1=0, u2=0 mantendo a peça presa).
"""

import argparse
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import uaibot as ub

from ..config.parametros import (
    ANIMATION_STEP,
    ConfigBancada,
    ConfigPratica3,
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
    apply_constraints_dual,
    compute_min_obs_distance_euclidean,
    compute_min_rob_to_rob_distance_euclidean,
    compute_min_self_distance_euclidean,
)


class EstadoCooperacao(Enum):
    FASE_1_APROXIMACAO_COLETA_A = 1
    FASE_2_DESCIDA_E_PEGA_A = 2
    FASE_3_TRANSPORTE_ATE_ZONA_TRANSFERENCIA = 3
    FASE_4_APROXIMACAO_ROBO_B = 4
    FASE_5_TRANSFERENCIA_SINCRONIZADA = 5
    FASE_6_RECUO_A_E_TRANSPORTE_B = 6
    FASE_7_ENTREGA_DESTINO_B = 7
    FASE_8_RETORNO_SEGURO_AMBOS = 8
    CONCLUIDO = 9
    ESTADO_SEGURO_FALHA = 99


@dataclass
class MetricasPratica3:
    """Métricas registradas durante a execução colaborativa da Prática 3."""
    estado_final: str
    sucesso_coleta_a: bool
    sucesso_sincronizacao_handover: bool
    sucesso_entrega_b: bool
    iteracoes_totais: int
    tempo_total_s: float
    erro_sincronizacao_handover_m: float
    erro_entrega_final_b_m: float
    min_dist_rob_ctrl_m: float
    min_dist_rob_eucl_m: float
    min_dist_obs_ctrl_m: float
    min_dist_self_ctrl_m: float
    max_restricoes_rob_ativas: int
    total_ativacoes_qp_rob: int
    falhas_qp: int
    motivo_estado_seguro: str = "Nenhum"
    caminho_html: Optional[str] = None
    hist_t: List[float] = field(default_factory=list)
    hist_estado: List[str] = field(default_factory=list)
    hist_erro_r1: List[float] = field(default_factory=list)
    hist_erro_r2: List[float] = field(default_factory=list)
    hist_dist_rob_ctrl: List[float] = field(default_factory=list)
    hist_dist_rob_eucl: List[float] = field(default_factory=list)
    hist_dist_obs_ctrl: List[float] = field(default_factory=list)
    hist_n_rob_ativas: List[int] = field(default_factory=list)


def construir_cenario_pratica_3(
    config_p3: Optional[ConfigPratica3] = None,
    mostrar_primitivas_colisao: bool = SHOW_COLLISION_PRIMITIVES
) -> Tuple[BancadaPadrao, BancadaPadrao, List[ub.SimObject]]:
    """
    Inicializa duas instâncias da BancadaPadrao (Bancada A em y=0.0 e Bancada B em y=dy_bancada_b)
    mais os indicadores da zona compartilhada de transferência.
    """
    cfg = config_p3 if config_p3 is not None else ConfigPratica3()

    cfg_a = ConfigBancada(
        sufixo="A",
        origem_bancada=(0.0, 0.0, 0.0),
        pos_origem_rel=cfg.pos_coleta_a_rel,
        pos_destino_rel=(0.25, -0.30, 0.0),
        incluir_acessorios_padrao=True,
        incluir_gaiola_seguranca=False,
    )
    cfg_b = ConfigBancada(
        sufixo="B",
        origem_bancada=(0.0, cfg.dy_bancada_b, 0.0),
        pos_origem_rel=(0.25, 0.30, 0.0),
        pos_destino_rel=cfg.pos_destino_b_rel,
        incluir_acessorios_padrao=True,
        incluir_gaiola_seguranca=False,
    )

    bancada_a = inicializar_bancada_padrao(
        config=cfg_a,
        tipo_ferramenta="garra",
        mostrar_primitivas_colisao=mostrar_primitivas_colisao
    )
    bancada_b = inicializar_bancada_padrao(
        config=cfg_b,
        tipo_ferramenta="garra",
        mostrar_primitivas_colisao=mostrar_primitivas_colisao
    )

    # Objeto manipulável inicialmente posicionado na Bancada A
    bancada_a.adicionar_objeto_manipulavel(
        nome="caixa_coop",
        pos_rel=cfg.pos_coleta_a_rel,
        dimensoes=cfg.dim_objeto,
        cor="#D2691E",
        formato="box"
    )

    # Marcador visual da região compartilhada de transferência entre as bancadas
    tx, ty, tz = cfg.pos_transferencia_mundo
    esfera_zona_transf = ub.Ball(
        htm=ub.Utils.trn([tx, ty, tz]),
        radius=0.022,
        name="zona_transferencia_compartilhada",
        color="#00CED1",
        opacity=0.40
    )
    frame_transf = ub.Frame(htm=ub.Utils.trn([tx, ty, tz]), size=0.10)
    extras_visuais: List[ub.SimObject] = [esfera_zona_transf, frame_transf]

    if cfg.usar_estacao_intermediaria:
        h_pedestal = max(0.05, tz - bancada_a.z_superficie - cfg.dim_objeto[2] / 2.0)
        pedestal = ub.Cylinder(
            htm=ub.Utils.trn([tx, ty, bancada_a.z_superficie + h_pedestal / 2.0]),
            radius=0.04,
            height=h_pedestal,
            name="estacao_intermediaria",
            color="#708090",
            opacity=0.85
        )
        extras_visuais.append(pedestal)

    return bancada_a, bancada_b, extras_visuais


def executar_pratica_3(
    config_p3: Optional[ConfigPratica3] = None,
    dy_bancada_b: Optional[float] = None,
    salvar_html: bool = True,
    executar_animacao_browser: bool = False,
    nome_arquivo_html: str = "pratica_3_colaborativa",
    verbose: bool = True
) -> Tuple[BancadaPadrao, BancadaPadrao, ub.Simulation, MetricasPratica3]:
    """
    Executa a simulação colaborativa da Prática 3 com máquina de estados,
    sincronização de garras e filtro QP acoplado de 12 juntas.
    """
    cfg = config_p3 if config_p3 is not None else ConfigPratica3()
    if dy_bancada_b is not None:
        cfg.dy_bancada_b = float(dy_bancada_b)

    bancada_a, bancada_b, extras_visuais = construir_cenario_pratica_3(cfg)
    robot_a = bancada_a.manipulador.robot
    robot_b = bancada_b.manipulador.robot
    garra_a = bancada_a.manipulador.ferramenta
    garra_b = bancada_b.manipulador.ferramenta
    assert isinstance(garra_a, GarraSimulada) and isinstance(garra_b, GarraSimulada)

    caixa = bancada_a.objetos_manipulaveis["caixa_coop"]

    # Orientações ortogonais para handover seguro sem colisão de flanges (Teste_CCR_V7.ipynb)
    rot_a_down = np.asarray(ub.Utils.rotx(np.pi), dtype=float)
    rot_b_perp = np.asarray(ub.Utils.rotx(-np.pi / 2.0), dtype=float)
    rot_b_down = np.asarray(ub.Utils.rotx(np.pi), dtype=float)

    ax, ay, az = cfg.pos_coleta_a_rel
    h_pega_a = az + cfg.dim_objeto[2] / 2.0

    bx, by, bz = cfg.pos_destino_b_rel
    h_solta_b = bz + cfg.dim_objeto[2] / 2.0

    tx, ty, tz = cfg.pos_transferencia_mundo

    # Poses-chave do Robô A (segura a extremidade superior da peça com garra voltada para baixo)
    htm_a_home = bancada_a.pose_rel_para_mundo((0.25, 0.05, 0.28), rot_a_down)
    htm_a_aprox_coleta = bancada_a.pose_rel_para_mundo((ax, ay, 0.24), rot_a_down)
    htm_a_pega = bancada_a.pose_rel_para_mundo((ax, ay, h_pega_a + 0.04), rot_a_down)
    htm_a_transf = np.asarray(ub.Utils.trn([tx, ty, tz + 0.045]) @ rot_a_down, dtype=float)
    htm_a_recuo_pos_transf = np.asarray(ub.Utils.trn([tx, ty + 0.14, tz + 0.14]) @ rot_a_down, dtype=float)

    # Poses-chave do Robô B (pega lateral perpendicular na parte inferior da peça e mantém peça vertical)
    htm_b_home = bancada_b.pose_rel_para_mundo((0.25, 0.05, 0.28), rot_b_perp)
    htm_b_pre_transf = np.asarray(ub.Utils.trn([tx, ty - 0.12, tz - 0.045]) @ rot_b_perp, dtype=float)
    htm_b_transf = np.asarray(ub.Utils.trn([tx, ty - 0.040, tz - 0.045]) @ rot_b_perp, dtype=float)
    htm_b_transporte = bancada_b.pose_rel_para_mundo((bx, by, 0.20), rot_b_perp)
    htm_b_entrega = bancada_b.pose_rel_para_mundo((bx, by, h_solta_b + 0.02), rot_b_perp)
    htm_b_home_final = bancada_b.pose_rel_para_mundo((0.25, 0.05, 0.26), rot_b_perp)

    todos_obstaculos = (
        bancada_a.obter_obstaculos_colisao(incluir_objetos_manipulaveis=True)
        + bancada_b.obter_obstaculos_colisao(incluir_objetos_manipulaveis=False)
    )

    ignored_obs_a = {"rampa_solida_A", "rampa_solida_B"}
    ignored_obs_b = {"caixa_coop", "rampa_solida_A", "rampa_solida_B"}

    params_qp = ParametrosSegurancaQP(
        eta_obs=0.15,
        obs_delta=0.012,
        obs_activation=0.030,
        max_dist_obs=0.06,
        eta_rob=0.18,
        rob_delta=0.015,
        rob_activation=0.055,
        max_dist_rob=0.16,
    )

    rastro_a = ub.Ball(
        htm=ub.Utils.trn(bancada_a.manipulador.posicao_efetuador()),
        radius=0.008,
        name="rastro_robo_a",
        color="cyan"
    )
    rastro_b = ub.Ball(
        htm=ub.Utils.trn(bancada_b.manipulador.posicao_efetuador()),
        radius=0.008,
        name="rastro_robo_b",
        color="magenta"
    )

    sim = ub.Simulation(
        bancada_a.obter_objetos_simulacao()
        + bancada_b.obter_objetos_simulacao()
        + extras_visuais
        + [rastro_a, rastro_b]
    )
    if SHOW_COLLISION_PRIMITIVES:
        robot_a.add_col_object(sim)
        robot_b.add_col_object(sim)

    estado = EstadoCooperacao.FASE_1_APROXIMACAO_COLETA_A
    iter_inicio_estado = 0
    i = 0

    sucesso_coleta_a = False
    sucesso_handover = False
    sucesso_entrega_b = False
    erro_sinc_handover = np.inf
    erro_entrega_b = np.inf
    falhas_qp = 0
    falhas_qp_consecutivas = 0
    motivo_estado_seguro = "Nenhum"

    hist_t: List[float] = []
    hist_estado: List[str] = []
    hist_erro_r1: List[float] = []
    hist_erro_r2: List[float] = []
    hist_dist_rob_ctrl: List[float] = []
    hist_dist_rob_eucl: List[float] = []
    hist_dist_obs_ctrl: List[float] = []
    hist_n_rob_ativas: List[int] = []

    if verbose:
        print(
            f"\n=== Iniciando Prática 3 — Colaboração entre Duas Bancadas "
            f"(DY_BANCADA_B = {cfg.dy_bancada_b:.2f} m) ==="
        )

    while (
        estado not in (EstadoCooperacao.CONCLUIDO, EstadoCooperacao.ESTADO_SEGURO_FALHA)
        and i < cfg.max_iter_total
    ):
        i += 1
        t_atual = i * DT
        iter_local = i - iter_inicio_estado
        allow_handover = False

        # Seleção dos alvos cartesianos conforme o estado da máquina de cooperação
        if estado == EstadoCooperacao.FASE_1_APROXIMACAO_COLETA_A:
            htm_alvo_a = htm_a_aprox_coleta
            htm_alvo_b = htm_b_home
        elif estado == EstadoCooperacao.FASE_2_DESCIDA_E_PEGA_A:
            htm_alvo_a = htm_a_pega
            htm_alvo_b = htm_b_home
            ignored_obs_a.add("caixa_coop")
        elif estado == EstadoCooperacao.FASE_3_TRANSPORTE_ATE_ZONA_TRANSFERENCIA:
            htm_alvo_a = htm_a_transf
            htm_alvo_b = htm_b_pre_transf
        elif estado == EstadoCooperacao.FASE_4_APROXIMACAO_ROBO_B:
            htm_alvo_a = htm_a_transf
            htm_alvo_b = htm_b_transf
            allow_handover = True
        elif estado == EstadoCooperacao.FASE_5_TRANSFERENCIA_SINCRONIZADA:
            htm_alvo_a = htm_a_transf
            htm_alvo_b = htm_b_transf
            allow_handover = True
        elif estado == EstadoCooperacao.FASE_6_RECUO_A_E_TRANSPORTE_B:
            htm_alvo_a = htm_a_recuo_pos_transf
            htm_alvo_b = htm_b_transporte
            allow_handover = (iter_local < 80)
        elif estado == EstadoCooperacao.FASE_7_ENTREGA_DESTINO_B:
            htm_alvo_a = htm_a_home
            htm_alvo_b = htm_b_entrega
        else:  # FASE_8_RETORNO_SEGURO_AMBOS
            htm_alvo_a = htm_a_home
            htm_alvo_b = htm_b_home_final

        q_a = bancada_a.manipulador.ler_juntas()
        q_b = bancada_b.manipulador.ler_juntas()

        bancada_a.atualizar_alvo_visual(htm_alvo_a, tempo_sim=t_atual)
        bancada_b.atualizar_alvo_visual(htm_alvo_b, tempo_sim=t_atual)

        u_nom_a, erro_r1, _, _ = compute_u_nominal(
            _robot=robot_a,
            _q=q_a,
            _htm_tg=htm_alvo_a,
            _lambda_dls=PARAM_LAMBDA_DLS,
            _param_k=PARAM_K
        )
        u_nom_b, erro_r2, _, _ = compute_u_nominal(
            _robot=robot_b,
            _q=q_b,
            _htm_tg=htm_alvo_b,
            _lambda_dls=PARAM_LAMBDA_DLS,
            _param_k=PARAM_K
        )

        if cfg.usar_filtro_qp:
            res_qp = apply_constraints_dual(
                _robot1=robot_a,
                _q1=q_a,
                _u_nom1=u_nom_a,
                _robot2=robot_b,
                _q2=q_b,
                _u_nom2=u_nom_b,
                _all_obstacles=todos_obstaculos,
                ignored_link_for_table=0,
                ignored_obs_names_r1=ignored_obs_a,
                ignored_obs_names_r2=ignored_obs_b,
                robot2_stationary=False,
                allow_tool_handover=allow_handover,
                params=params_qp
            )
            u_a = res_qp.u1
            u_b = res_qp.u2 if res_qp.u2 is not None else np.zeros_like(u_nom_b)
            d_rob_ctrl = res_qp.min_rob_dist_ctrl
            d_obs_ctrl = res_qp.min_obs_dist_ctrl
            d_self_ctrl = res_qp.min_self_dist_ctrl
            n_rob_act = res_qp.n_rob_constraints

            if not res_qp.qp_viavel:
                falhas_qp += 1
                falhas_qp_consecutivas += 1
                if falhas_qp_consecutivas >= 5:
                    estado = EstadoCooperacao.ESTADO_SEGURO_FALHA
                    motivo_estado_seguro = "Inviabilidade consecutiva do controlador QP de 12 juntas."
                    if verbose:
                        print(f"  [ALERTA DE SEGURANÇA] {motivo_estado_seguro}")
                    break
            else:
                falhas_qp_consecutivas = 0
        else:
            u_a = u_nom_a.copy()
            u_b = u_nom_b.copy()
            d_rob_ctrl = compute_min_rob_to_rob_distance_euclidean(robot_a, q_a, robot_b, q_b)
            d_obs_ctrl = min(
                compute_min_obs_distance_euclidean(robot_a, q_a, todos_obstaculos, ignored_obs_names=ignored_obs_a),
                compute_min_obs_distance_euclidean(robot_b, q_b, todos_obstaculos, ignored_obs_names=ignored_obs_b),
            )
            d_self_ctrl = min(
                compute_min_self_distance_euclidean(robot_a, q_a),
                compute_min_self_distance_euclidean(robot_b, q_b),
            )
            n_rob_act = 0

        gravar_frame = (i % ANIMATION_STEP == 0)
        q_a_novo = bancada_a.manipulador.aplicar_velocidade_articular(
            u=u_a, dt=DT, tempo_sim=t_atual, registrar_animacao=gravar_frame
        )
        q_b_novo = bancada_b.manipulador.aplicar_velocidade_articular(
            u=u_b, dt=DT, tempo_sim=t_atual, registrar_animacao=gravar_frame
        )

        htm_eef_a = bancada_a.manipulador.pose_efetuador(q_a_novo)
        htm_eef_b = bancada_b.manipulador.pose_efetuador(q_b_novo)
        p_eef_a = htm_eef_a[0:3, 3].reshape(3)
        p_eef_b = htm_eef_b[0:3, 3].reshape(3)

        if gravar_frame:
            rastro_a.add_ani_frame(time=t_atual, htm=ub.Utils.trn(p_eef_a))
            rastro_b.add_ani_frame(time=t_atual, htm=ub.Utils.trn(p_eef_b))
            if caixa.preso_por is None:
                caixa.atualizar_pose(caixa.htm_atual, tempo_sim=t_atual)

        erro_pos_a, erro_ori_a = calcular_erros_pose(htm_eef_a, htm_alvo_a)
        erro_pos_b, erro_ori_b = calcular_erros_pose(htm_eef_b, htm_alvo_b)

        d_rob_eucl = compute_min_rob_to_rob_distance_euclidean(robot_a, q_a_novo, robot_b, q_b_novo)

        hist_t.append(t_atual)
        hist_estado.append(estado.name)
        hist_erro_r1.append(erro_r1)
        hist_erro_r2.append(erro_r2)
        hist_dist_rob_ctrl.append(d_rob_ctrl)
        hist_dist_rob_eucl.append(d_rob_eucl)
        hist_dist_obs_ctrl.append(d_obs_ctrl)
        hist_n_rob_ativas.append(n_rob_act)

        conv_a = (erro_pos_a <= 0.004 and erro_ori_a <= 0.05) or (erro_r1 <= cfg.tol_erro_tarefa)
        conv_b = (erro_pos_b <= 0.004 and erro_ori_b <= 0.05) or (erro_r2 <= cfg.tol_erro_tarefa)
        timeout_estado = iter_local >= cfg.max_iter_por_estado

        # Transições da Máquina de Estados
        if estado == EstadoCooperacao.FASE_1_APROXIMACAO_COLETA_A:
            if conv_a or timeout_estado:
                if verbose:
                    print(f"  [Estado 1 -> 2] Robô A posicionado sobre a coleta (erro_pos={erro_pos_a*1000:.2f} mm)")
                estado = EstadoCooperacao.FASE_2_DESCIDA_E_PEGA_A
                iter_inicio_estado = i

        elif estado == EstadoCooperacao.FASE_2_DESCIDA_E_PEGA_A:
            if conv_a or timeout_estado:
                sucesso_coleta_a = garra_a.fechar(
                    robot=robot_a,
                    q=q_a_novo,
                    objeto_candidato=caixa,
                    tempo_sim=t_atual,
                    obstaculos_ignorados=ignored_obs_a,
                    forcar_apreensao=False
                )
                if not sucesso_coleta_a:
                    estado = EstadoCooperacao.ESTADO_SEGURO_FALHA
                    motivo_estado_seguro = "Falha na apreensão inicial pelo Robô A (objeto fora de alcance)."
                    if verbose:
                        print(f"  [ESTADO SEGURO] {motivo_estado_seguro}")
                    break
                if verbose:
                    print(f"  [Estado 2 -> 3] Robô A coletou o objeto (erro_pos={erro_pos_a*1000:.2f} mm)")
                estado = EstadoCooperacao.FASE_3_TRANSPORTE_ATE_ZONA_TRANSFERENCIA
                iter_inicio_estado = i

        elif estado == EstadoCooperacao.FASE_3_TRANSPORTE_ATE_ZONA_TRANSFERENCIA:
            if (conv_a and conv_b) or timeout_estado:
                if verbose:
                    print(
                        f"  [Estado 3 -> 4] Robô A chegou na região de transferência "
                        f"(erro_A={erro_pos_a*1000:.2f} mm). Robô B iniciando aproximação final."
                    )
                estado = EstadoCooperacao.FASE_4_APROXIMACAO_ROBO_B
                iter_inicio_estado = i

        elif estado == EstadoCooperacao.FASE_4_APROXIMACAO_ROBO_B:
            if (conv_a and conv_b) or timeout_estado:
                erro_sinc_handover = float(max(erro_pos_a, erro_pos_b))
                if verbose:
                    print(
                        f"  [Estado 4 -> 5] Robô B alinhado para transferência "
                        f"(erro_sinc={erro_sinc_handover*1000:.2f} mm)"
                    )
                estado = EstadoCooperacao.FASE_5_TRANSFERENCIA_SINCRONIZADA
                iter_inicio_estado = i

        elif estado == EstadoCooperacao.FASE_5_TRANSFERENCIA_SINCRONIZADA:
            # Condição estrita de sincronização: verifica convergência simultânea de A e B na peça antes de liberar A!
            erro_sinc_handover = float(max(erro_pos_a, erro_pos_b))
            if erro_sinc_handover <= cfg.tol_sincronizacao_m:
                # 1º Robô B fecha a garra segurando a peça
                apreendeu_b = garra_b.fechar(
                    robot=robot_b,
                    q=q_b_novo,
                    objeto_candidato=caixa,
                    tempo_sim=t_atual,
                    obstaculos_ignorados=ignored_obs_b,
                    forcar_apreensao=True
                )
                if apreendeu_b:
                    # 2º Somente após confirmação de pega do Robô B, Robô A abre a garra (sem soltar o objeto no ar)
                    garra_a._remover_colisao_acoplada(robot_a)
                    garra_a.aberta = True
                    garra_a.abertura_atual_mm = garra_a.abertura_max_mm
                    garra_a.objeto_apreendido = None
                    # Mantém posse lógica e cinemática com o Robô B
                    caixa.preso_por = robot_b.name
                    sucesso_handover = True
                    if verbose:
                        print(
                            f"  [Estado 5 -> 6] Transferência sincronizada concluída com sucesso! "
                            f"(erro_sinc={erro_sinc_handover*1000:.2f} mm)"
                        )
                    estado = EstadoCooperacao.FASE_6_RECUO_A_E_TRANSPORTE_B
                    iter_inicio_estado = i
                else:
                    estado = EstadoCooperacao.ESTADO_SEGURO_FALHA
                    motivo_estado_seguro = "Robô B não confirmou apreensão; Robô A manteve garra fechada por segurança."
                    break
            elif timeout_estado:
                estado = EstadoCooperacao.ESTADO_SEGURO_FALHA
                motivo_estado_seguro = (
                    f"Falha de sincronização no handover: erro de alinhamento ({erro_sinc_handover*1000:.1f} mm) "
                    f"acima da tolerância ({cfg.tol_sincronizacao_m*1000:.1f} mm). Objeto retido pelo Robô A."
                )
                if verbose:
                    print(f"  [ESTADO SEGURO] {motivo_estado_seguro}")
                break

        elif estado == EstadoCooperacao.FASE_6_RECUO_A_E_TRANSPORTE_B:
            if (conv_a and conv_b) or timeout_estado:
                if verbose:
                    print(
                        f"  [Estado 6 -> 7] Robô A em recuo seguro e Robô B sobre o destino "
                        f"(erro_B={erro_pos_b*1000:.2f} mm)"
                    )
                estado = EstadoCooperacao.FASE_7_ENTREGA_DESTINO_B
                iter_inicio_estado = i

        elif estado == EstadoCooperacao.FASE_7_ENTREGA_DESTINO_B:
            if conv_b or timeout_estado:
                garra_b.abrir(
                    robot=robot_b,
                    q=q_b_novo,
                    tempo_sim=t_atual,
                    obstaculos_ignorados=ignored_obs_b
                )
                erro_entrega_b = float(erro_pos_b)
                sucesso_entrega_b = erro_entrega_b <= 0.015
                if verbose:
                    print(
                        f"  [Estado 7 -> 8] Robô B liberou o objeto no destino da Bancada B "
                        f"(erro_entrega={erro_entrega_b*1000:.2f} mm)"
                    )
                estado = EstadoCooperacao.FASE_8_RETORNO_SEGURO_AMBOS
                iter_inicio_estado = i

        elif estado == EstadoCooperacao.FASE_8_RETORNO_SEGURO_AMBOS:
            if (conv_a and conv_b) or timeout_estado:
                if verbose:
                    print("  [Estado 8 -> CONCLUIDO] Ambos os robôs retornaram às posições seguras.")
                estado = EstadoCooperacao.CONCLUIDO
                break

    caminho_salvo = None
    if salvar_html:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sim.save(OUTPUT_DIR, file_name=nome_arquivo_html)
        caminho_salvo = os.path.join(OUTPUT_DIR, f"{nome_arquivo_html}.html")

    if executar_animacao_browser:
        sim.run()

    metricas = MetricasPratica3(
        estado_final=estado.name,
        sucesso_coleta_a=sucesso_coleta_a,
        sucesso_sincronizacao_handover=sucesso_handover,
        sucesso_entrega_b=sucesso_entrega_b,
        iteracoes_totais=len(hist_t),
        tempo_total_s=hist_t[-1] if hist_t else 0.0,
        erro_sincronizacao_handover_m=erro_sinc_handover,
        erro_entrega_final_b_m=erro_entrega_b,
        min_dist_rob_ctrl_m=float(np.min(hist_dist_rob_ctrl)) if hist_dist_rob_ctrl else np.inf,
        min_dist_rob_eucl_m=float(np.min(hist_dist_rob_eucl)) if hist_dist_rob_eucl else np.inf,
        min_dist_obs_ctrl_m=float(np.min(hist_dist_obs_ctrl)) if hist_dist_obs_ctrl else np.inf,
        min_dist_self_ctrl_m=float(
            min(
                compute_min_self_distance_euclidean(robot_a, bancada_a.manipulador.ler_juntas()),
                compute_min_self_distance_euclidean(robot_b, bancada_b.manipulador.ler_juntas()),
            )
        ),
        max_restricoes_rob_ativas=int(np.max(hist_n_rob_ativas)) if hist_n_rob_ativas else 0,
        total_ativacoes_qp_rob=int(np.sum(np.array(hist_n_rob_ativas) > 0)) if hist_n_rob_ativas else 0,
        falhas_qp=falhas_qp,
        motivo_estado_seguro=motivo_estado_seguro,
        caminho_html=caminho_salvo,
        hist_t=hist_t,
        hist_estado=hist_estado,
        hist_erro_r1=hist_erro_r1,
        hist_erro_r2=hist_erro_r2,
        hist_dist_rob_ctrl=hist_dist_rob_ctrl,
        hist_dist_rob_eucl=hist_dist_rob_eucl,
        hist_dist_obs_ctrl=hist_dist_obs_ctrl,
        hist_n_rob_ativas=hist_n_rob_ativas,
    )

    if verbose:
        print(
            f"=== Resumo Prática 3 (Colaboração 2 Bancadas) ===\n"
            f"  Estado Final: {metricas.estado_final}\n"
            f"  Iterações: {metricas.iteracoes_totais} ({metricas.tempo_total_s:.2f} s)\n"
            f"  Coleta Robô A: {metricas.sucesso_coleta_a} | "
            f"Handover A->B: {metricas.sucesso_sincronizacao_handover} "
            f"(erro_sinc={metricas.erro_sincronizacao_handover_m*1000:.2f} mm) | "
            f"Entrega Robô B: {metricas.sucesso_entrega_b} "
            f"(erro={metricas.erro_entrega_final_b_m*1000:.2f} mm)\n"
            f"  Menor distância entre Robô A e Robô B (controlador): {metricas.min_dist_rob_ctrl_m*1000:.2f} mm\n"
            f"  Menor distância entre Robô A e Robô B (euclidiana):  {metricas.min_dist_rob_eucl_m*1000:.2f} mm\n"
            f"  Menor distância a obstáculos: {metricas.min_dist_obs_ctrl_m*1000:.2f} mm\n"
            f"  Passos com restrição acoplada Robô A <-> Robô B ativa: {metricas.total_ativacoes_qp_rob} "
            f"(máx {metricas.max_restricoes_rob_ativas} simultâneas)\n"
            f"  Arquivo HTML: {metricas.caminho_html}"
        )

    return bancada_a, bancada_b, sim, metricas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prática 3 — Colaboração entre duas bancadas e dois manipuladores no UAIBot."
    )
    parser.add_argument(
        "--dy-bancada",
        type=float,
        default=-0.66,
        help="Distância lateral Y entre a Bancada A e a Bancada B em metros (padrão: -0.66)."
    )
    parser.add_argument(
        "--estacao-intermediaria",
        action="store_true",
        help="Inclui estação/pedestal intermediário na zona compartilhada de transferência."
    )
    args = parser.parse_args()

    cfg = ConfigPratica3(
        dy_bancada_b=args.dy_bancada,
        usar_estacao_intermediaria=args.estacao_intermediaria,
    )
    executar_pratica_3(config_p3=cfg, salvar_html=True, verbose=True)


if __name__ == "__main__":
    main()

