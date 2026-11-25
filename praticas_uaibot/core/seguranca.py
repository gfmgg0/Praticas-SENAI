"""
Módulo de segurança, distâncias diferenciáveis/euclidianas e filtro QP/CBF
(reaproveitado integralmente de Teste_CCR_V7.ipynb, com suporte para 1 robô (6 juntas)
e 2 robôs colaborativos (12 juntas) + tratamento seguro de falha do QP).
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import uaibot as ub
from cvxopt import solvers

from ..config.parametros import (
    MAX_VEL_MODULO,
    PARAM_AUTO_ACTIVATION,
    PARAM_AUTO_DELTA,
    PARAM_EPS_DIST_CTRL,
    PARAM_EPS_DIST_PLOT,
    PARAM_ETA_AUTO,
    PARAM_ETA_JOINT,
    PARAM_ETA_OBS,
    PARAM_ETA_ROB,
    PARAM_H_DIST_CTRL,
    PARAM_H_DIST_PLOT,
    PARAM_JOINT_DELTA,
    PARAM_MAX_DIST_OBS,
    PARAM_MAX_DIST_ROB,
    PARAM_NO_ITER_MAX_DIST_CTRL,
    PARAM_NO_ITER_MAX_DIST_PLOT,
    PARAM_OBS_ACTIVATION,
    PARAM_OBS_CONTACT_TOL,
    PARAM_OBS_DELTA,
    PARAM_ROB_ACTIVATION,
    PARAM_ROB_DELTA,
    PARAM_TOL_DIST_CTRL,
    PARAM_TOL_DIST_PLOT,
)
from .manipulador import sanitize_joint_limits

solvers.options["show_progress"] = False


@dataclass
class ParametrosSegurancaQP:
    """Parâmetros ajustáveis do filtro QP/CBF para experimentação didática."""
    eta_joint: float = PARAM_ETA_JOINT
    eta_auto: float = PARAM_ETA_AUTO
    eta_obs: float = PARAM_ETA_OBS
    eta_rob: float = PARAM_ETA_ROB
    joint_delta: float = PARAM_JOINT_DELTA
    auto_delta: float = PARAM_AUTO_DELTA
    auto_activation: float = PARAM_AUTO_ACTIVATION
    obs_delta: float = PARAM_OBS_DELTA
    obs_activation: float = PARAM_OBS_ACTIVATION
    obs_contact_tol: float = PARAM_OBS_CONTACT_TOL
    max_dist_obs: float = PARAM_MAX_DIST_OBS
    rob_delta: float = PARAM_ROB_DELTA
    rob_activation: float = PARAM_ROB_ACTIVATION
    max_dist_rob: float = PARAM_MAX_DIST_ROB


@dataclass
class ResultadoFiltroQP:
    """Estrutura padronizada de saída do filtro de segurança QP/CBF."""
    u1: np.ndarray
    u2: Optional[np.ndarray]
    min_self_dist_ctrl: float
    n_self_constraints: int
    min_obs_dist_ctrl: float
    n_obs_constraints: int
    min_rob_dist_ctrl: float
    n_rob_constraints: int
    time_self_dist: float
    time_obs_dist_total: float
    time_rob_dist: float
    time_qp: float
    time_apply_constraints: float
    qp_viavel: bool = True
    mensagem_status: str = "OK"
    detalhes_restricoes: List[Tuple] = field(default_factory=list)


def get_link_indices_from_dist_struct(dist_struct) -> Optional[List[int]]:
    if hasattr(dist_struct, "_list_info"):
        return [int(item.link_number) for item in dist_struct._list_info]
    return None


def should_ignore_obs_constraint(
    obstacle: ub.SimObject,
    link_idx: Optional[int],
    ignored_link_for_table: int = 0,
    ignored_obs_names: Optional[Set[str]] = None
) -> bool:
    if ignored_obs_names and obstacle.name in ignored_obs_names:
        return True
    if obstacle.name.startswith("mesa") and link_idx == ignored_link_for_table:
        return True
    return False


def extract_robot_col_objects(_robot: ub.Robot, _q: np.ndarray) -> List[Tuple[int, int, ub.SimObject]]:
    """
    Atualiza a pose (HTM) dos objetos de colisão do robô para a configuração _q
    usando apenas a cinemática direta (fkm), SEM apagar os frames de animação
    (reaproveitado de Teste_CCR_V7.ipynb).
    """
    htm_dh = _robot.fkm(q=_q, axis="dh")
    col_objs = []

    for link_idx, link in enumerate(_robot.links):
        htm_link = htm_dh[link_idx]
        for sub_idx, col_item in enumerate(link.col_objects):
            obj_3d = col_item[0]
            htm_local = col_item[1]
            obj_3d._htm = htm_link @ htm_local
            col_objs.append((link_idx, sub_idx, obj_3d))

    return col_objs


def compute_min_self_distance_euclidean(_robot: ub.Robot, _q: np.ndarray) -> float:
    ds_auto_plot = _robot.compute_dist_auto(
        q=_q,
        h=PARAM_H_DIST_PLOT,
        eps=PARAM_EPS_DIST_PLOT,
        tol=PARAM_TOL_DIST_PLOT,
        no_iter_max=PARAM_NO_ITER_MAX_DIST_PLOT
    )
    dist_auto_plot = np.asarray(ds_auto_plot.dist_vect, dtype=float).reshape(-1)
    if dist_auto_plot.size == 0:
        return np.inf
    return float(np.min(dist_auto_plot))


def compute_min_obs_distance_euclidean(
    _robot: ub.Robot,
    _q: np.ndarray,
    _all_obstacles: List[ub.SimObject],
    ignored_link_for_table: int = 0,
    ignored_obs_names: Optional[Set[str]] = None,
    max_dist: float = PARAM_MAX_DIST_OBS
) -> float:
    min_obs_dist_eucl = np.inf

    for obstacle in _all_obstacles:
        if ignored_obs_names and obstacle.name in ignored_obs_names:
            continue

        ds_obs_plot = _robot.compute_dist(
            q=_q,
            obj=obstacle,
            h=PARAM_H_DIST_PLOT,
            eps=PARAM_EPS_DIST_PLOT,
            tol=PARAM_TOL_DIST_PLOT,
            no_iter_max=PARAM_NO_ITER_MAX_DIST_PLOT,
            max_dist=max_dist
        )

        dist_obs_plot = np.asarray(ds_obs_plot.dist_vect, dtype=float).reshape(-1)
        if dist_obs_plot.size == 0:
            continue

        link_idx_list = get_link_indices_from_dist_struct(ds_obs_plot)

        for k in range(dist_obs_plot.shape[0]):
            d_k = float(dist_obs_plot[k])
            link_idx = None
            if link_idx_list is not None and k < len(link_idx_list):
                link_idx = int(link_idx_list[k])

            if should_ignore_obs_constraint(
                obstacle=obstacle,
                link_idx=link_idx,
                ignored_link_for_table=ignored_link_for_table,
                ignored_obs_names=ignored_obs_names
            ):
                continue

            min_obs_dist_eucl = min(min_obs_dist_eucl, d_k)

    return min_obs_dist_eucl


def compute_min_rob_to_rob_distance_euclidean(
    _robot1: ub.Robot,
    _q1: np.ndarray,
    _robot2: ub.Robot,
    _q2: np.ndarray,
    max_dist: float = PARAM_MAX_DIST_ROB,
    ignorar_objetos_acoplados: bool = True
) -> float:
    min_rob_dist_eucl = np.inf
    col_objs_2 = extract_robot_col_objects(_robot2, _q2)

    for _, _, obj_2 in col_objs_2:
        if ignorar_objetos_acoplados and "_acoplado_" in obj_2.name:
            continue
        ds_plot = _robot1.compute_dist(
            q=_q1,
            obj=obj_2,
            h=PARAM_H_DIST_PLOT,
            eps=PARAM_EPS_DIST_PLOT,
            tol=PARAM_TOL_DIST_PLOT,
            no_iter_max=PARAM_NO_ITER_MAX_DIST_PLOT,
            max_dist=max_dist
        )
        dist_vect = np.asarray(ds_plot.dist_vect, dtype=float).reshape(-1)
        if dist_vect.size > 0:
            if ignorar_objetos_acoplados and hasattr(ds_plot, "_list_info"):
                for k, info in enumerate(ds_plot._list_info):
                    lnk_idx = int(info.link_number)
                    sub_idx = int(getattr(info, "link_col_obj_number", getattr(info, "obj_col_number", 0)))
                    if sub_idx < len(_robot1.links[lnk_idx].col_objects):
                        obj_1 = _robot1.links[lnk_idx].col_objects[sub_idx][0]
                        if "_acoplado_" in obj_1.name:
                            continue
                    min_rob_dist_eucl = min(min_rob_dist_eucl, float(dist_vect[k]))
            else:
                min_rob_dist_eucl = min(min_rob_dist_eucl, float(np.min(dist_vect)))

    return min_rob_dist_eucl


def _build_single_robot_constraints(
    _robot: ub.Robot,
    _q: np.ndarray,
    _all_obstacles: List[ub.SimObject],
    col_offset: int,
    total_joints: int,
    robot_label: str,
    ignored_link_for_table: int = 0,
    ignored_obs_names: Optional[Set[str]] = None,
    params: Optional[ParametrosSegurancaQP] = None
):
    """
    Monta os blocos de restrições individuais de um robô (limites articulares,
    velocidade, autocolisão e obstáculos estáticos) posicionados em
    col_offset : col_offset + no_joint (reaproveitado de Teste_CCR_V7.ipynb).
    """
    if params is None:
        params = ParametrosSegurancaQP()

    _q = np.asarray(_q, dtype=float).reshape(-1, 1)
    no_joint = np.shape(_robot.q)[0]
    joint_limits = sanitize_joint_limits(_robot.joint_limit)

    A_list = []
    b_list = []
    debug_rows = []

    def pad_cols(mat_local):
        mat_full = np.zeros((mat_local.shape[0], total_joints), dtype=float)
        mat_full[:, col_offset:col_offset + no_joint] = mat_local
        return mat_full

    # 1. LIMITES ARTICULARES
    A1 = pad_cols(np.eye(no_joint))
    b1 = -params.eta_joint * (_q - joint_limits[:, [0]] - params.joint_delta)

    A2 = pad_cols(-np.eye(no_joint))
    b2 = -params.eta_joint * (joint_limits[:, [1]] - _q - params.joint_delta)

    A_list.extend([A1, A2])
    b_list.extend([b1, b2])

    # 2. LIMITES DE VELOCIDADE
    A3 = pad_cols(np.eye(no_joint))
    b3 = -MAX_VEL_MODULO

    A4 = pad_cols(-np.eye(no_joint))
    b4 = -MAX_VEL_MODULO

    A_list.extend([A3, A4])
    b_list.extend([b3, b4])

    # 3. AUTOCOLISÃO
    min_self_dist_ctrl = np.inf
    n_self_constraints = 0

    t0 = time.perf_counter()
    ds_auto = _robot.compute_dist_auto(
        q=_q,
        h=PARAM_H_DIST_CTRL,
        eps=PARAM_EPS_DIST_CTRL,
        tol=PARAM_TOL_DIST_CTRL,
        no_iter_max=PARAM_NO_ITER_MAX_DIST_CTRL
    )
    time_self_dist = time.perf_counter() - t0

    jac_auto = np.asarray(ds_auto.jac_dist_mat, dtype=float)
    dist_auto = np.asarray(ds_auto.dist_vect, dtype=float).reshape(-1, 1)

    if jac_auto.ndim == 1 and jac_auto.size > 0:
        jac_auto = jac_auto.reshape(1, -1)
    if jac_auto.size > 0 and jac_auto.shape[1] != no_joint:
        jac_auto = jac_auto.reshape(-1, no_joint)

    if dist_auto.size > 0:
        min_self_dist_ctrl = float(np.min(dist_auto))

    for k in range(dist_auto.shape[0]):
        d_k = float(dist_auto[k, 0])
        if d_k <= params.auto_activation:
            rhs = -params.eta_auto * (d_k - params.auto_delta)
            A_list.append(pad_cols(jac_auto[k:k + 1, :]))
            b_list.append(np.array([[rhs]], dtype=float))
            debug_rows.append((f"{robot_label}_self", k, d_k, rhs))
            n_self_constraints += 1

    # 4. OBSTÁCULOS EXTERNOS
    min_obs_dist_ctrl = np.inf
    n_obs_constraints = 0
    time_obs_dist_total = 0.0
    time_obs_dist_by_obj: Dict[str, float] = {}

    for obstacle in _all_obstacles:
        if ignored_obs_names and obstacle.name in ignored_obs_names:
            continue
        t0 = time.perf_counter()
        ds_obs = _robot.compute_dist(
            q=_q,
            obj=obstacle,
            h=PARAM_H_DIST_CTRL,
            eps=PARAM_EPS_DIST_CTRL,
            tol=PARAM_TOL_DIST_CTRL,
            no_iter_max=PARAM_NO_ITER_MAX_DIST_CTRL,
            max_dist=params.max_dist_obs
        )
        dt_obs = time.perf_counter() - t0
        time_obs_dist_total += dt_obs
        time_obs_dist_by_obj[obstacle.name] = time_obs_dist_by_obj.get(obstacle.name, 0.0) + dt_obs

        jac_obs = np.asarray(ds_obs.jac_dist_mat, dtype=float)
        dist_obs = np.asarray(ds_obs.dist_vect, dtype=float).reshape(-1, 1)

        if jac_obs.ndim == 1 and jac_obs.size > 0:
            jac_obs = jac_obs.reshape(1, -1)
        if jac_obs.size > 0 and jac_obs.shape[1] != no_joint:
            jac_obs = jac_obs.reshape(-1, no_joint)

        link_idx_list = get_link_indices_from_dist_struct(ds_obs)

        for k in range(dist_obs.shape[0]):
            d_k = float(dist_obs[k, 0])
            link_idx = None
            if link_idx_list is not None and k < len(link_idx_list):
                link_idx = int(link_idx_list[k])

            if should_ignore_obs_constraint(
                obstacle=obstacle,
                link_idx=link_idx,
                ignored_link_for_table=ignored_link_for_table,
                ignored_obs_names=ignored_obs_names
            ):
                continue

            min_obs_dist_ctrl = min(min_obs_dist_ctrl, d_k)

            if d_k <= params.obs_activation and d_k > params.obs_contact_tol:
                rhs = -params.eta_obs * (d_k - params.obs_delta)
                A_list.append(pad_cols(jac_obs[k:k + 1, :]))
                b_list.append(np.array([[rhs]], dtype=float))
                debug_rows.append((f"{robot_label}_obs", obstacle.name, k, link_idx, d_k, rhs))
                n_obs_constraints += 1

    A_rob = np.vstack(A_list) if len(A_list) > 0 else np.zeros((0, total_joints), dtype=float)
    b_rob = np.vstack(b_list) if len(b_list) > 0 else np.zeros((0, 1), dtype=float)

    return (
        A_rob,
        b_rob,
        debug_rows,
        min_self_dist_ctrl,
        n_self_constraints,
        min_obs_dist_ctrl,
        n_obs_constraints,
        time_self_dist,
        time_obs_dist_total,
        time_obs_dist_by_obj
    )


def apply_constraints_single(
    _robot: ub.Robot,
    _q: np.ndarray,
    _u_nom: np.ndarray,
    _all_obstacles: List[ub.SimObject],
    ignored_link_for_table: int = 0,
    ignored_obs_names: Optional[Set[str]] = None,
    params: Optional[ParametrosSegurancaQP] = None
) -> ResultadoFiltroQP:
    """
    Aplica o filtro de segurança QP/CBF para um único manipulador (6 juntas),
    reaproveitando integralmente a montagem de restrições de _build_single_robot_constraints.
    """
    t_apply_ini = time.perf_counter()
    if params is None:
        params = ParametrosSegurancaQP()

    _q = np.asarray(_q, dtype=float).reshape(-1, 1)
    _u_nom = np.asarray(_u_nom, dtype=float).reshape(-1, 1)
    n_joints = _q.shape[0]

    (
        A, b, dbg_rows,
        min_self, n_self,
        min_obs, n_obs,
        t_self, t_obs, _
    ) = _build_single_robot_constraints(
        _robot=_robot,
        _q=_q,
        _all_obstacles=_all_obstacles,
        col_offset=0,
        total_joints=n_joints,
        robot_label="R1",
        ignored_link_for_table=ignored_link_for_table,
        ignored_obs_names=ignored_obs_names,
        params=params
    )

    H = 2.0 * np.eye(n_joints)
    f = -2.0 * _u_nom

    qp_viavel = True
    msg = "OK"
    t0_qp = time.perf_counter()
    try:
        u_sol = ub.Utils.solve_qp(
            np.matrix(H),
            np.matrix(f),
            np.matrix(A),
            np.matrix(b),
            None,
            None
        )
        time_qp = time.perf_counter() - t0_qp
        if u_sol is None:
            raise RuntimeError("solve_qp retornou None.")
        u1 = np.asarray(u_sol, dtype=float).reshape(-1, 1)
        if np.any(~np.isfinite(u1)):
            raise RuntimeError("Solução do QP contém valores não finitos.")
    except Exception as exc:
        time_qp = time.perf_counter() - t0_qp
        qp_viavel = False
        msg = f"Fallback estado seguro (QP inviável: {exc})"
        u1 = np.zeros_like(_u_nom)

    time_apply = time.perf_counter() - t_apply_ini

    return ResultadoFiltroQP(
        u1=u1,
        u2=None,
        min_self_dist_ctrl=min_self,
        n_self_constraints=n_self,
        min_obs_dist_ctrl=min_obs,
        n_obs_constraints=n_obs,
        min_rob_dist_ctrl=np.inf,
        n_rob_constraints=0,
        time_self_dist=t_self,
        time_obs_dist_total=t_obs,
        time_rob_dist=0.0,
        time_qp=time_qp,
        time_apply_constraints=time_apply,
        qp_viavel=qp_viavel,
        mensagem_status=msg,
        detalhes_restricoes=dbg_rows
    )


def apply_constraints_dual(
    _robot1: ub.Robot,
    _q1: np.ndarray,
    _u_nom1: np.ndarray,
    _robot2: ub.Robot,
    _q2: np.ndarray,
    _u_nom2: np.ndarray,
    _all_obstacles: List[ub.SimObject],
    ignored_link_for_table: int = 0,
    ignored_obs_names_r1: Optional[Set[str]] = None,
    ignored_obs_names_r2: Optional[Set[str]] = None,
    robot2_stationary: bool = False,
    allow_tool_handover: bool = False,
    params: Optional[ParametrosSegurancaQP] = None
) -> ResultadoFiltroQP:
    """
    Aplica o filtro de segurança QP/CBF conjunto para 2 manipuladores (12 juntas),
    incluindo autocolisão, obstáculos das bancadas e acoplamento Robô 1 <-> Robô 2
    (reaproveitado de Teste_CCR_V7.ipynb, com fallback seguro caso o QP falhe).
    """
    t_apply_ini = time.perf_counter()
    if params is None:
        params = ParametrosSegurancaQP()

    _q1 = np.asarray(_q1, dtype=float).reshape(-1, 1)
    _q2 = np.asarray(_q2, dtype=float).reshape(-1, 1)
    _u_nom1 = np.asarray(_u_nom1, dtype=float).reshape(-1, 1)
    _u_nom2 = np.asarray(_u_nom2, dtype=float).reshape(-1, 1)
    if robot2_stationary:
        _u_nom2 = np.zeros_like(_u_nom2)

    n1 = _q1.shape[0]
    n2 = _q2.shape[0]
    total_joints = n1 + n2

    (
        A_r1, b_r1, dbg_r1,
        min_self_1, n_self_1,
        min_obs_1, n_obs_1,
        t_self_1, t_obs_1, _
    ) = _build_single_robot_constraints(
        _robot=_robot1,
        _q=_q1,
        _all_obstacles=_all_obstacles,
        col_offset=0,
        total_joints=total_joints,
        robot_label="R1",
        ignored_link_for_table=ignored_link_for_table,
        ignored_obs_names=ignored_obs_names_r1,
        params=params
    )

    (
        A_r2, b_r2, dbg_r2,
        min_self_2, n_self_2,
        min_obs_2, n_obs_2,
        t_self_2, t_obs_2, _
    ) = _build_single_robot_constraints(
        _robot=_robot2,
        _q=_q2,
        _all_obstacles=_all_obstacles,
        col_offset=n1,
        total_joints=total_joints,
        robot_label="R2",
        ignored_link_for_table=ignored_link_for_table,
        ignored_obs_names=ignored_obs_names_r2,
        params=params
    )

    A_list = [A_r1, A_r2]
    b_list = [b_r1, b_r2]
    debug_rows = dbg_r1 + dbg_r2

    # 5. COLISÃO ENTRE ROBÔ 1 E ROBÔ 2 (Acoplamento 12 juntas)
    t0_rob = time.perf_counter()
    min_rob_dist_ctrl = np.inf
    n_rob_constraints = 0

    col_objs_r1 = extract_robot_col_objects(_robot1, _q1)
    col_objs_r2 = extract_robot_col_objects(_robot2, _q2)

    jac_r1_map = {}
    dist_r1_map = {}

    for link2_idx, sub2_idx, obj2 in col_objs_r2:
        ds_12 = _robot1.compute_dist(
            q=_q1,
            obj=obj2,
            h=PARAM_H_DIST_CTRL,
            eps=PARAM_EPS_DIST_CTRL,
            tol=PARAM_TOL_DIST_CTRL,
            no_iter_max=PARAM_NO_ITER_MAX_DIST_CTRL,
            max_dist=params.max_dist_rob
        )
        jac_12 = np.asarray(ds_12.jac_dist_mat, dtype=float)
        dist_12 = np.asarray(ds_12.dist_vect, dtype=float).reshape(-1, 1)

        if jac_12.ndim == 1 and jac_12.size > 0:
            jac_12 = jac_12.reshape(1, -1)
        if jac_12.size > 0 and jac_12.shape[1] != n1:
            jac_12 = jac_12.reshape(-1, n1)

        if hasattr(ds_12, "_list_info"):
            for k, item in enumerate(ds_12._list_info):
                lnk1_idx = int(item.link_number)
                sub1_idx = int(getattr(item, "link_col_obj_number", getattr(item, "obj_col_number", 0)))
                key = (lnk1_idx, sub1_idx, link2_idx, sub2_idx)
                d_val = float(dist_12[k, 0])

                # Durante a entrega (handover), ignora apenas o contato entre a peça acoplada e a garra receptora,
                # preservando a proteção integral entre os corpos físicos das duas garras e elos dos robôs.
                if allow_tool_handover:
                    nome_obj1 = (
                        _robot1.links[lnk1_idx].col_objects[sub1_idx][0].name
                        if sub1_idx < len(_robot1.links[lnk1_idx].col_objects)
                        else ""
                    )
                    if ("_acoplado_" in nome_obj1) or ("_acoplado_" in obj2.name):
                        continue

                min_rob_dist_ctrl = min(min_rob_dist_ctrl, d_val)
                if d_val <= params.rob_activation:
                    jac_r1_map[key] = jac_12[k:k + 1, :]
                    dist_r1_map[key] = d_val

    jac_r2_map = {}
    if not robot2_stationary and len(jac_r1_map) > 0:
        active_r1_subobjs = {(k[0], k[1]) for k in jac_r1_map.keys()}
        for link1_idx, sub1_idx, obj1 in col_objs_r1:
            if (link1_idx, sub1_idx) not in active_r1_subobjs:
                continue
            ds_21 = _robot2.compute_dist(
                q=_q2,
                obj=obj1,
                h=PARAM_H_DIST_CTRL,
                eps=PARAM_EPS_DIST_CTRL,
                tol=PARAM_TOL_DIST_CTRL,
                no_iter_max=PARAM_NO_ITER_MAX_DIST_CTRL,
                max_dist=params.max_dist_rob
            )
            jac_21 = np.asarray(ds_21.jac_dist_mat, dtype=float)
            if jac_21.ndim == 1 and jac_21.size > 0:
                jac_21 = jac_21.reshape(1, -1)
            if jac_21.size > 0 and jac_21.shape[1] != n2:
                jac_21 = jac_21.reshape(-1, n2)

            if hasattr(ds_21, "_list_info"):
                for k, item in enumerate(ds_21._list_info):
                    key = (
                        link1_idx,
                        sub1_idx,
                        int(item.link_number),
                        int(getattr(item, "link_col_obj_number", getattr(item, "obj_col_number", 0)))
                    )
                    jac_r2_map[key] = jac_21[k:k + 1, :]

    for key, j1_row in jac_r1_map.items():
        d_val = dist_r1_map[key]
        j2_row = jac_r2_map.get(key, np.zeros((1, n2), dtype=float))
        row_12 = np.hstack((j1_row, j2_row))
        rhs = -params.eta_rob * (d_val - params.rob_delta)

        A_list.append(row_12)
        b_list.append(np.array([[rhs]], dtype=float))
        n_rob_constraints += 1
        debug_rows.append(("rob1_rob2", key, d_val, rhs))

    time_rob_dist = time.perf_counter() - t0_rob

    A = np.vstack(A_list)
    b = np.vstack(b_list)

    u_nominal_12 = np.vstack((_u_nom1, _u_nom2))
    H = 2.0 * np.eye(total_joints)
    f = -2.0 * u_nominal_12

    if robot2_stationary:
        A_eq = np.hstack((np.zeros((n2, n1)), np.eye(n2)))
        b_eq = np.zeros((n2, 1))
    else:
        A_eq = None
        b_eq = None

    qp_viavel = True
    msg = "OK"
    t0_qp = time.perf_counter()
    try:
        u_12 = ub.Utils.solve_qp(
            np.matrix(H),
            np.matrix(f),
            np.matrix(A),
            np.matrix(b),
            np.matrix(A_eq) if A_eq is not None else None,
            np.matrix(b_eq) if b_eq is not None else None
        )
        time_qp = time.perf_counter() - t0_qp
        if u_12 is None:
            raise RuntimeError("solve_qp retornou None.")
        u_12 = np.asarray(u_12, dtype=float).reshape(-1, 1)
        if np.any(~np.isfinite(u_12)):
            raise RuntimeError("Solução do QP inválida.")
        u1 = u_12[0:n1, :]
        u2 = u_12[n1:total_joints, :]
        if robot2_stationary:
            u2 = np.zeros_like(u2)
    except Exception as exc:
        time_qp = time.perf_counter() - t0_qp
        qp_viavel = False
        msg = f"Fallback estado seguro (QP 12 juntas inviável: {exc})"
        u1 = np.zeros_like(_u_nom1)
        u2 = np.zeros_like(_u_nom2)

    time_apply = time.perf_counter() - t_apply_ini

    return ResultadoFiltroQP(
        u1=u1,
        u2=u2,
        min_self_dist_ctrl=min(min_self_1, min_self_2),
        n_self_constraints=n_self_1 + n_self_2,
        min_obs_dist_ctrl=min(min_obs_1, min_obs_2),
        n_obs_constraints=n_obs_1 + n_obs_2,
        min_rob_dist_ctrl=min_rob_dist_ctrl,
        n_rob_constraints=n_rob_constraints,
        time_self_dist=t_self_1 + t_self_2,
        time_obs_dist_total=t_obs_1 + t_obs_2,
        time_rob_dist=time_rob_dist,
        time_qp=time_qp,
        time_apply_constraints=time_apply,
        qp_viavel=qp_viavel,
        mensagem_status=msg,
        detalhes_restricoes=debug_rows
    )

