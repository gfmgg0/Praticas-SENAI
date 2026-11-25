# Diretrizes do Projeto — Mestrado CCR (Controle Cinemático de Robôs)

Este repositório contém os experimentos de mestrado em controle cinemático com robôs manipuladores UR3e, simulados via UaiBot e integrados em tempo real via RTDE com restrições por otimização quadrática (CVXOPT QP).

---

## 🐍 Ambiente de Execução

- **Python Virtualenv**: `.venv\Scripts\python.exe` (Windows)
- **Kernel Jupyter**: `Python 3.11 (Mestrado CCR)`
- **Dependências Chave**: `uaibot`, `ur_rtde`, `cvxopt`, `numpy`, `scipy`, `matplotlib`, `ipykernel`

---

## 📓 Notebook Principal: `Teste_CCR_V7.ipynb`

O notebook está estruturado nas seguintes células:

| Célula | Nome / Responsabilidade |
|---|---|
| **0** | Configuração Geral, Ganhos, Tolerâncias, Parâmetros CBF, `SHOW_COLLISION_PRIMITIVES`, `ROBOT_2_STATIONARY` e Constantes |
| **1** | RTDE (Conexão, desconexão, envio de velocidade `speedJ` e posição) |
| **2** | Funções Auxiliares (cálculo de distâncias euclidianas, colisão, saturação) |
| **3** | Controle Diferencial (`compute_u_nominal`, DLS Jacobian, Lyapunov) |
| **4** | Filtro de Restrições QP (CVXOPT, 12 juntas, autocolisão e obstáculos) |
| **5** | Cenário e Gêmeo Digital (2 Bancadas + 2 UR3e + Obstáculos) |
| **6** | Loop Principal de Controle em Tempo Real |
| **7** | Finalização e pós-processamento |
| **8** | Métricas e Tempos Computacionais |
| **9** | Gráficos de Convergência ($V(r)$, distâncias mínimas, velocidades) |
| **10-15** | Células adicionais de análise, exportação de simulação e limites |

---

## ⚡ Protocolo de Alteração em Tempo Real

Para realizar alterações diretamente no notebook `.ipynb` em tempo real sem corromper o arquivo JSON:

1. **Ferramenta CLI**: `scripts/nb_manager.py`
   - Listar células: `.venv\Scripts\python.exe scripts/nb_manager.py list Teste_CCR_V7.ipynb`
   - Ler célula: `.venv\Scripts\python.exe scripts/nb_manager.py get <idx> Teste_CCR_V7.ipynb`
   - Atualizar célula: `.venv\Scripts\python.exe scripts/nb_manager.py set <idx> <arquivo_ou_codigo> Teste_CCR_V7.ipynb`
   - Substituir trecho: `.venv\Scripts\python.exe scripts/nb_manager.py replace <idx> <busca> <substituicao> Teste_CCR_V7.ipynb`
2. **Backups automáticos**: Toda modificação gera automaticamente uma cópia de segurança em `Teste_CCR_V7.ipynb.bak`.
3. **Sincronização imediata**: O editor/VS Code atualiza o notebook aberto no disco imediatamente.
