# Práticas Didáticas de Robótica no UAIBot (UR3e + QP/CBF)

Implementação modular de três práticas didáticas progressivas no simulador **UAIBot** com manipuladores **Universal Robots UR3e**, controle cinemático diferencial por inversa amortecida (DLS) e filtragem de segurança por Programação Quadrática e *Control Barrier Functions* (**QP/CBF**).

---

## 📁 Estrutura de Arquivos

```text
praticas_uaibot/
│
├── config/
│   ├── __init__.py
│   └── parametros.py        # Parâmetros geométricos, cinemáticos, ganhos QP/CBF e configs das práticas
│
├── core/
│   ├── __init__.py
│   ├── bancada.py           # Bancada padrão modular, referenciais, regiões e inicialização/reset
│   ├── manipulador.py       # Interface do UR3e desacoplada da visualização (preparada p/ RTDE)
│   ├── garra.py             # Interfaces de GarraSimulada (RG6) e CanetaFerramenta com TCP configurável
│   ├── objetos.py           # ObjetoManipulavel, ObstaculoConfiguravel e FolhaPapel (PointCloud)
│   ├── controle.py          # Controlador cinemático diferencial DLS e função de tarefa V(r)
│   ├── seguranca.py         # Distâncias diferenciáveis/euclidianas e filtros QP de 6 e 12 juntas
│   └── trajetorias.py       # Gerador de trajetórias cartesianas (linha, quadrado, círculo, palavras)
│
├── pratica_1/
│   ├── __init__.py
│   └── main.py              # Prática 1: Pick-and-Place com prevenção de colisões e comparativo
│
├── pratica_2/
│   ├── __init__.py
│   └── main.py              # Prática 2: Rastreamento de trajetória com escrita robótica sobre folha
│
├── pratica_3/
│   ├── __init__.py
│   └── main.py              # Prática 3: Cooperação entre 2 bancadas com handover sincronizado
│
└── README.md                # Instruções de uso e parametrização para alunos
```

---

## 🚀 Como Executar as Simulações

A partir da raiz do repositório (`/home/gabriel/Práticas_SENAI`), com o ambiente virtual `.venv` ativo:

```bash
source .venv/bin/activate
```

### 1. Prática 1 — Manipulação Simples com Filtro de Segurança
Executa o ciclo completo de coleta (*pick*), transporte com desvio de obstáculo e entrega (*place*), comparando o comportamento **com filtro QP/CBF** e **sem filtro (nominal)**:

```bash
# Executar comparativo completo (Com Filtro vs Sem Filtro)
.venv/bin/python -m praticas_uaibot.pratica_1.main --modo comparar

# Executar apenas com filtro QP/CBF alterando a posição do obstáculo (x, y, z relativos à bancada)
.venv/bin/python -m praticas_uaibot.pratica_1.main --modo com_filtro --obs-xyz 0.335 0.00 0.075
```

Arquivos HTML interativos gerados em `outputs_simulacao/`:
- `outputs_simulacao/pratica_1_com_filtro.html`
- `outputs_simulacao/pratica_1_sem_filtro.html`

---

### 2. Prática 2 — Rastreamento de Trajetória com Escrita Robótica
Substitui a garra pela ferramenta de escrita (`CanetaFerramenta`), posiciona uma folha sobre a bancada e executa o rastreamento cartesiano registrando o traço com `ub.PointCloud`:

```bash
# Escrever uma palavra curta sobre a folha (padrão: "UAI")
.venv/bin/python -m praticas_uaibot.pratica_2.main --tipo palavra --palavra UAI --escala 0.06

# Desenhar figuras geométricas simples
.venv/bin/python -m praticas_uaibot.pratica_2.main --tipo linha
.venv/bin/python -m praticas_uaibot.pratica_2.main --tipo quadrado
.venv/bin/python -m praticas_uaibot.pratica_2.main --tipo circulo
.venv/bin/python -m praticas_uaibot.pratica_2.main --tipo combo
```

Arquivo HTML interativo gerado em `outputs_simulacao/`:
- `outputs_simulacao/pratica_2_<tipo>.html`

---

### 3. Prática 3 — Colaboração entre Duas Bancadas
Instancia duas bancadas padrão (`Bancada A` e `Bancada B`), dois manipuladores UR3e e executa a máquina de estados cooperativa com transferência sincronizada de objeto (*handover* perpendicular direto entre garras ou via estação intermediária) sob proteção do filtro QP acoplado de 12 juntas:

```bash
# Executar transferência direta entre as garras (padrão: DY = -0.66 m)
.venv/bin/python -m praticas_uaibot.pratica_3.main --dy-bancada -0.66

# Executar com estação intermediária visualizada na zona compartilhada
.venv/bin/python -m praticas_uaibot.pratica_3.main --dy-bancada -0.66 --estacao-intermediaria
```

Arquivo HTML interativo gerado em `outputs_simulacao/`:
- `outputs_simulacao/pratica_3_colaborativa.html`

---

## ⚙️ Parâmetros que os Alunos Podem Modificar

Todos os parâmetros experimentais estão centralizados em [`praticas_uaibot/config/parametros.py`](config/parametros.py):

1. **Bancada Padrão (`ConfigBancada`)**:
   - `origem_bancada`, `yaw_bancada`: Posição e orientação da bancada no mundo.
   - `pos_origem_rel`, `pos_destino_rel`, `pos_home_rel`: Coordenadas relativas à superfície da bancada.
   - `TCP_GARRA_OFFSET` `(0.0, 0.0, 0.20)` e `TCP_CANETA_OFFSET` `(0.0, 0.0, 0.24)`.
2. **Prática 1 (`ConfigPratica1`)**:
   - `pos_obstaculo_rel`, `dim_obstaculo`: Posição e dimensões do obstáculo no caminho.
   - `eta_obs`, `obs_delta`, `obs_activation`: Ganho da CBF, margem mínima de segurança e raio de ativação.
   - `usar_filtro_qp`: Habilita/desabilita o filtro QP para comparação.
3. **Prática 2 (`ConfigPratica2`)**:
   - `tipo_trajetoria` (`"linha"`, `"quadrado"`, `"circulo"`, `"palavra"`, `"combo"`), `texto_palavra`.
   - `escala_desenho`, `pontos_por_segmento` (resolução), `passos_por_ponto` (velocidade).
   - `altura_escrita_rel`, `altura_retracao_rel`, `inclinacao_caneta_rad`.
4. **Prática 3 (`ConfigPratica3`)**:
   - `dy_bancada_b`: Distância relativa entre as duas bancadas.
   - `pos_transferencia_mundo`: Ponto de encontro na zona compartilhada.
   - `tol_sincronizacao_m`: Tolerância máxima entre TCPs para autorizar a transferência segura do objeto.

