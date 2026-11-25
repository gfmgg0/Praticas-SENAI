# Mestrado — Controle Cinemático de Robôs (CCR)

Projeto de pesquisa e experimentos de controle cinemático para manipuladores robóticos (Universal Robots UR3 / UR3e) utilizando o simulador e biblioteca [UaiBot](https://github.com/viniciusam/uaibot), comunicação em tempo real via **RTDE** (`ur_rtde`) e otimização por restrições via **CVXOPT**.

---

## 📁 Estrutura do Repositório

```text
├── Teste_CCR_V6.ipynb   # Notebook principal com simulação, controle diferencial e integração RTDE
├── requirements.txt     # Dependências do projeto Python 3.11
├── .gitignore           # Arquivos e diretórios ignorados pelo Git
└── README.md            # Documentação do projeto
```

---

## 🚀 Como Executar

### 1. Pré-requisitos
- Python 3.11 instalado.

### 2. Ativar o Ambiente Virtual
O ambiente virtual já foi configurado em `.venv`:

```bash
# Ativar o ambiente virtual (Linux/macOS)
source .venv/bin/activate
```

Caso precise recriar o ambiente:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name mestrado-ccr --display-name "Python 3.11 (Mestrado CCR)"
```

### 3. Abrir o Jupyter Notebook
Abra o arquivo [`Teste_CCR_V6.ipynb`](Teste_CCR_V6.ipynb) no VS Code / Jupyter Notebook e selecione o kernel **Python 3.11 (Mestrado CCR)** ou o interpretador em `.venv/bin/python`.

---

## 📦 Principais Dependências
- `uaibot` (1.2.6): Modelagem e simulação de robôs manipuladores.
- `ur_rtde` (1.6.1): Interface de comunicação em tempo real com o robô UR (Universal Robots).
- `cvxopt` (1.3.2): Solução de problemas de otimização convexa (QP) para cálculo das ações de controle com restrições.
- `numpy`, `scipy`, `matplotlib`: Computação numérica, álgebra linear e visualização gráfica.
- `ipykernel`: Execução interativa via Jupyter.
