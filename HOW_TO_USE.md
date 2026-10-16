# Como Usar o Projeto (HOW TO USE)

Guia passo a passo para configurar o ambiente, preparar os dados e executar todos os componentes do projeto (Análise Exploratória, Pipeline de Pré-processamento, Treinamento do Modelo XGBoost e Recommendations Engine CLI).

---

## 1. Pré-requisitos

* **Python**: versão 3.10 ou superior recomendada.
* **Git**: para clonar o repositório.

---

## 2. Instalação e Configuração do Ambiente

### Passo 1: Clonar o repositório
```bash
git clone https://github.com/SEU_USUARIO/modelo_preditivo_klike.git
cd modelo_preditivo_klike
```

### Passo 2: Criar e ativar o ambiente virtual (venv)
* **No Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **No Windows:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```

### Passo 3: Instalar as dependências
Com o ambiente virtual ativado, instale todos os pacotes necessários:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Configuração da Base de Dados

O projeto espera que o dataset bruto esteja localizado na pasta `base_de_dados/`. Se você acabou de clonar o projeto ou o arquivo não estiver presente:

1. Crie o diretório `base_de_dados` na raiz do projeto (se ainda não existir):
   ```bash
   mkdir -p base_de_dados
   ```
2. Adicione o arquivo `klike_challenge_dataset.csv` dentro desta pasta:
   ```
   modelo_preditivo_klike/
   ├── base_de_dados/
   │   └── klike_challenge_dataset.csv
   ├── inputs_exemplo/
   │   ├── campanha_exemplo_linkedin.json
   │   └── campanha_exemplo_tiktok.json
   ├── pipeline_preparacao.py
   ├── modelo_xgboost.py
   ├── recomendador.py
   └── ...
   ```

---

## 4. Executando os Componentes

### 📊 Parte 1: Análise Exploratória (EDA)
O notebook [`analise_exploratoria.ipynb`](analise_exploratoria.ipynb) contém a investigação completa dos dados, distribuições, outliers, correlações e os 3 gráficos para apresentação de marketing.

Para visualizar no Jupyter Notebook ou JupyterLab:
```bash
jupyter notebook analise_exploratoria.ipynb
```
*(Você também pode abrir e rodar diretamente pela extensão Jupyter do VS Code ou Cursor)*.

---

### ⚙️ Parte 2.1: Pipeline de Preparação e Feature Engineering
O arquivo [`pipeline_preparacao.py`](pipeline_preparacao.py) calcula a nova feature de taxa de retenção (`retention_rate`), realiza imputação de nulos, transformações (`log1p`, `StandardScaler`, `RobustScaler`, `MinMaxScaler`), codificação categórica (`OneHotEncoder`, `OrdinalEncoder`) e divide os dados em treino (80%) e teste (20%).

Para rodar e inspecionar a preparação:
```bash
python pipeline_preparacao.py
```
**Saída esperada:** Confirmação da matriz preparada com 400 linhas de treino e 100 linhas de teste em 32 features finais tratadas.

---

### 🚀 Parte 2.2: Treinamento e Avaliação do Modelo XGBoost
O arquivo [`modelo_xgboost.py`](modelo_xgboost.py) treina o estimador `XGBRegressor`, avalia o desempenho em amostras nunca vistas de teste e exibe o ranking de importância das variáveis.

Para executar o treinamento e teste:
```bash
python modelo_xgboost.py
```
**Saída esperada:**
* Métricas no teste: $R^2 \approx 0.81$, $\text{RMSE} \approx 6.57$, $\text{MAE} \approx 5.25$.
* Tabela comparativa com valores reais vs. previstos das 10 primeiras amostras de teste.
* Top 5 variáveis mais determinantes do modelo (`has_hook`, `has_face`, `ctr`, etc.).

---

### 🎯 Parte 3: Recommendations Engine (CLI)
A ferramenta [`recomendador.py`](recomendador.py) recebe um arquivo `.json` contendo os dados de uma campanha publicitária e devolve um relatório em texto simples com recomendações de melhoria quantificadas, contextuais e priorizadas por impacto.

> [!IMPORTANT]
> A ferramenta aceita **estritamente arquivos com extensão `.json`**. Qualquer outro formato é rejeitado.

#### Exemplos de Execução:

1. **Testar com o exemplo do LinkedIn (formato vertical inadequado e texto excessivo):**
   ```bash
   python recomendador.py inputs_exemplo/campanha_exemplo_linkedin.json
   ```

2. **Testar com o exemplo do TikTok (formato quadrado e sem hook):**
   ```bash
   python recomendador.py inputs_exemplo/campanha_exemplo_tiktok.json
   ```

3. **Exibir mais recomendações (usando a flag `--top` ou `-k`):**
   ```bash
   python recomendador.py inputs_exemplo/campanha_exemplo_tiktok.json --top 4
   ```

#### Como testar com uma campanha nova:
Crie um arquivo `.json` (ex: `minha_campanha.json`) com os atributos do vídeo que deseja avaliar:
```json
{
  "platform": "TikTok",
  "category": "SaaS",
  "objective": "traffic",
  "format": "quadrado",
  "video_duration_s": 25.0,
  "avg_watch_time_s": 6.0,
  "has_hook": false,
  "has_face": false,
  "has_cta": false,
  "has_subtitle": false,
  "text_density": "high",
  "music_voice_ratio": 0.4
}
```
E execute:
```bash
python recomendador.py minha_campanha.json
```

---

## 5. Estrutura do Repositório

```
.
├── base_de_dados/
│   └── klike_challenge_dataset.csv     # Base de dados tabular (500 linhas x 26 colunas)
├── inputs_exemplo/
│   ├── campanha_exemplo_linkedin.json  # Exemplo de entrada .json para teste
│   └── campanha_exemplo_tiktok.json    # Exemplo de entrada .json para teste
├── analise_exploratoria.ipynb          # Notebook da Parte 1 (EDA e visualizações)
├── pipeline_preparacao.py              # Pipeline modular de pré-processamento e split
├── modelo_xgboost.py                   # Script de treinamento, avaliação e métricas
├── recomendador.py                     # Motor de recomendações via CLI (.json)
├── requirements.txt                    # Dependências e bibliotecas Python
├── README.md                           # Respostas detalhadas de todas as partes do desafio
└── HOW_TO_USE.md                       # Este guia de execução passo a passo
```

---

## 6. Solução de Problemas Comuns

* **`FileNotFoundError: Arquivo base_de_dados/klike_challenge_dataset.csv não encontrado`**:
  Certifique-se de que a pasta `base_de_dados/` existe e que o arquivo `.csv` foi colocado nela com o nome exato.
* **`Formato de arquivo inválido: ... A ferramenta aceita estritamente arquivos com extensão .json`**:
  O `recomendador.py` só aceita arquivos `.json`. Verifique a extensão informada no comando.
* **`ModuleNotFoundError: No module named 'xgboost'`**:
  Certifique-se de ativar o ambiente virtual (`source .venv/bin/activate` ou `.venv\Scripts\activate`) antes de executar os comandos.
