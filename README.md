# Desafio Técnico — Millai / Klike (Modelo Preditivo de Criativos)

> 💡 **Para instruções de instalação, configuração de pastas e execução passo a passo, consulte o [HOW_TO_USE.md](HOW_TO_USE.md).**

---

## Parte 1 — Análise Exploratória de Dados (EDA) - analise_exploratoria.ipynb e pipeline_preparacao.py



### 1. Perfil Geral dos Dados, Distribuições, Valores Faltantes e Outliers

* **Perfil Geral**: 500 registros e 26 colunas de campanhas veiculadas em 2024 no Meta, TikTok e LinkedIn. Target `klike_score` contínuo (média 60.1, desvio padrão 15.6, min 18.0, max 99.1).
* **Valores Faltantes**:
  - `has_subtitle` (46 nulos / 9.2%): Imputação pela moda (`most_frequent`).
  - `video_duration_s` (39 / 7.8%), `music_voice_ratio` (38 / 7.6%), `avg_watch_time_s` (27 / 5.4%), `engagement_rate` (32 / 6.4%): Imputação pela mediana.
  - `cpc` (28 / 5.6%): Imputação pela mediana (ou regra contábil direta: $spend / clicks$).
  - `revenue` (25 / 5.0%): Imputação pela mediana (ou regra contábil direta: $spend \times roas$).
* **Distribuições**:
  - Fortemente assimétricas à direita / cauda longa: `impressions`, `clicks`, `spend`, `conversions`, `revenue`, `roas`.
  - Tratamento: Transformação logarítmica com deslocamento zero (`log1p`), comprimindo ordens de grandeza e reduzindo a distorção das caudas.
* **Outliers**:
  - Concentrados em métricas de custo e financeiras (ex: 3 CPCs anômalos na Meta entre $20 e $26; investimentos atípicos no LinkedIn).
  - Tratamento: Aplicação de `RobustScaler` (baseado em mediana e IQR) para variáveis contínuas com outliers e `log1p` nas métricas de cauda longa.

---

### 2. Atributos do Criativo Mais Correlacionados com Performance

* **`has_hook` (Gancho nos primeiros 3s)**: Maior correlação com o score ($\rho = +0.55$) e maior alavanca de CTR (CTR médio salta de 4.96% para 8.59%).
* **`has_face` (Presença de Rosto Humano)**: Forte correlação com o score ($\rho = +0.39$) e retenção.
* **`has_cta` (Call to Action)** e **`has_subtitle` (Legendas)**: Correlações moderadas positivas ($\rho = +0.24$ e $\rho = +0.23$).
* **`video_duration_s` e `music_voice_ratio`**: Correlações próximas de zero ($\rho = -0.04$ e $-0.08$), provando que a minutagem bruta não define o sucesso, e sim a dinâmica dos primeiros segundos (`hook` + `face`).

---

### 3. Padrões por Plataforma: O que funciona no TikTok funciona no LinkedIn?

**Não, os padrões são opostos**:

* **Formato**: TikTok é 100% vertical (86.4%) ou quadrado (13.6%), com 0% horizontal. LinkedIn é 53% horizontal e 34% quadrado.
* **Duração e Retenção**: TikTok tem vídeos ágeis (média 14.3s duração e 7.4s retenção). LinkedIn tem vídeos longos (27.3s duração e 15.8s retenção).
* **Custo e Retorno**: CPC mediano do LinkedIn ($2.56) é 5x mais caro que o do TikTok ($0.52). O TikTok entrega ROAS médio de **3.59x** vs. **1.13x** no LinkedIn.
* **Adequação por Vertical**: TikTok domina em SaaS (ROAS 6.39x) e E-commerce (3.98x). O LinkedIn é eficiente apenas para Lead Gen B2B (3.89x) e opera em prejuízo no E-commerce (ROAS 0.82x).

---

### 4. As 3 Visualizações para o Time de Marketing

1. **A Matriz do Criativo Campeão (Hook $\times$ Face)**:
   - Cruza a presença de gancho e rosto com o score e o CTR. Sem Hook e Sem Face: nota média 45.0 e CTR 4.96%; Com Hook e Com Face: nota média 72.9 e CTR 8.59% (+62% no score e dobro de tráfego).
2. **Onde o Dinheiro é Mais Aproveitado (Eficiência de Capital)**:
   - Painel com ROAS por plataforma comparado à linha de Breakeven ($1.00x) e Heatmap de ROAS por Vertical $\times$ Canal (revela que SaaS rende 6.39x no TikTok e E-commerce no LinkedIn dá prejuízo).
3. **Playbook de Produção / Lift Analysis**:
   - Gráfico de ganho absoluto (*Lift*): Hook (+17.1 pts), Face (+12.1 pts), CTA (+8.1 pts) e Legenda (+7.1 pts), funcionando como checklist obrigatório de aprovação de criativos.

---

## Parte 2 — Modelagem Preditiva - modelo_xgboost.py

### 1. Feature Engineering

* **Features Escolhidas**:
  - Criativo (`has_hook`, `has_face`, `has_cta`, `has_subtitle`, `format`, `text_density`, `video_duration_s`, `music_voice_ratio`): atributos controláveis na produção do vídeo.
  - Performance e Custo (`ctr`, `engagement_rate`, `cpc`, `spend`, `conversions`, `revenue`, `roas`): tração do público e eficiência de mídia.
  - Contexto (`platform`, `category`, `objective`, `target_audience_age`, `is_retargeting`).
* **Features Criadas (Nova Feature Engineering)**:
  - `retention_rate` ($\frac{\text{avg\_watch\_time\_s}}{\text{video\_duration\_s}}$): percentual do vídeo assistido pelo usuário (limitado em $[0, 1]$). Possui correlação de **+0.401** com o `klike_score` e aumentou o $R^2$ do modelo de 0.799 para 0.809.
* **Features Descartadas**:
  - `campaign_id`: ID único (cardinalidade 100%), geraria overfitting e ruído sem generalização.
  - `date`: alta dispersão temporal em base de 500 registros sem ganho preditivo direto.
* **Engenharia e Tratamento**:
  - Transformação `log1p` para colunas assimétricas (`impressions`, `clicks`, `spend`, `conversions`, `revenue`, `roas`).
  - Escalonamento `RobustScaler` para contínuas com outliers e `MinMaxScaler` para taxas limitadas [0, 1] (`ctr`, `engagement_rate`, `music_voice_ratio`, `retention_rate`).
  - `OneHotEncoder(drop='first')` para nominais e `OrdinalEncoder` para ordinais (`text_density` e faixa etária).

---

### 2. Modelo de Regressão Escolhido

* **Modelo**: **XGBoost Regressor (`XGBRegressor`)** com `n_estimators=100`, `learning_rate=0.08`, `max_depth=4`.
* **Por que essa escolha?**:
  1. Captura interações não-lineares complexas entre criativo e plataforma (ex: gancho em vídeo vertical no TikTok) sem exigir especificação manual de termos cruzados.
  2. Robusto a retornos decrescentes e saturação de escala em métricas financeiras.
  3. Regularização nativa ($L_1$ e $L_2$) que previne overfitting em amostras pequenas (400 treinos / 100 testes).
  4. Superou modelos lineares no teste ($R^2 \approx 0.81$ vs. $R^2 \approx 0.74$ do Ridge).

---

### 3. Avaliação no Conjunto de Teste (100 Amostras Não Vistas)

* **$R^2 = 0.8092$ (~81%)**: O modelo explica 81% da variabilidade da qualidade dos criativos.
* **MAE = 5.25 pontos**: Em média, a previsão erra por apenas $\approx 5.2$ pontos na escala de 0 a 100.
* **RMSE = 6.57 pontos**: Valor próximo ao MAE confirma estabilidade e ausência de erros discrepantes graves.
* **Interpretação**: Margem de erro de $\pm 5$ pontos garante alta confiabilidade para classificar criativos em faixas de potencial antes de investir verba de mídia.

---

### 4. Feature Importance e Sentido de Negócio

* **Top 5 Variáveis Mais Importantes**:
  1. `has_hook`: **45.9%** de importância relativa
  2. `has_face`: **9.5%**
  3. `ctr`: **8.2%**
  4. `text_density`: **4.2%**
  5. `has_cta`: **3.8%**

* **Faz sentido do ponto de vista de negócio?**:
  - **Sim, total sentido**:
    - `has_hook` lidera isolado porque nas redes de consumo rápido o usuário decide nos primeiros 2 a 3 segundos se assiste ou pula o anúncio. Sem gancho, o criativo é abandonado.
    - `has_face` humaniza o criativo e gera identificação imediata, superando anúncios estáticos ou genéricos.
    - `ctr` comprova que a mensagem inicial converteu atenção em clique.

---

## Parte 3 — Recommendations Engine (CLI) - recomendador.py

O motor de recomendações está implementado no script CLI [`recomendador.py`](recomendador.py). Ele recebe estritamente um arquivo `.json` com os dados da campanha publicitária, avalia os atributos contextuais e retorna sugestões acionáveis ordenadas por impacto estimado.

### 1. Pilares do Sistema
1. **Acionável e Quantificado**: cada sugestão apresenta a ação criativa objetiva e o ganho estimado no `klike_score` e em métricas de negócio observadas no EDA.
2. **Contextualidade por Plataforma**:
   - **TikTok**: exige formato vertical (9:16), duração curta (<20s) e densidade de texto moderada. Vídeos longos (>45s) despencam o score de 60.1 para 37.2 pts.
   - **LinkedIn**: favorece formato horizontal (16:9) ou quadrado (1:1), penalizando fortemente vídeos verticais (média 45.5 vs 58.9 horizontal). Requer presença humana B2B e texto explicativo moderado.
   - **Meta**: favorece formatos imersivos mobile (1:1 e 9:16), presença humana e ganchos dinâmicos.
3. **Priorização por Impacto**: as recomendações combinam o ganho marginal contrafactual do modelo XGBoost com os benchmarks históricos do dataset, ordenando as alavancas mais críticas no topo.
4. **Garantia de Cobertura**: mesmo campanhas de alto desempenho recebem pelo menos 3 sugestões de otimização contínua (testes A/B de hook, refinamento de CTA, retargeting).

### 2. Como Executar a CLI

A CLI exige obrigatoriamente como argumento o caminho para um arquivo `.json`:

```bash
# Analisar exemplo no LinkedIn (formato vertical inadequado, texto denso, sem hook)
python recomendador.py inputs_exemplo/campanha_exemplo_linkedin.json

# Analisar exemplo no TikTok (formato quadrado, sem presença humana, sem hook)
python recomendador.py inputs_exemplo/campanha_exemplo_tiktok.json
```

Se for fornecido qualquer arquivo que não termine em `.json`, a ferramenta rejeita a entrada com mensagem de erro explícita.


## Parte 4 — Visão de Produto

### 4.1 Features Adicionais com Vídeo Original


#### 1. Uso de IA / Mídia Sintética vs. Humano Real
* **Por que importa**: Criativos gerados por IA (voz sintetizada ou avatares) têm desempenhos opostos dependendo do público. No TikTok e Instagram, o público jovem (18-24 anos) valoriza autenticidade e tende a rejeitar anúncios que parecem "robóticos", derrubando a retenção. No LinkedIn, vídeos explicativos com avatares polidos têm boa aceitação.


#### 2. Intensidade e Energia da Voz (Entusiasmo)
* **Por que importa**: A energia vocal nos primeiros 3 segundos é o motor do gancho (*hook*). Uma fala monótona ou desanimada faz o usuário pular o anúncio imediatamente. Da mesma forma, uma entonação firme e confiante no final aumenta a taxa de conversão da Call to Action (CTA).


---

### 4.2 Colocando em produção

Para colocar o Recommendations Engine em produção, eu empacotaria o pipeline e o modelo em um container Docker leve, orquestrado em um cluster Kubernetes (Amazon EKS) na AWS, garantindo alta escalabilidade através do HPA (Horizontal Pod Autoscaler) e Karpenter para escalar pods e instâncias EC2 automaticamente sob demanda com inferência em memória (<20ms). A integração com o produto real ocorreria por meio da API em FastAPI que criaríamos, permitindo que a aplicação consuma os endpoints diretamente durante o upload ou edição do anúncio para retornar em tempo real o score preditivo e as recomendações priorizadas no painel do usuário, além de receber os resultados de performance pós-veiculação para alimentar um ciclo contínuo de monitoramento e retreinamento do modelo.

### 4.3 Melhorias


1. **Modelos mais complexos vs. Explicabilidade**:
   - Eu testaria modelos mais pesados (como redes neurais para dados tabulares ou *ensembles* combinando vários algoritmos) pra ver se daria pra espremer um $R^2$ ainda maior. Abri mão disso nesta entrega porque preferi garantir a explicabilidade do XGBoost, que deixa bem claro para o time de marketing e negócio o porquê de cada nota e recomendação.

2. **Mais dados e acompanhamento no tempo**:
   - A base de 500 linhas é pequena para cravar certezas absolutas. Com mais tempo, eu buscaria coletar muito mais dados e acompanhar as campanhas ao longo das semanas. Isso ajudaria a entender a fadiga do criativo (o anúncio perdendo força com o tempo) e a fazer validações temporais mais fiéis ao dia a dia real.

3. **Uma CLI mais interativa e flexível**:
   - A ferramenta que montei foi direto ao ponto para resolver o desafio recebendo o `.json`. Com mais tempo, transformaria ela numa experiência interativa no terminal (estilo passo a passo interativo ou dashboard simples), onde o usuário pudesse brincar com simulações na hora — mudando variáveis do vídeo para ver o impacto no score antes mesmo de subir a campanha.


