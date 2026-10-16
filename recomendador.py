"""
Módulo e CLI: Sistema de Recomendações de Criativos (KlikeRec Engine)
Desafio Millai / Klike — Parte 3: Recommendations Engine

Este script analisa uma campanha publicitária (uma linha do dataset ou JSON de entrada)
e gera recomendações de melhoria que são:
1. Acionáveis e Quantificadas: estimam o ganho em klike_score e métricas de negócio (CTR, retenção);
2. Contextuais: consideram especificidades de cada plataforma (TikTok, LinkedIn, Meta) e categoria;
3. Priorizadas: ordenadas por ordem decrescente de impacto potencial na performance;
4. Executável via CLI: aceita estritamente arquivos com extensão .json como entrada, retornando texto simples.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline_preparacao import X_train, X_train_prep, preprocessor, y_train

DATASET_PATH = 'base_de_dados/klike_challenge_dataset.csv'

# Cache do modelo para evitar retreinamento em múltiplas chamadas
_MODELO_CACHE: Optional[xgb.XGBRegressor] = None


def obter_modelo_treinado() -> xgb.XGBRegressor:
    """
    Retorna o modelo XGBoost treinado no conjunto de treinamento.
    Utiliza cache em memória para resposta instantânea.
    """
    global _MODELO_CACHE
    if _MODELO_CACHE is None:
        modelo = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            objective='reg:squarederror',
        )
        modelo.fit(X_train_prep, y_train)
        _MODELO_CACHE = modelo
    return _MODELO_CACHE


def prever_score(campanha_dict: Dict[str, Any], modelo: xgb.XGBRegressor) -> float:
    """
    Gera a predição pontual do klike_score utilizando o pipeline de preparação e o XGBoost.
    """
    df_input = pd.DataFrame([campanha_dict])

    # Garantir cálculo da retenção se duração e watch time existirem
    if 'retention_rate' not in df_input.columns or pd.isna(df_input['retention_rate'].iloc[0]):
        dur = df_input.get('video_duration_s', pd.Series([np.nan])).iloc[0]
        wt = df_input.get('avg_watch_time_s', pd.Series([np.nan])).iloc[0]
        if pd.notna(dur) and pd.notna(wt) and dur > 0:
            df_input['retention_rate'] = min(max(wt / dur, 0.0), 1.0)
        else:
            df_input['retention_rate'] = np.nan

    # Alinhar colunas com as features esperadas pelo pipeline
    for col in X_train.columns:
        if col not in df_input.columns:
            df_input[col] = np.nan

    X_prep = preprocessor.transform(df_input[X_train.columns])
    pred = float(modelo.predict(X_prep)[0])
    return pred


def gerar_recomendacoes(
    campanha: Dict[str, Any],
    modelo: Optional[xgb.XGBRegressor] = None,
    top_n: int = 3,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Analisa os atributos de uma campanha e gera recomendações quantificadas, contextuais
    e priorizadas por impacto estimado no klike_score e nas métricas de performance.
    """
    if modelo is None:
        modelo = obter_modelo_treinado()

    # Predição baseline da campanha
    score_baseline = prever_score(campanha, modelo)

    plat = str(campanha.get('platform', 'Meta'))
    formato = str(campanha.get('format', 'vertical'))
    has_hook = bool(campanha.get('has_hook', False))
    has_face = bool(campanha.get('has_face', False))
    has_cta = bool(campanha.get('has_cta', False))
    has_sub = bool(campanha.get('has_subtitle', False))
    text_density = str(campanha.get('text_density', 'medium'))
    duracao = float(campanha.get('video_duration_s', 20.0)) if pd.notna(campanha.get('video_duration_s')) else 20.0
    watch_time = float(campanha.get('avg_watch_time_s', 10.0)) if pd.notna(campanha.get('avg_watch_time_s')) else 10.0
    retencao = float(campanha.get('retention_rate', watch_time / max(duracao, 1.0)))
    music_ratio = float(campanha.get('music_voice_ratio', 0.5)) if pd.notna(campanha.get('music_voice_ratio')) else 0.5
    is_retargeting = bool(campanha.get('is_retargeting', False))

    recs: List[Dict[str, Any]] = []

    # Helper para simulação contrafactual no modelo XGBoost
    def simular_ganho(alteracoes: Dict[str, Any]) -> float:
        campanha_mod = dict(campanha)
        campanha_mod.update(alteracoes)
        score_mod = prever_score(campanha_mod, modelo)
        return max(score_mod - score_baseline, 0.5)

    # 1. GANCHO (HOOK) INICIAL
    if not has_hook:
        delta_model = simular_ganho({'has_hook': True})
        bench_score = 17.3 if plat == 'TikTok' else (16.0 if plat == 'LinkedIn' else 17.9)
        pct_ctr = '+40.0%' if plat == 'TikTok' else ('+52.4%' if plat == 'LinkedIn' else '+34.3%')
        recs.append({
            'area': 'Criativo — Hook Inicial',
            'acao': 'Adicionar um gancho visual e verbal impactante nos primeiros 3 segundos de vídeo.',
            'contexto': (
                f'Nas redes de consumo rápido ({plat}), o usuário decide em até 3 segundos se assiste ou pula. '
                f'Sem gancho inicial, o criativo sofre abandono imediato. '
                f'No dataset, anúncios no {plat} com gancho alcançam um salto médio de {pct_ctr} no CTR.'
            ),
            'impacto_quantificado': f'{pct_ctr} no CTR | +{bench_score:.1f} pts de benchmark histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(bench_score, delta_model),
        })

    # 2. FORMATO VS. PLATAFORMA (CONTEXTUALIDADE DE REDE)
    if plat == 'TikTok' and formato != 'vertical':
        delta_model = simular_ganho({'format': 'vertical'})
        recs.append({
            'area': 'Formato & Enquadramento',
            'acao': 'Migrar o formato do vídeo para vertical full-screen (9:16).',
            'contexto': (
                'O TikTok é uma plataforma 100% nativa em tela vertical (9:16). '
                'Formatos quadrados (1:1) ou horizontais (16:9) geram barras pretas, quebram a imersão e '
                'têm score médio de apenas 53.8 contra 62.9 dos vídeos verticais.'
            ),
            'impacto_quantificado': f'+9.1 pts de benchmark histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(9.1, delta_model),
        })
    elif plat == 'LinkedIn' and formato == 'vertical':
        delta_model = simular_ganho({'format': 'horizontal'})
        recs.append({
            'area': 'Formato & Enquadramento',
            'acao': 'Alterar o formato de vertical (9:16) para horizontal (16:9) ou quadrado (1:1).',
            'contexto': (
                'O feed do LinkedIn é predominantemente consumido em ambiente web/desktop. '
                'Vídeos verticais sofrem cortes no feed e apresentam o pior score médio de toda a base (45.5 pts), '
                'enquanto vídeos horizontais entregam média de 58.9 pts (+13.4 pontos de diferença).'
            ),
            'impacto_quantificado': f'+13.4 pts no score histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(13.4, delta_model),
        })
    elif plat == 'Meta' and formato == 'horizontal':
        delta_model = simular_ganho({'format': 'quadrado'})
        recs.append({
            'area': 'Formato & Enquadramento',
            'acao': 'Adaptar formato horizontal para quadrado (1:1 no feed) ou vertical (9:16 nos Reels/Stories).',
            'contexto': (
                'No ecossistema Meta mobile (Instagram e Facebook), criativos quadrados (1:1) e verticais '
                'ocupam até 78% mais área de tela móvel que vídeos horizontais, favorecendo retenção e cliques.'
            ),
            'impacto_quantificado': f'+3.5 pts de klike_score médio (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(3.5, delta_model),
        })

    # 3. PRESENÇA HUMANA (HAS_FACE)
    if not has_face:
        delta_model = simular_ganho({'has_face': True})
        bench_score = 13.1 if plat == 'LinkedIn' else (12.9 if plat == 'Meta' else 10.7)
        recs.append({
            'area': 'Criativo — Conexão Humana',
            'acao': 'Inserir uma pessoa real (especialista, criador, cliente ou apresentador) em destaque.',
            'contexto': (
                f'A presença humana gera conexão instantânea e sensação de autenticidade (UGC). '
                f'No {plat}, anúncios com presença humana aumentam o CTR em +22.8% e superam criativos puramente gráficos '
                f'por uma margem histórica de +{bench_score:.1f} pontos de klike_score.'
            ),
            'impacto_quantificado': f'+22.8% no CTR médio | +{bench_score:.1f} pts no benchmark histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(bench_score, delta_model),
        })

    # 4. DENSIDADE DE TEXTO (TEXT DENSITY)
    if text_density == 'high':
        delta_model = simular_ganho({'text_density': 'medium'})
        bench_score = 15.0 if plat == 'LinkedIn' else (18.3 if plat == 'Meta' else 11.6)
        recs.append({
            'area': 'Copywriting & Design Visual',
            'acao': 'Reduzir a densidade de texto na tela de alta (high) para moderada (medium).',
            'contexto': (
                f'Textos excessivos competem com o vídeo e aumentam a carga cognitiva do usuário, provocando rejeição. '
                f'No {plat}, anúncios com densidade média superam anúncios com densidade alta por uma diferença '
                f'média de +{bench_score:.1f} pontos de score.'
            ),
            'impacto_quantificado': f'+{bench_score:.1f} pts no score médio histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(bench_score, delta_model),
        })
    elif text_density == 'low' and plat == 'LinkedIn':
        delta_model = simular_ganho({'text_density': 'medium'})
        recs.append({
            'area': 'Copywriting & Design Visual',
            'acao': 'Adicionar pontos-chave ou dados em texto moderado (medium) sobre o vídeo.',
            'contexto': (
                'No LinkedIn, o público B2B valoriza informações claras e métricas em tela. '
                'Criativos com densidade moderada (medium) têm score médio de 62.4 vs 56.5 em densidade baixa (+5.9 pts).'
            ),
            'impacto_quantificado': f'+5.9 pts no score médio histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(5.9, delta_model),
        })

    # 5. DURAÇÃO EXCESSIVA NO TIKTOK
    if plat == 'TikTok' and duracao > 30.0:
        delta_model = simular_ganho({'video_duration_s': 15.0, 'retention_rate': 0.65})
        recs.append({
            'area': 'Edição & Ritmo de Vídeo',
            'acao': f'Reduzir a duração total do vídeo de {duracao:.0f}s para a faixa ideal de 12 a 18 segundos.',
            'contexto': (
                f'No TikTok, vídeos longos sofrem queda acentuada de engajamento: no dataset, vídeos acima de 45s '
                f'têm média de apenas 37.2 pts vs 60.1 pts em vídeos com menos de 20s (queda de 22.9 pontos).'
            ),
            'impacto_quantificado': f'+22.9 pts no score histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(22.9, delta_model),
        })

    # 6. RETENÇÃO E TEMPO ASSISTIDO
    if retencao < 0.40:
        delta_model = simular_ganho({'retention_rate': 0.65})
        recs.append({
            'area': 'Retenção do Usuário',
            'acao': f'Acelerar a dinâmica dos cortes a cada 2-3s para elevar a retenção de {retencao*100:.1f}% para >60%.',
            'contexto': (
                f'A taxa de retenção tem forte correlação positiva (+0.40) com a performance do anúncio. '
                f'Vídeos com retenção >60% no {plat} entregam média de 67.4 pts vs 54.8 pts em criativos com retenção <40%.'
            ),
            'impacto_quantificado': f'+12.6 pts no score médio histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(12.6, delta_model),
        })

    # 7. CALL TO ACTION (HAS_CTA)
    if not has_cta:
        delta_model = simular_ganho({'has_cta': True})
        recs.append({
            'area': 'Conversão & Chamada para Ação',
            'acao': 'Inserir Call-to-Action (CTA) clara, direta e verbalizada nos últimos 3 a 5 segundos.',
            'contexto': (
                'Anúncios sem instrução explícita deixam o lead sem saber qual o próximo passo. '
                'A inclusão de CTA no dataset eleva o CTR em +8.8% e traz um ganho histórico de +8.05 pontos no klike_score.'
            ),
            'impacto_quantificado': f'+8.8% no CTR / conversões | +8.05 pts no score histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(8.05, delta_model),
        })

    # 8. LEGENDAS (HAS_SUBTITLE)
    if not has_sub:
        delta_model = simular_ganho({'has_subtitle': True})
        recs.append({
            'area': 'Acessibilidade & Consumo Mudo',
            'acao': 'Adicionar legendas dinâmicas sincronizadas palavra por palavra.',
            'contexto': (
                f'No {plat}, uma fração significativa dos usuários navega sem som ativo (especialmente no LinkedIn em escritório). '
                'Criativos legendados asseguram a entrega da mensagem e geram +6.9% no CTR e +6.69 pts no score histórico.'
            ),
            'impacto_quantificado': f'+6.9% no CTR | +6.69 pts no score histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(6.69, delta_model),
        })

    # 9. EQUILÍBRIO MÚSICA / VOZ (MUSIC_VOICE_RATIO)
    if music_ratio > 0.70:
        delta_model = simular_ganho({'music_voice_ratio': 0.40})
        recs.append({
            'area': 'Sonorização & Voz',
            'acao': 'Reduzir a proeminência da música e adicionar locução/narração humana (manter proporção entre 0.3 e 0.5).',
            'contexto': (
                'Vídeos em que a música abafa a narração têm média de score inferior (56.8 vs 61.0 em proporção equilibrada). '
                'Uma voz humana clara guia a narrativa e reforça a retenção da mensagem publicitária.'
            ),
            'impacto_quantificado': f'+4.1 pts no score médio histórico (+{delta_model:.1f} pts no XGBoost)',
            'impacto_score': max(4.1, delta_model),
        })

    # 10. RECOMENDAÇÕES COMPLEMENTARES (Garantir no mínimo 3 recomendações mesmo para criativos de alta performance)
    if len(recs) < top_n:
        if not is_retargeting:
            recs.append({
                'area': 'Segmentação & Audiência',
                'acao': 'Criar uma variação desta peça especificamente focada em retargeting (usuários já impactados).',
                'contexto': (
                    'No dataset analisado, campanhas com retargeting ativado obtiveram uma taxa de clique (CTR) '
                    '+43.6% superior às de topo de funil amplo, aproveitando criativos de alta qualidade.'
                ),
                'impacto_quantificado': '+43.6% no CTR de retargeting | +2.6 pts no score histórico',
                'impacto_score': 3.5,
            })

    if len(recs) < top_n:
        recs.append({
            'area': 'Otimização Contínua — Teste A/B de Hook',
            'acao': 'Produzir 2 variações adicionais de gancho visual (pergunta provocativa vs. estatística de impacto).',
            'contexto': (
                'Mesmo para criativos já performáticos, o gancho é a variável de maior peso (45.9% de importância no XGBoost). '
                'Iterações de hook são a alavanca mais eficiente para evitar saturação de criativo e manter o CTR elevado.'
            ),
            'impacto_quantificado': '+5% a +15% de sustentação de CTR a médio prazo',
            'impacto_score': 3.0,
        })

    if len(recs) < top_n:
        recs.append({
            'area': 'Otimização Contínua — Refinamento de CTA',
            'acao': 'Testar chamada para ação com senso de urgência ou benefício específico (ex: "Acesse grátis").',
            'contexto': 'CTAs contextualizadas geram maior taxa de conversão final para o mesmo volume de cliques.',
            'impacto_quantificado': '+5% a +10% na taxa de conversão final',
            'impacto_score': 2.5,
        })

    # Ordenação estrita por impacto potencial decrescente
    recs.sort(key=lambda r: r['impacto_score'], reverse=True)

    return recs[: max(top_n, 3)], score_baseline


def formatar_relatorio_texto(
    campanha: Dict[str, Any],
    recomendacoes: List[Dict[str, Any]],
    score_baseline: float,
) -> str:
    """
    Formata o resultado em texto simples, limpo e direto para visualização em terminal.
    """
    cid = campanha.get('campaign_id', 'Campanha Nova / Draft')
    plat = campanha.get('platform', 'N/A')
    fmt = campanha.get('format', 'N/A')
    cat = campanha.get('category', 'N/A')
    obj = campanha.get('objective', 'N/A')
    score_real = campanha.get('klike_score')

    dur = campanha.get('video_duration_s', 'N/A')
    dur_str = f'{dur:.1f}s' if isinstance(dur, (int, float)) and pd.notna(dur) else 'N/A'

    ret = campanha.get('retention_rate')
    if (ret is None or pd.isna(ret)) and isinstance(campanha.get('avg_watch_time_s'), (int, float)) and isinstance(dur, (int, float)) and dur > 0:
        ret = campanha['avg_watch_time_s'] / dur
    ret_str = f'{ret*100:.1f}%' if isinstance(ret, (int, float)) and pd.notna(ret) else 'N/A'

    txt_dens = campanha.get('text_density', 'N/A')

    linhas: List[str] = []
    linhas.append("=" * 80)
    linhas.append("🎯 SISTEMA DE RECOMENDAÇÕES DE CRIATIVOS — KLIKEREC ENGINE")
    linhas.append("=" * 80)
    linhas.append(f"Campanha Analisada : {cid}")
    linhas.append(f"Plataforma         : {plat} | Formato: {fmt} | Categoria: {cat} | Objetivo: {obj}")
    linhas.append(f"Métricas do Vídeo  : Duração: {dur_str} | Retenção: {ret_str} | Densidade de Texto: {txt_dens}")

    if score_real is not None and pd.notna(score_real):
        linhas.append(f"klike_score Real   : {float(score_real):.1f} pontos")
    linhas.append(f"klike_score Modelo : {score_baseline:.1f} pontos (Predição XGBoost)")

    linhas.append("\n" + "-" * 80)
    linhas.append(f"📋 {len(recomendacoes)} RECOMENDAÇÕES PRIORIZADAS DE MELHORIA (ORDENADAS POR IMPACTO)")
    linhas.append("-" * 80)

    score_acumulado = score_baseline

    for i, rec in enumerate(recomendacoes, 1):
        prioridade = "ALTA" if rec['impacto_score'] >= 10.0 else ("MÉDIA" if rec['impacto_score'] >= 6.0 else "CONTÍNUA")
        linhas.append(f"\n{i}. [{prioridade} | IMPACTO: ~+{rec['impacto_score']:.1f} pts] — {rec['area']}")
        linhas.append(f"   • Ação Recomendada     : {rec['acao']}")
        linhas.append(f"   • Por quê (Contexto)   : {rec['contexto']}")
        linhas.append(f"   • Ganho Quantificado   : {rec['impacto_quantificado']}")

    # Estimativa de projeção consolidada
    # Considera retornos decrescentes (raiz da soma ponderada dos top 3)
    ganhos_top3 = [r['impacto_score'] for r in recomendacoes[:3]]
    ganho_projetado = float(ganhos_top3[0] + 0.5 * sum(ganhos_top3[1:]))
    score_projetado = min(score_baseline + ganho_projetado, 98.0)

    linhas.append("\n" + "-" * 80)
    linhas.append("📈 PROJEÇÃO DE PERFORMANCE")
    linhas.append("-" * 80)
    linhas.append(f"Score Atual (Modelo)        : {score_baseline:.1f} pontos")
    linhas.append(f"Ganho Projetado (Top 3)     : +{ganho_projetado:.1f} pontos (considerando sinergia entre melhorias)")
    linhas.append(f"Score Estimado Pós-Ajustes  : {score_projetado:.1f} pontos (Escala 0 a 100)")
    linhas.append("=" * 80)

    return "\n".join(linhas)


def carregar_campanha_json(caminho_arquivo: str) -> Dict[str, Any]:
    """
    Carrega e valida os dados de uma campanha estritamente a partir de um arquivo .json.
    """
    # 1. Validação da extensão do arquivo
    if not caminho_arquivo.lower().endswith('.json'):
        raise ValueError(
            f"Formato de arquivo inválido: '{caminho_arquivo}'. "
            "A ferramenta aceita estritamente arquivos com extensão .json."
        )

    # 2. Validação da existência do arquivo
    if not os.path.isfile(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: '{caminho_arquivo}'.")

    # 3. Leitura e decodificação do JSON
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"O arquivo '{caminho_arquivo}' não contém um JSON válido: {err}")

    # 4. Tratamento se for lista de campanhas ou objeto único
    if isinstance(dados, list):
        if not dados or not isinstance(dados[0], dict):
            raise ValueError(f"O arquivo JSON '{caminho_arquivo}' contém uma lista vazia ou inválida.")
        return dados[0]
    elif isinstance(dados, dict):
        return dados
    else:
        raise ValueError("O conteúdo raiz do JSON deve ser um objeto (chave-valor) representando a campanha.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendações de Criativos Publicitários (KlikeRec CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de Uso:
  python recomendador.py campanha.json
  python recomendador.py caminho/para/campanha.json --top 5
        """,
    )

    parser.add_argument(
        "arquivo_json",
        type=str,
        help="Caminho para o arquivo .json contendo os dados da campanha publicitária",
    )
    parser.add_argument(
        "-k",
        "--top",
        type=int,
        default=3,
        help="Número mínimo de recomendações a exibir (padrão: 3)",
    )

    args = parser.parse_args()

    try:
        campanha = carregar_campanha_json(args.arquivo_json)
        recomendacoes, score_baseline = gerar_recomendacoes(campanha, top_n=args.top)
        relatorio = formatar_relatorio_texto(campanha, recomendacoes, score_baseline)
        print(relatorio)
    except Exception as e:
        print(f"❌ Erro ao processar recomendação: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

