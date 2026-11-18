import os
import re
import glob
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ECAC_DIR = os.path.join(BASE_DIR, "build", "MERGE_ECAC", "merge_ecac")
ORIG_DIR = os.path.join(BASE_DIR, "build", "MERGE_ECAC", "merge_original")
TRUTH_DIR = os.path.join(BASE_DIR, "build", "MERGE_ECAC", "truth")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


# ==============================================================================
# TAREFA 1: SEPARAÇÃO DO CONJUNTO DE TESTE (THE DATASET)
# ==============================================================================
def tarefa_1_separar_teste(metadata_path=None, test_songs_path=None):
    """
    Tarefa 1:
    Isolar as 409 músicas do conjunto de teste (31% do dataset) antes de qualquer
    análise ou ajuste de modelo. Nenhuma transformação ou imputação pode ser ajustada
    sobre este conjunto.
    Retorna:
        meta_train (DataFrame): 914 músicas para treino/validação
        meta_test (DataFrame): 409 músicas de teste intocadas
        meta_all (DataFrame): dataset completo original lido
    """
    if metadata_path is None:
        metadata_path = os.path.join(ECAC_DIR, "metadata.csv")
    if test_songs_path is None:
        test_songs_path = os.path.join(ECAC_DIR, "test_songs.csv")
    meta_all = pd.read_csv(metadata_path)
    tests = pd.read_csv(test_songs_path)
    is_test = meta_all["Audio_Song"].isin(tests["Audio_Song"])
    meta_test = meta_all[is_test].copy()
    meta_train = meta_all[~is_test].copy()
    print(f"Total de linhas lidas: {len(meta_all)}")
    print(f"Partição de Treino/Validação: {len(meta_train)} linhas")
    print(f"Partição de Teste (congelada): {len(meta_test)} linhas")
    return meta_train, meta_test, meta_all


# ==============================================================================
# TAREFA 2: COMPREENSÃO DOS DADOS E VERIFICAÇÃO DE FONTES (DATA UNDERSTANDING)
# ==============================================================================
def tarefa_2_2_verificar_fontes(dataset_dir=None, metadata_df=None):
    """
    Tarefa 2.2:
    Verificar se as quatro fontes (áudio, letras, capas de álbum e tabela de metadados)
    se interligam numa relação 1 para 1.
    """
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    if metadata_df is None:
        metadata_df = pd.read_csv(os.path.join(ECAC_DIR, "metadata.csv"))
    audio_files = glob.glob(os.path.join(dataset_dir, "audio", "*", "*"))
    lyrics_files = glob.glob(os.path.join(dataset_dir, "lyrics", "*", "*"))
    cover_files = glob.glob(os.path.join(dataset_dir, "album_covers", "*", "*"))
    audio_names = set(os.path.splitext(os.path.basename(f))[0] for f in audio_files)
    lyrics_names = set(os.path.splitext(os.path.basename(f))[0] for f in lyrics_files)
    cover_names = set(os.path.splitext(os.path.basename(f))[0] for f in cover_files)
    meta_audio = set(metadata_df["Audio_Song"])
    meta_lyrics = set(metadata_df["Lyric_Song"])
    relatorio = {"total_audio": len(audio_names),"total_lyrics": len(lyrics_names),"total_covers": len(cover_names),"meta_linhas": len(metadata_df),"meta_audio_unicos": len(meta_audio),"audio_sem_meta": len(audio_names - meta_audio),"meta_sem_audio": len(meta_audio - audio_names),"covers_em_falta": len(meta_audio - cover_names)}
    print("\nVerificação da relação 1:1 entre as 4 fontes:")
    print(f" - Ficheiros de Áudio: {relatorio['total_audio']}")
    print(f" - Ficheiros de Letras: {relatorio['total_lyrics']}")
    print(f" - Capas de Álbuns: {relatorio['total_covers']}")
    print(f" - Músicas em metadata.csv: {relatorio['meta_linhas']} linhas ({relatorio['meta_audio_unicos']} únicas)")
    print(f" - Músicas no áudio não presentes em metadata_ecac: {relatorio['audio_sem_meta']} (subamostradas)")
    print(f" - Capas em falta para as músicas dos metadados: {relatorio['covers_em_falta']}")
    print(" O relacionamento não é 1:1.")
    return relatorio

# ==============================================================================
# TAREFA 3: ANÁLISE EXPLORATÓRIA (EXPLORATORY ANALYSIS)
# ==============================================================================
def tarefa_3_1_descrever_colunas(df):
    """
    Tarefa 3.1:
    Descrever cada coluna: tipo, distribuição, intervalo de valores e taxa de valores em falta.
    """
    resumo = []
    for col in df.columns:
        n_miss = df[col].isna().sum()
        pct_miss = (n_miss / len(df)) * 100
        n_unique = df[col].nunique()
        dtype = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            vmin = df[col].min()
            vmax = df[col].max()
            vmean = df[col].mean()
            vstd = df[col].std()
            stats_str = f"min={vmin:.2f}, max={vmax:.2f}, mean={vmean:.2f}, std={vstd:.2f}"
        else:
            top_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
            stats_str = f"top_val='{str(top_val)[:20]}'"
        resumo.append({
            "coluna": col,
            "tipo": dtype,
            "em_falta": n_miss,
            "pct_em_falta": round(pct_miss, 2),
            "unicos": n_unique,
            "resumo": stats_str
        })

    tabela_resumo = pd.DataFrame(resumo)
    print("\nResumo das Primeiras Colunas:")
    print(tabela_resumo.head(10).to_string(index=False))
    return tabela_resumo

def tarefa_3_2_colunas_vazias_e_constantes(df):
    """
    Tarefa 3.2:
    Identificar colunas quase vazias, constantes ou que medem algo diferente do nome.
    """
    quase_vazias = [col for col in df.columns if df[col].isna().mean() > 0.90]
    constantes = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    anomalas = ["Duration", "LowestYear", "ActualYear"]
    print("\nColunas quase vazias (>90% NaN):", quase_vazias)
    print("Colunas constantes/quase constantes:", constantes)
    print("Colunas com distorções de escala/formato:", anomalas)
    return {"quase_vazias": quase_vazias,"constantes": constantes,"anomalas": anomalas}

def tarefa_3_3_fuga_do_alvo(df):
    """
    Tarefa 3.3:
    Medir o poder preditivo de cada coluna isolada sobre a variável alvo Quadrant.
    Detetar fuga de dados (target leakage).
    """
    y = df["Quadrant"].astype(str)
    resultados = []
    for col in df.columns:
        if col in ["Quadrant", "Audio_Song", "Lyric_Song", "AllMusic Id"]:
            continue

        s = df[col].copy()
        if pd.api.types.is_numeric_dtype(s):
            val = s.fillna(s.median()).values.reshape(-1, 1)
        else:
            val = pd.factorize(s.fillna("__MISSING__"))[0].reshape(-1, 1)
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(val, y)
        preds = clf.predict(val)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average='macro')
        resultados.append({"coluna": col,"acuracia": round(acc * 100, 2),"macro_f1": round(f1 * 100, 2)})
    res_df = pd.DataFrame(resultados).sort_values(by="macro_f1", ascending=False)
    print("\nTop 5 Colunas com Maior Associação com o Alvo (Potencial Leakage):")
    print(res_df.head(5).to_string(index=False))
    return res_df

def tarefa_3_5_analise_artistas(meta_df, test_songs_df):
    """
    Tarefa 3.5:
    Contar músicas por artista e calcular a sobreposição de artistas entre teste e treino.
    """
    meta_dedup = meta_df.drop_duplicates(subset=["Audio_Song"]).copy()
    contagem_artistas = meta_dedup["Artist"].value_counts()
    is_test = meta_dedup["Audio_Song"].isin(test_songs_df["Audio_Song"])
    test_meta = meta_dedup[is_test]
    train_meta = meta_dedup[~is_test]
    artistas_teste = set(test_meta["Artist"])
    artistas_treino = set(train_meta["Artist"])
    artistas_comuns = artistas_teste.intersection(artistas_treino)
    musicas_teste_partilhadas = test_meta[test_meta["Artist"].isin(artistas_comuns)]
    pct_partilhada = (len(musicas_teste_partilhadas) / len(test_meta)) * 100
    print(f"\nMúsicas por artista: Mínimo={contagem_artistas.min()}, Mediana={contagem_artistas.median()}, Máximo={contagem_artistas.max()} ({contagem_artistas.index[0]})")
    print(f"Artistas em teste: {len(artistas_teste)} | Em treino: {len(artistas_treino)}")
    print(f"Artistas partilhados entre teste e treino: {len(artistas_comuns)}")
    print(f"Músicas de teste que partilham artista com o treino: {len(musicas_teste_partilhadas)} ({pct_partilhada:.2f}%)")
    return {"artistas_comuns": len(artistas_comuns),"musicas_teste_partilhadas": len(musicas_teste_partilhadas),"pct_partilhada": pct_partilhada}


# ==============================================================================
# TAREFA 4: LIMPEZA DE DADOS (CLEANING)
# ==============================================================================
def tarefa_4_1_identificar_duplicados(df, test_songs_df):
    """
    Tarefa 4.1:
    Identificar linhas duplicadas idênticas (A1), gravações repetidas e sobreposição
    na fronteira de teste.
    """
    dup_identicos = df.duplicated().sum()
    df_temp = df.drop_duplicates().copy()
    df_temp["clean_art"] = df_temp["Artist"].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    df_temp["clean_tit"] = df_temp["Title"].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    agrupado = df_temp.groupby(["clean_art", "clean_tit"]).size()
    mesma_musica = agrupado[agrupado > 1]
    test_ids = set(test_songs_df["Audio_Song"])
    conflitos_teste = []
    for (art, tit), cnt in mesma_musica.items():
        sub = df_temp[(df_temp["clean_art"] == art) & (df_temp["clean_tit"] == tit)]
        tem_teste = any(sub["Audio_Song"].isin(test_ids))
        tem_treino = any(~sub["Audio_Song"].isin(test_ids))
        if tem_teste and tem_treino:
            conflitos_teste.append({"artista": art,"titulo": tit,"ids": sub["Audio_Song"].tolist()})
    print(f"\nLinhas idênticas encontradas: {dup_identicos}")
    print(f"Títulos de músicas com múltiplas entradas: {len(mesma_musica)}")
    print(f"Músicas presentes em ambos os lados do teste: {len(conflitos_teste)} (ex: {conflitos_teste})")
    return {"linhas_identicas": dup_identicos,"conflitos_fronteira": conflitos_teste}

def tarefa_4_2_imputar_valores(df, estrategia_numerica="median"):
    """
    Tarefa 4.2:
    Imputar valores em falta tendo em conta os diferentes mecanismos (MCAR, MAR, MNAR).
    """
    df_imp = df.copy()
    num_cols = df_imp.select_dtypes(include=[np.number]).columns

    imputer = SimpleImputer(strategy=estrategia_numerica)
    df_imp[num_cols] = imputer.fit_transform(df_imp[num_cols])
    cat_cols = df_imp.select_dtypes(include=['object']).columns
    for c in cat_cols:
        if c != "Quadrant":
            df_imp[c] = df_imp[c].fillna("Desconhecido")
    print(f"\nImputação concluída. Valores nulos restantes: {df_imp.isna().sum().sum()}")
    return df_imp


def extrair_ano(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    nums = re.findall(r'\b(19\d\d|20\d\d)\b', val_str)
    return int(nums[0]) if nums else np.nan


def conta_itens(v):
    return len(str(v).split(',')) if pd.notna(v) else 0


def tarefa_4_3_detetar_e_corrigir_erros(df):
    """
    Tarefa 4.3:
    Detetar e corrigir erros impossíveis (C1 milissegundos, C2 minutos, C3 sentinelas,
    C4 anos no futuro, C6 ActualYear < LowestYear, C7 listas de pesos com tamanho inconsistente).
    """
    df_clean = df.copy()
    erros_encontrados = {}
    dur = df_clean["Duration"]
    mask_c1 = dur > 1000
    mask_c2 = (dur < 15) & (dur > 0)
    mask_c3_dur = dur.isin([-1, 0, 9999])
    erros_encontrados["C1_duracao_ms"] = mask_c1.sum()
    erros_encontrados["C2_duracao_min"] = mask_c2.sum()
    erros_encontrados["C3_duracao_sentinela"] = mask_c3_dur.sum()
    df_clean.loc[mask_c1, "Duration"] = df_clean.loc[mask_c1, "Duration"] / 1000.0
    df_clean.loc[mask_c2, "Duration"] = df_clean.loc[mask_c2, "Duration"] * 60.0
    df_clean.loc[mask_c3_dur, "Duration"] = np.nan
    num_actual = pd.to_numeric(df_clean["ActualYear"], errors="coerce")
    mask_c4 = num_actual > 2026
    mask_c3_yr = num_actual < 1920
    erros_encontrados["C4_ano_futuro"] = mask_c4.sum()
    erros_encontrados["C3_ano_sentinela"] = mask_c3_yr.sum()
    df_clean.loc[mask_c4 | mask_c3_yr, "ActualYear"] = np.nan
    lowest_yr = df_clean["LowestYear"].apply(extrair_ano)
    actual_yr = pd.to_numeric(df_clean["ActualYear"], errors="coerce")
    mask_c6 = (actual_yr.notna()) & (lowest_yr.notna()) & (actual_yr < lowest_yr)
    erros_encontrados["C6_ano_menor_lowest"] = mask_c6.sum()
    df_clean.loc[mask_c6, "ActualYear"] = np.nan
    c7_count = 0
    for tags, pesos in [('Themes', 'ThemeWeights'), ('Genres', 'GenreWeights'), ('Styles', 'StyleWeights'), ('MoodsAll', 'MoodsAllWeights')]:
        if tags in df_clean.columns and pesos in df_clean.columns:
            mismatch = df_clean[tags].notna() & df_clean[pesos].notna() & (df_clean[tags].apply(conta_itens) != df_clean[pesos].apply(conta_itens))
            c7_count += mismatch.sum()
    erros_encontrados["C7_pesos_inconsistentes"] = c7_count
    print("\nResumo dos Erros Detetados:")
    for k, v in erros_encontrados.items():
        print(f" - {k}: {v}")
    return df_clean, erros_encontrados

def tarefa_4_4_detetar_outliers(df):
    """
    Tarefa 4.4:
    Detetar outliers com métodos univariados (Z-score, MAD Modified Z-score, Tukey IQR)
    e multivariados (DBSCAN e KMeans).
    """
    col = "Duration"
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    z = np.abs(stats.zscore(s))
    outliers_z = (z > 3).sum()
    med = s.median()
    mad = np.median(np.abs(s - med))
    mod_z = 0.6745 * np.abs(s - med) / (mad if mad > 0 else 1e-6)
    outliers_mad = (mod_z > 3.5).sum()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    outliers_tukey = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    print(f"\nOutliers Univariados em '{col}':")
    print(f" - Z-Score (>3): {outliers_z}")
    print(f" - Modified Z-Score MAD (>3.5): {outliers_mad}")
    print(f" - Tukey's Fences (1.5*IQR): {outliers_tukey}")
    num_cols = ['Duration', 'Relevance']
    df_sub = df[num_cols].dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sub)
    db = DBSCAN(eps=1.2, min_samples=5)
    labels = db.fit_predict(X_scaled)
    outliers_db = (labels == -1).sum()
    print(f"Outliers Multivariados (DBSCAN ruído eps=1.2): {outliers_db}")
    return {"z_score": outliers_z,"mad_score": outliers_mad,"tukey": outliers_tukey,"dbscan": outliers_db}

def tarefa_4_5_inspecionar_media(dataset_dir=None):
    """
    Tarefa 4.5:
    Inspecionar imagens de capas e áudio (duplicações e estatísticas).
    """
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    cover_files = glob.glob(os.path.join(dataset_dir, "album_covers", "*", "*"))
    hash_counts = {}
    for f in cover_files:
        try:
            with open(f, 'rb') as fp:
                h = hashlib.md5(fp.read()).hexdigest()
            hash_counts[h] = hash_counts.get(h, 0) + 1
        except Exception:
            pass
    capas_repetidas = sum(1 for cnt in hash_counts.values() if cnt > 1)
    total_imgs_repetidas = sum(cnt for cnt in hash_counts.values() if cnt > 1)
    print(f"\nInspeção de Media:")
    print(f" - Total de capas encontradas: {len(cover_files)}")
    print(f" - Grupos de capas visualmente idênticas: {capas_repetidas}")
    print(f" - Ficheiros envolvidos em capas repetidas: {total_imgs_repetidas}")
    return {"total_capas": len(cover_files),"grupos_repetidos": capas_repetidas,"ficheiros_repetidos": total_imgs_repetidas}

def tarefa_4_6_limpar_dataset_completo(meta_train, meta_test):
    """
    Tarefa 4.6:
    Produzir o dataset limpo combinando as regras de 4.1 a 4.5.
    As regras são ajustadas no treino e aplicadas ao teste.
    """
    train_clean = meta_train.drop_duplicates(subset=["Audio_Song"]).copy()
    test_clean = meta_test.drop_duplicates(subset=["Audio_Song"]).copy()
    train_clean = train_clean[train_clean["Audio_Song"] != "MT0007132421"]
    train_clean, _ = tarefa_4_3_detetar_e_corrigir_erros(train_clean)
    test_clean, _ = tarefa_4_3_detetar_e_corrigir_erros(test_clean)
    num_cols = train_clean.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    train_clean[num_cols] = imputer.fit_transform(train_clean[num_cols])
    test_clean[num_cols] = imputer.transform(test_clean[num_cols])
    print(f"\nLimpeza Concluída:")
    print(f" - Treino limpo: {train_clean.shape}")
    print(f" - Teste limpo: {test_clean.shape}")
    return train_clean, test_clean

# ==============================================================================
# TAREFA 5: DISTRIBUIÇÃO DE CLASSES (CLASS DISTRIBUTION)
# ==============================================================================
def tarefa_5_distribuicao_classes(df):
    """
    Tarefa 5:
    Calcular a distribuição das classes, a razão desequilíbrio e o desempenho
    de um classificador maioritário.
    """
    df_dedup = df.drop_duplicates(subset=["Audio_Song"])
    dist = df_dedup["Quadrant"].value_counts()
    maior = dist.max()
    menor = dist.min()
    razao = maior / menor
    classe_maj = dist.index[0]
    acuracia_maj = maior / len(df_dedup)
    macro_f1_maj = f1_score(df_dedup["Quadrant"], [classe_maj] * len(df_dedup), average='macro')
    print(f"\nDistribuição de Classes:\n{dist}")
    print(f"Razão Maior/Menor Classe ({dist.index[0]}/{dist.index[-1]}): {razao:.2f}:1")
    print(f"Classificador Maioritário -> Acurácia: {acuracia_maj*100:.2f}%, Macro-F1: {macro_f1_maj*100:.2f}%")
    return {"distribuicao": dist.to_dict(),"razao": razao,"acuracia_maioritaria": acuracia_maj,"macro_f1_maioritaria": macro_f1_maj}


# ==============================================================================
# TAREFA 6: AVALIAÇÃO DA LIMPEZA COM GROUND TRUTH (EVALUATION OF CLEANING)
# ==============================================================================
def tarefa_6_avaliar_limpeza_truth(truth_dir=None):
    """
    Tarefa 6:
    Avaliar os detetores do ponto 4 usando os ficheiros de truth/ e comparar
    métodos de imputação contra os valores originais.
    """
    if truth_dir is None:
        truth_dir = TRUTH_DIR
    truth_cells = pd.read_csv(os.path.join(truth_dir, "truth_cells.csv"))
    truth_rows = pd.read_csv(os.path.join(truth_dir, "truth_rows.csv"))
    print(f"\nResumo da Verdade Fundamental (Ground Truth):")
    print(f" - Modificações em células: {len(truth_cells)}")
    print(truth_cells['defect_code'].value_counts())
    print(f" - Modificações em linhas: {len(truth_rows)}")
    print(truth_rows['defect_code'].value_counts())
    b4 = truth_cells[truth_cells['defect_code'] == 'B4'].copy()
    b4_vals = b4['original_value'].astype(float)
    mae_b4_mean = mean_absolute_error(b4_vals, [b4_vals.mean()] * len(b4_vals))
    mae_b4_med = mean_absolute_error(b4_vals, [b4_vals.median()] * len(b4_vals))
    b1 = truth_cells[truth_cells['defect_code'] == 'B1'].copy()
    b1_vals = b1['original_value'].astype(float)
    mae_b1_mean = mean_absolute_error(b1_vals, [b1_vals.mean()] * len(b1_vals))
    mae_b1_med = mean_absolute_error(b1_vals, [b1_vals.median()] * len(b1_vals))
    print(f"\nComparação de Imputação contra Valores Reais:")
    print(f" - Relevância (B4 - MCAR): MAE Média = {mae_b4_mean:.4f} | MAE Mediana = {mae_b4_med:.4f}")
    print(f" - Duração (B1 - MNAR): MAE Média = {mae_b1_mean:.2f}s | MAE Mediana = {mae_b1_med:.2f}s")
    return {"truth_cells_total": len(truth_cells),"truth_rows_total": len(truth_rows),"mae_relevance_mean": mae_b4_mean,"mae_duration_mean": mae_b1_mean}


# ==============================================================================
# TAREFA 7: CLASSIFICADOR BASELINE KNN (THE CLASSIFIER)
# ==============================================================================
def tarefa_7_classificador_knn_baseline(meta_train_df):
    """
    Tarefa 7:
    Pipeline mais simples de ponta a ponta:
    - Colunas numéricas de metadata.csv como lidas
    - Imputação pela mediana
    - Sem escalonamento
    - kNN
    - Validação cruzada 5-fold estratificada por classe
    - Avaliar k=5 e k devidamente afinado (Nested CV com k ímpar de 1 a 31 e pesos uniform/distance)
    """
    df = meta_train_df.drop_duplicates(subset=["Audio_Song"]).copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[num_cols].values
    y = df["Quadrant"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs_k5, f1s_k5 = [], []
    for tr_idx, val_idx in skf.split(X, y):
        imp = SimpleImputer(strategy='median')
        X_tr = imp.fit_transform(X[tr_idx])
        X_val = imp.transform(X[val_idx])
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        clf.fit(X_tr, y[tr_idx])
        preds = clf.predict(X_val)
        accs_k5.append(accuracy_score(y[val_idx], preds))
        f1s_k5.append(f1_score(y[val_idx], preds, average='macro'))
    acc_k5_mean, acc_k5_std = np.mean(accs_k5), np.std(accs_k5)
    f1_k5_mean, f1_k5_std = np.mean(f1s_k5), np.std(f1s_k5)
    print(f"\n kNN Baseline (k=5 fixo, 5-Fold Stratified):")
    print(f" - Acurácia: {acc_k5_mean*100:.2f}% ± {acc_k5_std*100:.2f}%")
    print(f" - Macro-F1: {f1_k5_mean*100:.2f}% ± {f1_k5_std*100:.2f}%")
    nested_accs, nested_f1s = [], []
    melhores_k = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold)
        best_sc = -1
        best_param = (5, 'uniform')
        for k in range(1, 32, 2):
            for w in ['uniform', 'distance']:
                scs = []
                for in_tr, in_val in inner_cv.split(X_tr, y_tr):
                    imp = SimpleImputer(strategy='median')
                    in_X_tr = imp.fit_transform(X_tr[in_tr])
                    in_X_val = imp.transform(X_tr[in_val])

                    knn = KNeighborsClassifier(n_neighbors=k, weights=w)
                    knn.fit(in_X_tr, y_tr[in_tr])
                    scs.append(f1_score(y_tr[in_val], knn.predict(in_X_val), average='macro'))
                m_sc = np.mean(scs)
                if m_sc > best_sc:
                    best_sc = m_sc
                    best_param = (k, w)
        melhores_k.append(best_param)
        imp = SimpleImputer(strategy='median')
        X_tr_imp = imp.fit_transform(X_tr)
        X_val_imp = imp.transform(X_val)
        best_clf = KNeighborsClassifier(n_neighbors=best_param[0], weights=best_param[1])
        best_clf.fit(X_tr_imp, y_tr)
        preds = best_clf.predict(X_val_imp)
        nested_accs.append(accuracy_score(y_val, preds))
        nested_f1s.append(f1_score(y_val, preds, average='macro'))
    print(f"\n kNN com Hiperparâmetros Afinados (Nested CV):")
    print(f" - Melhores parâmetros por dobra: {melhores_k}")
    print(f" - Acurácia: {np.mean(nested_accs)*100:.2f}% ± {np.std(nested_accs)*100:.2f}%")
    print(f" - Macro-F1: {np.mean(nested_f1s)*100:.2f}% ± {np.std(nested_f1s)*100:.2f}%")
    return {"k5_acc": acc_k5_mean,"k5_f1": f1_k5_mean,"tuned_acc": np.mean(nested_accs),"tuned_f1": np.mean(nested_f1s),"best_params": melhores_k}

# ==============================================================================
# TAREFA 8: PROTOCOLO DE VALIDAÇÃO (VALIDATION PROTOCOL)
# ==============================================================================
def tarefa_8_protocolo_validacao_experiencia(meta_train_df):
    """
    Tarefa 8 (Experiência 1):
    Comparar a mesma pipeline avaliada de duas formas:
    1) 5-fold CV estratificada por classe
    2) 5-fold CV com todas as músicas de cada artista na mesma dobra (artist-disjoint)
    Reportar ambas as estimativas e a diferença (gap).
    """
    df = meta_train_df.drop_duplicates(subset=["Audio_Song"]).copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[num_cols].values
    y = df["Quadrant"].values
    artists = df["Artist"].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True).values

    def _run_nested_cv(cv_splitter, groups=None):
        accs, f1s = [], []
        splits = cv_splitter.split(X, y, groups) if groups is not None else cv_splitter.split(X, y)
        for fold, (tr_idx, val_idx) in enumerate(splits):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            g_tr = groups[tr_idx] if groups is not None else None
            inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=fold) if groups is not None else StratifiedKFold(n_splits=5, shuffle=True, random_state=fold)
            best_sc, best_param = -1, (5, 'uniform')
            for k in range(1, 32, 2):
                for w in ['uniform', 'distance']:
                    scs = []
                    in_splits = inner_cv.split(X_tr, y_tr, g_tr) if g_tr is not None else inner_cv.split(X_tr, y_tr)
                    for in_tr, in_val in in_splits:
                        imp = SimpleImputer(strategy='median')
                        in_X_tr = imp.fit_transform(X_tr[in_tr])
                        in_X_val = imp.transform(X_tr[in_val])
                        knn = KNeighborsClassifier(n_neighbors=k, weights=w)
                        knn.fit(in_X_tr, y_tr[in_tr])
                        scs.append(f1_score(y_tr[in_val], knn.predict(in_X_val), average='macro'))
                    m_sc = np.mean(scs)
                    if m_sc > best_sc:
                        best_sc, best_param = m_sc, (k, w)
            imp = SimpleImputer(strategy='median')
            X_tr_imp = imp.fit_transform(X_tr)
            X_val_imp = imp.transform(X_val)
            best_clf = KNeighborsClassifier(n_neighbors=best_param[0], weights=best_param[1])
            best_clf.fit(X_tr_imp, y_tr)
            preds = best_clf.predict(X_val_imp)
            accs.append(accuracy_score(y_val, preds))
            f1s.append(f1_score(y_val, preds, average='macro'))
        return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)

    strat_acc, strat_acc_std, strat_f1, strat_f1_std = _run_nested_cv(StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    disjoint_acc, disjoint_acc_std, disjoint_f1, disjoint_f1_std = _run_nested_cv(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), groups=artists)
    gap_acc = (strat_acc - disjoint_acc) * 100
    gap_f1 = (strat_f1 - disjoint_f1) * 100
    print(f"\nComparação dos Protocolos de Validação (Nested CV):")
    print(f" - Estratificado por Classe   -> Acurácia: {strat_acc*100:.2f}% ± {strat_acc_std*100:.2f}%, Macro-F1: {strat_f1*100:.2f}% ± {strat_f1_std*100:.2f}%")
    print(f" - Artist-Disjoint (por Grupo)-> Acurácia: {disjoint_acc*100:.2f}% ± {disjoint_acc_std*100:.2f}%, Macro-F1: {disjoint_f1*100:.2f}% ± {disjoint_f1_std*100:.2f}%")
    print(f" - Desvio (Gap de Generalização): Acurácia: {gap_acc:+.2f} pp, Macro-F1: {gap_f1:+.2f} pp")
    return {"stratified_acc": strat_acc,"stratified_f1": strat_f1,"disjoint_acc": disjoint_acc,"disjoint_f1": disjoint_f1,"gap_f1": gap_f1}

# ==============================================================================
# EXECUÇÃO PRINCIPAL DE TODAS AS TAREFAS (1 A 8)
# ==============================================================================
if __name__ == "__main__":
    meta_train, meta_test, meta_all = tarefa_1_separar_teste()
    tarefa_2_2_verificar_fontes(metadata_df=meta_all)
    tarefa_3_1_descrever_colunas(meta_train)
    tarefa_3_2_colunas_vazias_e_constantes(meta_train)
    tarefa_3_3_fuga_do_alvo(meta_train)
    tarefa_3_5_analise_artistas(meta_all, meta_test)
    tarefa_4_1_identificar_duplicados(meta_all, meta_test)
    df_imputado = tarefa_4_2_imputar_valores(meta_train)
    df_corrigido, erros = tarefa_4_3_detetar_e_corrigir_erros(meta_train)
    tarefa_4_4_detetar_outliers(meta_train)
    tarefa_4_5_inspecionar_media()
    train_clean, test_clean = tarefa_4_6_limpar_dataset_completo(meta_train, meta_test)
    tarefa_5_distribuicao_classes(meta_all)
    tarefa_6_avaliar_limpeza_truth()
    tarefa_7_classificador_knn_baseline(meta_train)
    tarefa_8_protocolo_validacao_experiencia(meta_train)
