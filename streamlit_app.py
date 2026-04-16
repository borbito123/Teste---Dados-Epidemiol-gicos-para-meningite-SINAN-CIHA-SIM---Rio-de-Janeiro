from __future__ import annotations

import glob
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Painel Epidemiológico com Parquets DATASUS",
    page_icon="📊",
    layout="wide",
)


SOURCE_CONFIG = {
    "SINAN": {
        "subtitle": "Notificações/casos (ex.: meningite)",
        "date_candidates": [
            "DT_SIN_PRI",
            "DT_NOTIFIC",
            "DT_INVEST",
            "DT_ENCERRA",
            "DT_DIGITA",
        ],
        "sex_candidates": ["CS_SEXO", "SEXO"],
        "age_candidates": ["NU_IDADE_N", "IDADE", "IDADE_ANOS", "IDADEANOS"],
        "municipality_candidates": [
            "ID_MN_RESI",
            "ID_MN_OCORR",
            "CODMUNRES",
            "CODMUNOCOR",
            "MUNIC_RES",
            "MUN_RES",
            "MUN_OCOR",
        ],
        "outcome_candidates": ["EVOLUCAO", "EVOLUÇÃO", "CLASSI_FIN", "CRITERIO"],
        "diagnosis_candidates": [
            "ID_AGRAVO",
            "AGRAVO",
            "CLASSI_FIN",
            "CRITERIO",
            "DIAG_PRINC",
            "CID10",
            "CID",
        ],
        "time_title": "Série temporal de notificações/casos",
    },
    "SIM": {
        "subtitle": "Óbitos",
        "date_candidates": ["DTOBITO", "DT_OBITO", "DTNASC", "DT_NASC"],
        "sex_candidates": ["SEXO", "CS_SEXO"],
        "age_candidates": ["IDADE", "IDADEANOS", "IDADE_ANOS"],
        "municipality_candidates": [
            "CODMUNRES",
            "CODMUNOCOR",
            "MUNRES",
            "MUNOCOR",
            "ID_MN_RESI",
            "ID_MN_OCORR",
        ],
        "outcome_candidates": ["LOCOCOR", "CIRCOBITO", "ASSISTMED"],
        "diagnosis_candidates": [
            "CAUSABAS",
            "CAUSABAS_O",
            "LINHAA",
            "LINHAB",
            "LINHAC",
            "LINHAD",
        ],
        "time_title": "Série temporal de óbitos",
    },
    "CIHA": {
        "subtitle": "Internações / atendimentos",
        "date_candidates": [
            "DT_INTER",
            "DT_INTERNA",
            "DT_ATEND",
            "DT_SAIDA",
            "DT_COMPET",
            "COMPET",
            "ANO_CMPT",
        ],
        "sex_candidates": ["SEXO", "CS_SEXO"],
        "age_candidates": ["IDADE", "IDADE_ANOS", "IDADEANOS", "NU_IDADE_N"],
        "municipality_candidates": [
            "CODMUNRES",
            "CODMUN",
            "MUNIC_RES",
            "MUN_RES",
            "ID_MN_RESI",
        ],
        "outcome_candidates": ["MOTSAI", "MOT_SAIDA", "TIPO_ALTA", "DESFECHO"],
        "diagnosis_candidates": [
            "CIDPRI",
            "CID_PRINC",
            "DIAG_PRINC",
            "CID",
            "DIAG",
            "PROC_PRINC",
        ],
        "time_title": "Série temporal de internações/atendimentos",
    },
}


@dataclass
class ColumnSelection:
    date_col: Optional[str]
    sex_col: Optional[str]
    age_col: Optional[str]
    municipality_col: Optional[str]
    outcome_col: Optional[str]
    diagnosis_col: Optional[str]
    age_mode: str


# ---------- Helpers de texto / SQL ----------

def normalize_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper().strip()
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch)
    return "".join(keep)


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def qstring(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parquet_ref(paths: Sequence[str]) -> str:
    quoted = ", ".join(qstring(p) for p in paths)
    return f"read_parquet([{quoted}], union_by_name=true)"


def clean_str_expr(col: str) -> str:
    q = qident(col)
    return f"NULLIF(TRIM(CAST({q} AS VARCHAR)), '')"


def choose_candidate(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    if not columns:
        return None
    norm_map = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        match = norm_map.get(normalize_name(candidate))
        if match:
            return match
    candidate_norms = [normalize_name(c) for c in candidates]
    for col in columns:
        ncol = normalize_name(col)
        if any(cand in ncol or ncol in cand for cand in candidate_norms):
            return col
    return None


def detect_columns(source: str, columns: Sequence[str]) -> ColumnSelection:
    cfg = SOURCE_CONFIG[source]
    age_col = choose_candidate(columns, cfg["age_candidates"])
    age_mode = "Automático"
    if age_col and normalize_name(age_col) in {"NUIDADEN", "IDADE"}:
        age_mode = "DATASUS codificada"
    return ColumnSelection(
        date_col=choose_candidate(columns, cfg["date_candidates"]),
        sex_col=choose_candidate(columns, cfg["sex_candidates"]),
        age_col=age_col,
        municipality_col=choose_candidate(columns, cfg["municipality_candidates"]),
        outcome_col=choose_candidate(columns, cfg["outcome_candidates"]),
        diagnosis_col=choose_candidate(columns, cfg["diagnosis_candidates"]),
        age_mode=age_mode,
    )


def date_expr(col: str) -> str:
    txt = clean_str_expr(col)
    q = qident(col)
    return f"""
    CAST(
        COALESCE(
            TRY_CAST({q} AS DATE),
            CASE WHEN regexp_matches({txt}, '^\\d{{4}}-\\d{{2}}-\\d{{2}}$')
                 THEN CAST(try_strptime({txt}, '%Y-%m-%d') AS DATE) END,
            CASE WHEN regexp_matches({txt}, '^\\d{{8}}$') AND SUBSTR({txt}, 1, 4) BETWEEN '1900' AND '2099'
                 THEN CAST(try_strptime({txt}, '%Y%m%d') AS DATE) END,
            CASE WHEN regexp_matches({txt}, '^\\d{{8}}$')
                 THEN CAST(try_strptime({txt}, '%d%m%Y') AS DATE) END,
            CASE WHEN regexp_matches({txt}, '^\\d{{2}}/\\d{{2}}/\\d{{4}}$')
                 THEN CAST(try_strptime({txt}, '%d/%m/%Y') AS DATE) END,
            CASE WHEN regexp_matches({txt}, '^\\d{{2}}-\\d{{2}}-\\d{{4}}$')
                 THEN CAST(try_strptime({txt}, '%d-%m-%Y') AS DATE) END
        ) AS DATE
    )
    """


def datasus_age_expr(col: str) -> str:
    txt = clean_str_expr(col)
    return f"""
    CASE
        WHEN {txt} IS NULL THEN NULL
        WHEN regexp_matches({txt}, '^\\d{{1,3}}$') AND TRY_CAST({txt} AS DOUBLE) BETWEEN 0 AND 120
            THEN TRY_CAST({txt} AS DOUBLE)
        WHEN regexp_matches({txt}, '^\\d{{2,4}}$') THEN
            CASE SUBSTR({txt}, 1, 1)
                WHEN '0' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / (365.25 * 24)
                WHEN '1' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / (365.25 * 24)
                WHEN '2' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / 365.25
                WHEN '3' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / 12
                WHEN '4' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE)
                WHEN '5' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE)
                ELSE NULL
            END
        ELSE NULL
    END
    """


def direct_age_expr(col: str) -> str:
    txt = clean_str_expr(col)
    return f"TRY_CAST(REPLACE({txt}, ',', '.') AS DOUBLE)"


def build_age_expr(col: Optional[str], age_mode: str) -> Optional[str]:
    if not col:
        return None
    if age_mode == "Anos diretos":
        return direct_age_expr(col)
    if age_mode == "DATASUS codificada":
        return datasus_age_expr(col)
    ncol = normalize_name(col)
    if ncol in {"NUIDADEN", "IDADE"}:
        return f"COALESCE({datasus_age_expr(col)}, {direct_age_expr(col)})"
    return direct_age_expr(col)


def sex_expr(col: str) -> str:
    txt = clean_str_expr(col)
    return f"""
    CASE UPPER({txt})
        WHEN 'M' THEN 'Masculino'
        WHEN '1' THEN 'Masculino'
        WHEN 'F' THEN 'Feminino'
        WHEN '2' THEN 'Feminino'
        WHEN 'I' THEN 'Ignorado/Outro'
        WHEN '0' THEN 'Ignorado/Outro'
        WHEN '9' THEN 'Ignorado/Outro'
        WHEN 'IGN' THEN 'Ignorado/Outro'
        ELSE COALESCE({txt}, 'Ignorado/Outro')
    END
    """


def pretty_label(col: Optional[str]) -> str:
    return col if col else "(não selecionado)"


# ---------- Leitura dos arquivos ----------

def save_uploaded_files(uploaded_files: Sequence, source_key: str) -> List[str]:
    if not uploaded_files:
        return []
    temp_dir = tempfile.mkdtemp(prefix=f"streamlit_{source_key.lower()}_")
    paths: List[str] = []
    for up in uploaded_files:
        dest = os.path.join(temp_dir, up.name)
        with open(dest, "wb") as f:
            f.write(up.getbuffer())
        paths.append(dest)
    return sorted(paths)


def resolve_paths(source_key: str, mode: str, local_glob: str, uploaded_files: Sequence) -> List[str]:
    if mode == "Pasta / glob local":
        if not local_glob.strip():
            return []
        return sorted(glob.glob(local_glob.strip()))
    return save_uploaded_files(uploaded_files, source_key)


def run_query(paths: Sequence[str], sql: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def get_schema(paths: Sequence[str]) -> pd.DataFrame:
    ref = parquet_ref(paths)
    sql = f"DESCRIBE SELECT * FROM {ref}"
    df = run_query(paths, sql)
    if "column_name" in df.columns and "column_type" in df.columns:
        return df[["column_name", "column_type"]].rename(
            columns={"column_name": "coluna", "column_type": "tipo"}
        )
    return df


def get_columns(paths: Sequence[str]) -> List[str]:
    schema = get_schema(paths)
    return schema["coluna"].astype(str).tolist()


def dataset_metrics(paths: Sequence[str], dt_expr: Optional[str], where_sql: str = "") -> Dict[str, object]:
    ref = parquet_ref(paths)
    metrics = {}
    total_sql = f"SELECT COUNT(*) AS n FROM {ref} {where_sql}"
    metrics["n"] = int(run_query(paths, total_sql).iloc[0, 0])
    if dt_expr:
        range_sql = f"SELECT MIN({dt_expr}) AS dt_min, MAX({dt_expr}) AS dt_max FROM {ref} {where_sql}"
        dfr = run_query(paths, range_sql)
        metrics["dt_min"] = dfr.iloc[0, 0]
        metrics["dt_max"] = dfr.iloc[0, 1]
    return metrics


def preview_df(paths: Sequence[str], limit: int = 10) -> pd.DataFrame:
    ref = parquet_ref(paths)
    return run_query(paths, f"SELECT * FROM {ref} LIMIT {limit}")


# ---------- Filtros ----------

def get_year_bounds(paths: Sequence[str], dt_expr: Optional[str]) -> Optional[Tuple[int, int]]:
    if not dt_expr:
        return None
    ref = parquet_ref(paths)
    sql = f"""
    SELECT
        MIN(EXTRACT(YEAR FROM {dt_expr})) AS ano_min,
        MAX(EXTRACT(YEAR FROM {dt_expr})) AS ano_max
    FROM {ref}
    WHERE {dt_expr} IS NOT NULL
    """
    df = run_query(paths, sql)
    if df.empty or pd.isna(df.iloc[0, 0]) or pd.isna(df.iloc[0, 1]):
        return None
    return int(df.iloc[0, 0]), int(df.iloc[0, 1])


def top_values(paths: Sequence[str], expr: str, limit: int = 25) -> List[str]:
    ref = parquet_ref(paths)
    sql = f"""
    SELECT {expr} AS valor, COUNT(*) AS n
    FROM {ref}
    WHERE {expr} IS NOT NULL
    GROUP BY 1
    ORDER BY 2 DESC, 1
    LIMIT {limit}
    """
    df = run_query(paths, sql)
    if df.empty:
        return []
    return [str(v) for v in df["valor"].dropna().tolist()]


def build_where_clause(
    dt_expr: Optional[str],
    year_range: Optional[Tuple[int, int]],
    sex_expression: Optional[str],
    selected_sex: Sequence[str],
    municipality_expression: Optional[str],
    selected_municipalities: Sequence[str],
    outcome_expression: Optional[str],
    selected_outcomes: Sequence[str],
) -> str:
    clauses: List[str] = []
    if dt_expr and year_range:
        clauses.append(
            f"EXTRACT(YEAR FROM {dt_expr}) BETWEEN {int(year_range[0])} AND {int(year_range[1])}"
        )
    if sex_expression and selected_sex:
        sex_list = ", ".join(qstring(x) for x in selected_sex)
        clauses.append(f"{sex_expression} IN ({sex_list})")
    if municipality_expression and selected_municipalities:
        muni_list = ", ".join(qstring(x) for x in selected_municipalities)
        clauses.append(f"{municipality_expression} IN ({muni_list})")
    if outcome_expression and selected_outcomes:
        out_list = ", ".join(qstring(x) for x in selected_outcomes)
        clauses.append(f"{outcome_expression} IN ({out_list})")
    if not clauses:
        return ""
    return "WHERE " + " AND ".join(clauses)


# ---------- Gráficos ----------

def query_time_series(paths: Sequence[str], dt_expr: str, where_sql: str, freq_sql: str) -> pd.DataFrame:
    ref = parquet_ref(paths)
    sql = f"""
    WITH base AS (
        SELECT {dt_expr} AS dt
        FROM {ref}
        {where_sql}
    )
    SELECT date_trunc('{freq_sql}', dt) AS periodo, COUNT(*) AS n
    FROM base
    WHERE dt IS NOT NULL
    GROUP BY 1
    ORDER BY 1
    """
    return run_query(paths, sql)


def query_month_heatmap(paths: Sequence[str], dt_expr: str, where_sql: str) -> pd.DataFrame:
    ref = parquet_ref(paths)
    sql = f"""
    WITH base AS (
        SELECT {dt_expr} AS dt
        FROM {ref}
        {where_sql}
    )
    SELECT
        EXTRACT(YEAR FROM dt) AS ano,
        EXTRACT(MONTH FROM dt) AS mes,
        COUNT(*) AS n
    FROM base
    WHERE dt IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    return run_query(paths, sql)


def query_top_category(paths: Sequence[str], cat_expr: str, where_sql: str, top_n: int = 20) -> pd.DataFrame:
    ref = parquet_ref(paths)
    sql = f"""
    SELECT {cat_expr} AS categoria, COUNT(*) AS n
    FROM {ref}
    {where_sql}
    AND {cat_expr} IS NOT NULL
    GROUP BY 1
    ORDER BY 2 DESC, 1
    LIMIT {top_n}
    """
    if not where_sql:
        sql = f"""
        SELECT {cat_expr} AS categoria, COUNT(*) AS n
        FROM {ref}
        WHERE {cat_expr} IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC, 1
        LIMIT {top_n}
        """
    return run_query(paths, sql)


def query_age_distribution(paths: Sequence[str], age_sql: str, where_sql: str) -> pd.DataFrame:
    ref = parquet_ref(paths)
    sql = f"""
    WITH base AS (
        SELECT {age_sql} AS idade
        FROM {ref}
        {where_sql}
    )
    SELECT FLOOR(idade / 5) * 5 AS faixa_ini, COUNT(*) AS n
    FROM base
    WHERE idade BETWEEN 0 AND 120
    GROUP BY 1
    ORDER BY 1
    """
    return run_query(paths, sql)


def query_pyramid(paths: Sequence[str], age_sql: str, sex_sql: str, where_sql: str) -> pd.DataFrame:
    ref = parquet_ref(paths)
    sql = f"""
    WITH base AS (
        SELECT {age_sql} AS idade, {sex_sql} AS sexo
        FROM {ref}
        {where_sql}
    )
    SELECT sexo, FLOOR(idade / 5) * 5 AS faixa_ini, COUNT(*) AS n
    FROM base
    WHERE idade BETWEEN 0 AND 120
      AND sexo IN ('Masculino', 'Feminino')
    GROUP BY 1, 2
    ORDER BY 2, 1
    """
    return run_query(paths, sql)


def query_missingness(
    paths: Sequence[str],
    selections: ColumnSelection,
    dt_expr: Optional[str],
    where_sql: str,
) -> pd.DataFrame:
    ref = parquet_ref(paths)
    checks = []
    if selections.date_col and dt_expr:
        checks.append(("Data", f"SUM(CASE WHEN {dt_expr} IS NULL THEN 1 ELSE 0 END)"))
    if selections.sex_col:
        checks.append(("Sexo", f"SUM(CASE WHEN {clean_str_expr(selections.sex_col)} IS NULL THEN 1 ELSE 0 END)"))
    if selections.age_col:
        age_expr_sql = build_age_expr(selections.age_col, selections.age_mode)
        checks.append(("Idade", f"SUM(CASE WHEN {age_expr_sql} IS NULL THEN 1 ELSE 0 END)"))
    if selections.municipality_col:
        checks.append(
            (
                "Município",
                f"SUM(CASE WHEN {clean_str_expr(selections.municipality_col)} IS NULL THEN 1 ELSE 0 END)",
            )
        )
    if selections.outcome_col:
        checks.append(("Desfecho", f"SUM(CASE WHEN {clean_str_expr(selections.outcome_col)} IS NULL THEN 1 ELSE 0 END)"))
    if selections.diagnosis_col:
        checks.append(("Diagnóstico/CID", f"SUM(CASE WHEN {clean_str_expr(selections.diagnosis_col)} IS NULL THEN 1 ELSE 0 END)"))
    if not checks:
        return pd.DataFrame()

    union_sql = " UNION ALL ".join(
        [f"SELECT {qstring(label)} AS campo, {expr} AS faltantes FROM {ref} {where_sql}" for label, expr in checks]
    )
    total_sql = f"SELECT COUNT(*) AS total FROM {ref} {where_sql}"
    total = int(run_query(paths, total_sql).iloc[0, 0])
    df = run_query(paths, union_sql)
    if total > 0:
        df["pct_faltante"] = (df["faltantes"] / total * 100).round(2)
    else:
        df["pct_faltante"] = 0.0
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def add_download_button(df: pd.DataFrame, file_name: str, label: str) -> None:
    if df is None or df.empty:
        return
    st.download_button(
        label=label,
        data=to_csv_bytes(df),
        file_name=file_name,
        mime="text/csv",
        use_container_width=False,
    )


# ---------- Renderização por fonte ----------

def render_source_tab(source: str) -> Optional[Dict[str, object]]:
    cfg = SOURCE_CONFIG[source]
    st.subheader(f"{source} — {cfg['subtitle']}")

    mode = st.radio(
        f"Como carregar os parquets de {source}?",
        ["Upload de arquivos", "Pasta / glob local"],
        horizontal=True,
        key=f"mode_{source}",
    )

    uploaded_files = []
    local_glob = ""

    if mode == "Upload de arquivos":
        uploaded_files = st.file_uploader(
            f"Envie um ou mais arquivos .parquet de {source}",
            type=["parquet"],
            accept_multiple_files=True,
            key=f"uploader_{source}",
        )
    else:
        local_glob = st.text_input(
            f"Informe um caminho ou glob local para os parquets de {source}",
            value="",
            placeholder="Ex.: Bases_Datasus_Municipio_Rio_de_Janeiro/SIM/data/parquet/*.parquet",
            key=f"glob_{source}",
        )

    paths = resolve_paths(source, mode, local_glob, uploaded_files)

    if not paths:
        st.info(f"Carregue ao menos um Parquet de {source} para habilitar os gráficos.")
        return None

    st.caption(f"Arquivos identificados: {len(paths)}")

    try:
        schema = get_schema(paths)
    except Exception as exc:
        st.error(f"Não foi possível ler os parquets de {source}: {exc}")
        return None

    columns = schema["coluna"].astype(str).tolist()
    suggestions = detect_columns(source, columns)

    with st.expander("Configuração automática das colunas", expanded=True):
        left, right = st.columns(2)
        with left:
            date_col = st.selectbox(
                "Coluna de data",
                options=[None] + columns,
                index=(columns.index(suggestions.date_col) + 1) if suggestions.date_col in columns else 0,
                format_func=lambda x: "(não usar)" if x is None else x,
                key=f"date_col_{source}",
            )
            sex_col = st.selectbox(
                "Coluna de sexo",
                options=[None] + columns,
                index=(columns.index(suggestions.sex_col) + 1) if suggestions.sex_col in columns else 0,
                format_func=lambda x: "(não usar)" if x is None else x,
                key=f"sex_col_{source}",
            )
            age_col = st.selectbox(
                "Coluna de idade",
                options=[None] + columns,
                index=(columns.index(suggestions.age_col) + 1) if suggestions.age_col in columns else 0,
                format_func=lambda x: "(não usar)" if x is None else x,
                key=f"age_col_{source}",
            )
            age_mode = st.selectbox(
                "Como interpretar a idade?",
                options=["Automático", "Anos diretos", "DATASUS codificada"],
                index=["Automático", "Anos diretos", "DATASUS codificada"].index(suggestions.age_mode)
                if suggestions.age_mode in ["Automático", "Anos diretos", "DATASUS codificada"]
                else 0,
                key=f"age_mode_{source}",
                help=(
                    "Use 'DATASUS codificada' quando a idade vier em formato compacto típico do DATASUS. "
                    "A conversão para anos é aproximada para horas/dias/meses."
                ),
            )
        with right:
            municipality_col = st.selectbox(
                "Coluna de município",
                options=[None] + columns,
                index=(columns.index(suggestions.municipality_col) + 1)
                if suggestions.municipality_col in columns
                else 0,
                format_func=lambda x: "(não usar)" if x is None else x,
                key=f"municipality_col_{source}",
            )
            outcome_col = st.selectbox(
                "Coluna de desfecho / classificação",
                options=[None] + columns,
                index=(columns.index(suggestions.outcome_col) + 1)
                if suggestions.outcome_col in columns
                else 0,
                format_func=lambda x: "(não usar)" if x is None else x,
                key=f"outcome_col_{source}",
            )
            diagnosis_col = st.selectbox(
                "Coluna de diagnóstico / CID / agravo",
                options=[None] + columns,
                index=(columns.index(suggestions.diagnosis_col) + 1)
                if suggestions.diagnosis_col in columns
                else 0,
                format_func=lambda x: "(não usar)" if x is None else x,
                key=f"diagnosis_col_{source}",
            )

        st.markdown(
            f"**Sugestões detectadas** — data: `{pretty_label(suggestions.date_col)}`, sexo: `{pretty_label(suggestions.sex_col)}`, "
            f"idade: `{pretty_label(suggestions.age_col)}`, município: `{pretty_label(suggestions.municipality_col)}`, "
            f"desfecho: `{pretty_label(suggestions.outcome_col)}`, diagnóstico: `{pretty_label(suggestions.diagnosis_col)}`."
        )

    selections = ColumnSelection(
        date_col=date_col,
        sex_col=sex_col,
        age_col=age_col,
        municipality_col=municipality_col,
        outcome_col=outcome_col,
        diagnosis_col=diagnosis_col,
        age_mode=age_mode,
    )

    dt_expr = date_expr(date_col) if date_col else None
    sex_sql = sex_expr(sex_col) if sex_col else None
    age_sql = build_age_expr(age_col, age_mode) if age_col else None
    municipality_sql = clean_str_expr(municipality_col) if municipality_col else None
    outcome_sql = clean_str_expr(outcome_col) if outcome_col else None
    diagnosis_sql = clean_str_expr(diagnosis_col) if diagnosis_col else None

    with st.expander("Filtros analíticos", expanded=True):
        year_range = None
        if dt_expr:
            bounds = get_year_bounds(paths, dt_expr)
            if bounds:
                year_range = st.slider(
                    "Faixa de anos",
                    min_value=bounds[0],
                    max_value=bounds[1],
                    value=bounds,
                    key=f"year_range_{source}",
                )
        c1, c2, c3 = st.columns(3)
        selected_sex: List[str] = []
        selected_municipalities: List[str] = []
        selected_outcomes: List[str] = []
        if sex_sql:
            with c1:
                sex_options = top_values(paths, sex_sql, limit=10)
                selected_sex = st.multiselect(
                    "Sexo",
                    options=sex_options,
                    default=[],
                    key=f"sex_filter_{source}",
                )
        if municipality_sql:
            with c2:
                muni_options = top_values(paths, municipality_sql, limit=40)
                selected_municipalities = st.multiselect(
                    "Município",
                    options=muni_options,
                    default=[],
                    key=f"municipality_filter_{source}",
                )
        if outcome_sql:
            with c3:
                outcome_options = top_values(paths, outcome_sql, limit=25)
                selected_outcomes = st.multiselect(
                    "Desfecho / classificação",
                    options=outcome_options,
                    default=[],
                    key=f"outcome_filter_{source}",
                )

    where_sql = build_where_clause(
        dt_expr=dt_expr,
        year_range=year_range,
        sex_expression=sex_sql,
        selected_sex=selected_sex,
        municipality_expression=municipality_sql,
        selected_municipalities=selected_municipalities,
        outcome_expression=outcome_sql,
        selected_outcomes=selected_outcomes,
    )

    metrics = dataset_metrics(paths, dt_expr, where_sql)
    m1, m2, m3 = st.columns(3)
    m1.metric("Registros após filtros", f"{metrics.get('n', 0):,}".replace(",", "."))
    if metrics.get("dt_min") is not None and not pd.isna(metrics.get("dt_min")):
        m2.metric("Data mínima", str(pd.to_datetime(metrics["dt_min"]).date()))
        m3.metric("Data máxima", str(pd.to_datetime(metrics["dt_max"]).date()))
    elif dt_expr:
        m2.metric("Data mínima", "sem registros válidos")
        m3.metric("Data máxima", "sem registros válidos")
    else:
        m2.metric("Coluna de data", "não configurada")
        m3.metric("Arquivos", str(len(paths)))

    with st.expander("Prévia dos dados", expanded=False):
        st.dataframe(preview_df(paths, 10), use_container_width=True)
        st.dataframe(schema, use_container_width=True)

    overview_tab, temporal_tab, demo_tab, category_tab, quality_tab = st.tabs(
        ["Resumo", "Temporal", "Demografia", "Categorias", "Qualidade"]
    )

    with overview_tab:
        st.markdown(
            "**Gráficos mais úteis para análise epidemiológica neste banco**"
        )
        recs = {
            "SINAN": [
                "Série temporal por mês ou semana",
                "Heatmap ano × mês para sazonalidade",
                "Pirâmide etária por sexo",
                "Distribuição de desfechos / classificação final",
                "Top municípios, quando o parquet não está municipalizado",
                "Completude das variáveis-chave",
            ],
            "SIM": [
                "Série temporal de óbitos",
                "Heatmap ano × mês",
                "Pirâmide etária por sexo",
                "Top causas básicas (CAUSABAS)",
                "Distribuição do local/condição do óbito",
                "Completude dos campos essenciais",
            ],
            "CIHA": [
                "Série temporal de internações/atendimentos",
                "Heatmap ano × mês",
                "Distribuição etária por sexo",
                "Top diagnósticos / procedimentos",
                "Distribuição de desfecho de saída, se disponível",
                "Completude dos campos essenciais",
            ],
        }
        for item in recs[source]:
            st.write(f"• {item}")

    with temporal_tab:
        if not dt_expr:
            st.warning("Selecione uma coluna de data para habilitar os gráficos temporais.")
        else:
            freq_label = st.selectbox(
                "Agregação temporal",
                options=["Ano", "Mês", "Semana"],
                index=1,
                key=f"freq_{source}",
            )
            freq_sql = {"Ano": "year", "Mês": "month", "Semana": "week"}[freq_label]
            ts = query_time_series(paths, dt_expr, where_sql, freq_sql)
            if not ts.empty:
                fig = px.line(
                    ts,
                    x="periodo",
                    y="n",
                    markers=True,
                    title=cfg["time_title"],
                    labels={"periodo": "Período", "n": "Registros"},
                )
                st.plotly_chart(fig, use_container_width=True)
                add_download_button(ts, f"{source.lower()}_serie_temporal.csv", "Baixar série temporal (CSV)")
            else:
                st.info("Não houve registros válidos para montar a série temporal com os filtros atuais.")

            heat = query_month_heatmap(paths, dt_expr, where_sql)
            if not heat.empty:
                pivot = heat.pivot(index="ano", columns="mes", values="n").fillna(0)
                pivot = pivot.reindex(sorted(pivot.index))
                pivot = pivot.reindex(columns=list(range(1, 13)), fill_value=0)
                month_labels = [
                    "Jan",
                    "Fev",
                    "Mar",
                    "Abr",
                    "Mai",
                    "Jun",
                    "Jul",
                    "Ago",
                    "Set",
                    "Out",
                    "Nov",
                    "Dez",
                ]
                fig = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=month_labels,
                        y=pivot.index.astype(str),
                        hovertemplate="Ano %{y}<br>Mês %{x}<br>Registros %{z}<extra></extra>",
                    )
                )
                fig.update_layout(title="Sazonalidade (ano × mês)", xaxis_title="Mês", yaxis_title="Ano")
                st.plotly_chart(fig, use_container_width=True)
                add_download_button(heat, f"{source.lower()}_heatmap_ano_mes.csv", "Baixar dados do heatmap (CSV)")

    with demo_tab:
        if not age_sql:
            st.warning("Selecione uma coluna de idade para habilitar os gráficos demográficos.")
        else:
            age_df = query_age_distribution(paths, age_sql, where_sql)
            if not age_df.empty:
                age_df["faixa"] = age_df["faixa_ini"].astype(int).astype(str) + "–" + (age_df["faixa_ini"].astype(int) + 4).astype(str)
                fig = px.bar(
                    age_df,
                    x="faixa",
                    y="n",
                    title="Distribuição etária (faixas de 5 anos)",
                    labels={"faixa": "Faixa etária", "n": "Registros"},
                )
                st.plotly_chart(fig, use_container_width=True)
                add_download_button(age_df, f"{source.lower()}_faixa_etaria.csv", "Baixar distribuição etária (CSV)")
            else:
                st.info("Não foi possível construir a distribuição etária com a coluna selecionada.")

            if sex_sql:
                pyramid = query_pyramid(paths, age_sql, sex_sql, where_sql)
                if not pyramid.empty:
                    pyramid["faixa"] = pyramid["faixa_ini"].astype(int).astype(str) + "–" + (pyramid["faixa_ini"].astype(int) + 4).astype(str)
                    pyramid["valor"] = np.where(
                        pyramid["sexo"].eq("Masculino"),
                        -pyramid["n"],
                        pyramid["n"],
                    )
                    fig = px.bar(
                        pyramid,
                        x="valor",
                        y="faixa",
                        color="sexo",
                        orientation="h",
                        title="Pirâmide etária por sexo",
                        labels={"valor": "Registros", "faixa": "Faixa etária", "sexo": "Sexo"},
                    )
                    fig.update_layout(barmode="relative")
                    st.plotly_chart(fig, use_container_width=True)
                    add_download_button(pyramid, f"{source.lower()}_piramide_etaria.csv", "Baixar pirâmide etária (CSV)")

    with category_tab:
        c1, c2, c3 = st.columns(3)
        if diagnosis_sql:
            with c1:
                diag_df = query_top_category(paths, diagnosis_sql, where_sql, top_n=20)
                if not diag_df.empty:
                    fig = px.bar(
                        diag_df,
                        x="n",
                        y="categoria",
                        orientation="h",
                        title="Top diagnósticos / CID / agravos",
                        labels={"categoria": "Categoria", "n": "Registros"},
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                    add_download_button(diag_df, f"{source.lower()}_top_diagnosticos.csv", "Baixar top diagnósticos (CSV)")
        if outcome_sql:
            with c2:
                out_df = query_top_category(paths, outcome_sql, where_sql, top_n=20)
                if not out_df.empty:
                    fig = px.bar(
                        out_df,
                        x="n",
                        y="categoria",
                        orientation="h",
                        title="Top desfechos / classificações",
                        labels={"categoria": "Categoria", "n": "Registros"},
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                    add_download_button(out_df, f"{source.lower()}_top_desfechos.csv", "Baixar top desfechos (CSV)")
        if municipality_sql:
            with c3:
                muni_df = query_top_category(paths, municipality_sql, where_sql, top_n=20)
                if not muni_df.empty:
                    fig = px.bar(
                        muni_df,
                        x="n",
                        y="categoria",
                        orientation="h",
                        title="Top municípios",
                        labels={"categoria": "Município", "n": "Registros"},
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                    add_download_button(muni_df, f"{source.lower()}_top_municipios.csv", "Baixar top municípios (CSV)")

    with quality_tab:
        miss = query_missingness(paths, selections, dt_expr, where_sql)
        if miss.empty:
            st.info("Não há colunas-chave configuradas para avaliar completude.")
        else:
            fig = px.bar(
                miss,
                x="campo",
                y="pct_faltante",
                text="pct_faltante",
                title="Completude dos campos-chave",
                labels={"campo": "Campo", "pct_faltante": "% faltante"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss, use_container_width=True)
            add_download_button(miss, f"{source.lower()}_completude.csv", "Baixar completude (CSV)")

    return {
        "source": source,
        "paths": paths,
        "dt_expr": dt_expr,
        "where_sql": where_sql,
    }


# ---------- Comparação entre bases ----------

def render_comparison_tab(loaded_sources: Sequence[Dict[str, object]]) -> None:
    st.subheader("Comparação exploratória entre bancos")
    available = [x for x in loaded_sources if x and x.get("dt_expr")]
    if len(available) < 2:
        st.info("Carregue pelo menos duas bases com coluna de data configurada para comparar as séries temporais.")
        return

    labels = [x["source"] for x in available]
    chosen = st.multiselect(
        "Bases para comparar",
        options=labels,
        default=labels,
        key="comparison_sources",
    )
    freq_label = st.selectbox(
        "Agregação temporal da comparação",
        options=["Ano", "Mês", "Semana"],
        index=1,
        key="comparison_freq",
    )
    normalize = st.checkbox(
        "Normalizar em índice (100 no primeiro período) para comparar tendências",
        value=False,
        key="comparison_normalize",
    )
    freq_sql = {"Ano": "year", "Mês": "month", "Semana": "week"}[freq_label]

    frames = []
    for item in available:
        if item["source"] not in chosen:
            continue
        ts = query_time_series(item["paths"], item["dt_expr"], item["where_sql"], freq_sql)
        if ts.empty:
            continue
        ts = ts.rename(columns={"n": "valor"})
        ts["base"] = item["source"]
        if normalize and not ts.empty:
            first_valid = ts.loc[ts["valor"].notna() & ts["valor"].ne(0), "valor"]
            if not first_valid.empty:
                base0 = first_valid.iloc[0]
                ts["valor"] = ts["valor"] / base0 * 100
        frames.append(ts)

    if not frames:
        st.warning("Nenhuma série temporal válida foi gerada para as bases escolhidas.")
        return

    comp = pd.concat(frames, ignore_index=True)
    ylabel = "Índice (base = 100)" if normalize else "Registros"
    fig = px.line(
        comp,
        x="periodo",
        y="valor",
        color="base",
        markers=True,
        title="Comparação de séries temporais entre bancos",
        labels={"periodo": "Período", "valor": ylabel, "base": "Base"},
    )
    st.plotly_chart(fig, use_container_width=True)
    add_download_button(comp, "comparacao_bases_serie_temporal.csv", "Baixar comparação (CSV)")

    st.caption(
        "Leitura recomendada: SINAN tende a refletir notificações/casos, SIM reflete óbitos e CIHA reflete internações/atendimentos. "
        "Comparações diretas são mais úteis quando o mesmo agravo, o mesmo território e a mesma janela temporal foram aplicados às três bases."
    )


# ---------- Página principal ----------

def main() -> None:
    st.title("Painel Streamlit para análise epidemiológica de Parquets do SINAN, SIM e CIHA")
    st.markdown(
        "Carregue seus Parquets anuais ou uma pasta local com vários arquivos, ajuste as colunas principais e gere gráficos epidemiológicos interativos. "
        "O app foi pensado para lidar com layouts variados do DATASUS e privilegiar gráficos que costumam responder melhor a perguntas de vigilância em saúde."
    )
    st.info(
        "Dica: para bases maiores, prefira o modo de pasta/glob local. O app usa DuckDB para agregar os Parquets sem precisar materializar tudo em memória a cada gráfico."
    )

    sinan_tab, sim_tab, ciha_tab, comp_tab = st.tabs(["SINAN", "SIM", "CIHA", "Comparação"])

    loaded = []
    with sinan_tab:
        loaded.append(render_source_tab("SINAN"))
    with sim_tab:
        loaded.append(render_source_tab("SIM"))
    with ciha_tab:
        loaded.append(render_source_tab("CIHA"))
    with comp_tab:
        render_comparison_tab([x for x in loaded if x])

    st.markdown("---")
    st.markdown(
        "**Recomendação epidemiológica prática**: use a série temporal como gráfico principal; o heatmap para sazonalidade; "
        "a pirâmide etária para perfis demográficos; top diagnósticos/desfechos para caracterização clínica; e a completude para avaliar qualidade dos dados."
    )


if __name__ == "__main__":
    main()
