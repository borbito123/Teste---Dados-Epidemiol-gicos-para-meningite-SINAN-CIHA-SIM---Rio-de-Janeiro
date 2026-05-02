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
    page_title="Painel Epidemiológico — Meningite (SINAN, SIM e CIHA)",
    page_icon="🧠",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Configurações orientadas pelos dicionários operacionais enviados
# -----------------------------------------------------------------------------

CID_MENINGITE_RULES = [
    {
        "grupo": "A170",
        "prefixo": "A170",
        "rotulo": "A170 — Meningite tuberculosa",
        "padrao": "A170",
    },
    {
        "grupo": "A390",
        "prefixo": "A390",
        "rotulo": "A390 — Meningite meningocócica",
        "padrao": "A390",
    },
    {
        "grupo": "A87",
        "prefixo": "A87",
        "rotulo": "A87 — Meningite viral",
        "padrao": "A87*",
    },
    {
        "grupo": "G00",
        "prefixo": "G00",
        "rotulo": "G00 — Meningite bacteriana",
        "padrao": "G00*",
    },
    {
        "grupo": "G01",
        "prefixo": "G01",
        "rotulo": "G01 — Meningite em doenças bacterianas classificadas em outra parte",
        "padrao": "G01*",
    },
    {
        "grupo": "G02",
        "prefixo": "G02",
        "rotulo": "G02 — Meningite em outras doenças infecciosas/parasitárias classificadas em outra parte",
        "padrao": "G02*",
    },
    {
        "grupo": "G03",
        "prefixo": "G03",
        "rotulo": "G03 — Meningite por outras causas / não especificada",
        "padrao": "G03*",
    },
]

# Expressão regular usada para localizar qualquer CID de meningite nos campos
# escolhidos. Ela captura códigos com ou sem pontuação/símbolos comuns em SIM
# (ex.: *G039, A419/G009, G03.9). O valor retornado é normalizado sem pontuação.
CID_MENINGITE_REGEX = r"(A17[\.]?0|A39[\.]?0|A87[\.]?[0-9A-Z]?|G00[\.]?[0-9A-Z]?|G01[\.]?[0-9A-Z]?|G02[\.]?[0-9A-Z]?|G03[\.]?[0-9A-Z]?)"


SOURCE_CONFIG = {
    "SINAN": {
        "subtitle": "Notificações/casos de meningite",
        "dictionary_scope": "RJ, 2007–2025; 35.417 registros e 144 variáveis no dicionário operacional.",
        "date_candidates": [
            "DT_SIN_PRI",
            "DT_NOTIFIC",
            "DT_INVEST",
            "DT_ENCERRA",
            "DT_DIGITA",
        ],
        "sex_candidates": ["CS_SEXO", "SEXO"],
        "age_candidates": ["NU_IDADE_N", "IDADE", "IDADE_ANOS", "IDADEANOS"],
        "age_unit_candidates": [],
        "municipality_candidates": [
            "ID_MN_RESI",
            "ID_MUNICIP",
            "ID_MN_OCORR",
            "CODMUNRES",
            "CODMUNOCOR",
            "MUNIC_RES",
            "MUN_RES",
            "MUN_OCOR",
        ],
        "outcome_candidates": ["EVOLUCAO", "CLASSI_FIN", "CRITERIO"],
        "diagnosis_candidates": [
            "ID_AGRAVO",
            "CON_DIAGES",
            "CLA_ME_BAC",
            "CLA_ME_ASS",
            "CLA_ME_ETI",
            "CLASSI_FIN",
            "CRITERIO",
            "AGRAVO",
            "CID10",
            "CID",
        ],
        "cid_priority_candidates": ["ID_AGRAVO", "CID10", "CID", "AGRAVO"],
        "time_title": "Série temporal de notificações/casos",
        "notes": [
            "No dicionário operacional, ID_AGRAVO aparece como G039 em todos os registros do recorte, portanto a classificação CID tende a cair em G03 para o SINAN.",
            "Para detalhamento etiológico no SINAN, consulte também CON_DIAGES, CLA_ME_BAC, CLA_ME_ASS, CLA_ME_ETI, CRITERIO e CLASSI_FIN.",
        ],
    },
    "SIM": {
        "subtitle": "Óbitos por meningite",
        "dictionary_scope": "RJ, 2007–2024; 2.953 registros e 112 variáveis no dicionário operacional.",
        "date_candidates": ["DTOBITO", "DT_OBITO", "DTNASC", "DT_NASC", "DTATESTADO"],
        "sex_candidates": ["SEXO", "CS_SEXO"],
        "age_candidates": ["IDADE", "IDADEANOS", "IDADE_ANOS"],
        "age_unit_candidates": [],
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
            "LINHAII",
            "ATESTADO",
            "CB_PRE",
        ],
        "cid_priority_candidates": [
            "CAUSABAS",
            "CAUSABAS_O",
            "LINHAA",
            "LINHAB",
            "LINHAC",
            "LINHAD",
            "LINHAII",
            "ATESTADO",
            "CB_PRE",
        ],
        "time_title": "Série temporal de óbitos",
        "notes": [
            "CAUSABAS e CAUSABAS_O são os campos principais para a causa básica, mas as linhas A–D e II podem registrar outros CIDs de meningite associados.",
            "O app procura o primeiro CID compatível na ordem dos campos selecionados, respeitando a prioridade configurada pelo usuário.",
        ],
    },
    "CIHA": {
        "subtitle": "Internações / atendimentos por meningite",
        "dictionary_scope": "RJ, 2011–2024; 2.180 registros e 44 variáveis no dicionário operacional.",
        "date_candidates": [
            "DT_ATEND",
            "DT_SAIDA",
            "DT_INTER",
            "DT_INTERNA",
            "DT_COMPET",
            "COMPET",
            "ANO_CMPT",
        ],
        "sex_candidates": ["SEXO", "CS_SEXO"],
        "age_candidates": ["IDADE", "IDADE_ANOS", "IDADEANOS", "NU_IDADE_N"],
        "age_unit_candidates": ["COD_IDADE"],
        "municipality_candidates": [
            "MUNIC_RES",
            "MUNIC_MOV",
            "CODMUNRES",
            "CODMUN",
            "MUN_RES",
            "ID_MN_RESI",
        ],
        "outcome_candidates": ["MODALIDADE", "COBRANCA", "MOTSAI", "MOT_SAIDA", "TIPO_ALTA", "DESFECHO"],
        "diagnosis_candidates": [
            "DIAG_PRINC",
            "DIAG_SECUN",
            "CIDPRI",
            "CID_PRINC",
            "CID",
            "DIAG",
            "PROC_REA",
            "PROC_PRINC",
        ],
        "cid_priority_candidates": ["DIAG_PRINC", "DIAG_SECUN", "CIDPRI", "CID_PRINC", "CID", "DIAG"],
        "time_title": "Série temporal de internações/atendimentos",
        "notes": [
            "DIAG_PRINC é o principal campo CID-10 observado no dicionário operacional; DIAG_SECUN tem baixa completude, mas pode capturar casos secundários.",
            "Para idade no CIHA, o dicionário indica uso conjunto de IDADE e COD_IDADE; o app já prioriza essa conversão quando ambos estão presentes.",
        ],
    },
}


OPERATIONAL_FIELD_GUIDE = {
    "SINAN": [
        {"campo": "DT_SIN_PRI", "uso": "data principal", "leitura": "Início dos primeiros sintomas", "observação": "100% preenchido no dicionário"},
        {"campo": "DT_NOTIFIC", "uso": "data alternativa", "leitura": "Data de notificação", "observação": "100% preenchido no dicionário"},
        {"campo": "NU_IDADE_N", "uso": "idade", "leitura": "Idade codificada DATASUS", "observação": "converter para anos"},
        {"campo": "CS_SEXO", "uso": "sexo", "leitura": "Sexo do paciente", "observação": "M/F/I"},
        {"campo": "ID_MN_RESI", "uso": "território", "leitura": "Município de residência", "observação": "código IBGE"},
        {"campo": "ID_AGRAVO", "uso": "CID/agravo", "leitura": "Código do agravo", "observação": "G039 constante no recorte"},
        {"campo": "CLASSI_FIN", "uso": "classificação", "leitura": "Classificação final", "observação": "variável-chave de encerramento"},
        {"campo": "CON_DIAGES", "uso": "classificação", "leitura": "Conclusão diagnóstica específica", "observação": "detalha conclusão quando preenchido"},
        {"campo": "EVOLUCAO", "uso": "desfecho", "leitura": "Evolução clínica", "observação": "alta utilidade epidemiológica"},
    ],
    "SIM": [
        {"campo": "DTOBITO", "uso": "data principal", "leitura": "Data do óbito", "observação": "100% preenchido no dicionário"},
        {"campo": "IDADE", "uso": "idade", "leitura": "Idade codificada SIM", "observação": "converter para anos"},
        {"campo": "SEXO", "uso": "sexo", "leitura": "Sexo da pessoa falecida", "observação": "1/2"},
        {"campo": "CODMUNRES", "uso": "território", "leitura": "Município de residência", "observação": "código IBGE"},
        {"campo": "CODMUNOCOR", "uso": "território", "leitura": "Município de ocorrência", "observação": "código IBGE"},
        {"campo": "CAUSABAS", "uso": "CID/agravo", "leitura": "Causa básica do óbito", "observação": "campo CID-10 principal"},
        {"campo": "CAUSABAS_O", "uso": "CID/agravo", "leitura": "Causa básica complementar/original", "observação": "útil para recorte e comparação"},
        {"campo": "LINHAA–LINHAII", "uso": "CID/agravo", "leitura": "Causas nas linhas da DO", "observação": "captura menções associadas"},
        {"campo": "LOCOCOR", "uso": "categoria", "leitura": "Local de ocorrência", "observação": "proxy categórico do evento"},
    ],
    "CIHA": [
        {"campo": "DT_ATEND", "uso": "data principal", "leitura": "Data de atendimento", "observação": "AAAAMMDD; 100% preenchido"},
        {"campo": "DT_SAIDA", "uso": "data alternativa", "leitura": "Data de saída", "observação": "AAAAMMDD; 100% preenchido"},
        {"campo": "IDADE + COD_IDADE", "uso": "idade", "leitura": "Idade e unidade da idade", "observação": "usar conversão conjunta"},
        {"campo": "SEXO", "uso": "sexo", "leitura": "Sexo", "observação": "1/3 no dicionário"},
        {"campo": "MUNIC_RES", "uso": "território", "leitura": "Município de residência", "observação": "código IBGE"},
        {"campo": "MUNIC_MOV", "uso": "território", "leitura": "Município do movimento/atendimento", "observação": "código IBGE"},
        {"campo": "DIAG_PRINC", "uso": "CID/agravo", "leitura": "Diagnóstico principal", "observação": "campo CID-10 principal"},
        {"campo": "DIAG_SECUN", "uso": "CID/agravo", "leitura": "Diagnóstico secundário", "observação": "baixa completude; usar como complementar"},
        {"campo": "MODALIDADE", "uso": "categoria", "leitura": "Modalidade do registro", "observação": "categoria administrativa"},
    ],
}


@dataclass
class ColumnSelection:
    date_col: Optional[str]
    sex_col: Optional[str]
    age_col: Optional[str]
    age_unit_col: Optional[str]
    municipality_col: Optional[str]
    outcome_col: Optional[str]
    diagnosis_col: Optional[str]
    cid_cols: List[str]
    age_mode: str


# -----------------------------------------------------------------------------
# Helpers de texto / SQL
# -----------------------------------------------------------------------------

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
    return "'" + str(value).replace("'", "''") + "'"


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


def choose_candidates(columns: Sequence[str], candidates: Sequence[str], max_items: int = 12) -> List[str]:
    found: List[str] = []
    norm_map = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        match = norm_map.get(normalize_name(candidate))
        if match and match not in found:
            found.append(match)
    if len(found) >= max_items:
        return found[:max_items]

    candidate_norms = [normalize_name(c) for c in candidates]
    for col in columns:
        if col in found:
            continue
        ncol = normalize_name(col)
        if any(cand in ncol or ncol in cand for cand in candidate_norms):
            found.append(col)
        if len(found) >= max_items:
            break
    return found


def detect_columns(source: str, columns: Sequence[str]) -> ColumnSelection:
    cfg = SOURCE_CONFIG[source]
    age_col = choose_candidate(columns, cfg["age_candidates"])
    age_unit_col = choose_candidate(columns, cfg.get("age_unit_candidates", []))
    cid_cols = choose_candidates(columns, cfg["cid_priority_candidates"], max_items=10)
    age_mode = "Automático"
    if age_col and age_unit_col:
        age_mode = "DATASUS com coluna de unidade"
    elif age_col and normalize_name(age_col) in {"NUIDADEN", "IDADE"}:
        age_mode = "DATASUS codificada"
    return ColumnSelection(
        date_col=choose_candidate(columns, cfg["date_candidates"]),
        sex_col=choose_candidate(columns, cfg["sex_candidates"]),
        age_col=age_col,
        age_unit_col=age_unit_col,
        municipality_col=choose_candidate(columns, cfg["municipality_candidates"]),
        outcome_col=choose_candidate(columns, cfg["outcome_candidates"]),
        diagnosis_col=choose_candidate(columns, cfg["diagnosis_candidates"]),
        cid_cols=cid_cols,
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
            CASE WHEN regexp_matches({txt}, '^\\d{{6}}$') AND SUBSTR({txt}, 1, 4) BETWEEN '1900' AND '2099'
                 THEN CAST(try_strptime({txt} || '01', '%Y%m%d') AS DATE) END,
            CASE WHEN regexp_matches({txt}, '^\\d{{4}}$') AND {txt} BETWEEN '1900' AND '2099'
                 THEN CAST(try_strptime({txt} || '0101', '%Y%m%d') AS DATE) END,
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
        WHEN regexp_matches({txt}, '^\\d{{3,4}}$') AND SUBSTR({txt}, 1, 1) IN ('0', '1', '2', '3', '4', '5') THEN
            CASE SUBSTR({txt}, 1, 1)
                WHEN '0' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / (365.25 * 24)
                WHEN '1' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / (365.25 * 24)
                WHEN '2' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / 365.25
                WHEN '3' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / 12
                WHEN '4' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE)
                WHEN '5' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE)
                ELSE NULL
            END
        WHEN regexp_matches({txt}, '^\\d{{1,3}}$') AND TRY_CAST({txt} AS DOUBLE) BETWEEN 0 AND 120
            THEN TRY_CAST({txt} AS DOUBLE)
        ELSE NULL
    END
    """


def datasus_age_with_unit_expr(age_col: str, unit_col: str) -> str:
    age_txt = clean_str_expr(age_col)
    unit_txt = clean_str_expr(unit_col)
    age_num = f"TRY_CAST(REPLACE({age_txt}, ',', '.') AS DOUBLE)"
    return f"""
    CASE
        WHEN {age_txt} IS NULL THEN NULL
        WHEN {age_num} IS NULL THEN NULL
        WHEN {unit_txt} IN ('0', '1') THEN {age_num} / (365.25 * 24)
        WHEN {unit_txt} = '2' THEN {age_num} / 365.25
        WHEN {unit_txt} = '3' THEN {age_num} / 12
        WHEN {unit_txt} IN ('4', '5') THEN {age_num}
        ELSE {age_num}
    END
    """


def direct_age_expr(col: str) -> str:
    txt = clean_str_expr(col)
    return f"TRY_CAST(REPLACE({txt}, ',', '.') AS DOUBLE)"


def build_age_expr(col: Optional[str], age_mode: str, age_unit_col: Optional[str] = None) -> Optional[str]:
    if not col:
        return None
    if age_unit_col and age_mode in {"Automático", "DATASUS com coluna de unidade"}:
        return datasus_age_with_unit_expr(col, age_unit_col)
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
        WHEN '3' THEN 'Feminino'
        WHEN 'I' THEN 'Ignorado/Outro'
        WHEN '0' THEN 'Ignorado/Outro'
        WHEN '9' THEN 'Ignorado/Outro'
        WHEN 'IGN' THEN 'Ignorado/Outro'
        ELSE COALESCE({txt}, 'Ignorado/Outro')
    END
    """


def cid_extract_expr_for_col(col: str) -> str:
    txt = clean_str_expr(col)
    raw = f"regexp_extract(COALESCE(UPPER({txt}), ''), '{CID_MENINGITE_REGEX}', 1)"
    # Remove ponto de CID caso o dado venha como G03.9.
    return f"NULLIF(regexp_replace({raw}, '\\.', '', 'g'), '')"


def cid_extract_expr(cols: Sequence[str]) -> Optional[str]:
    exprs = [cid_extract_expr_for_col(col) for col in cols if col]
    if not exprs:
        return None
    if len(exprs) == 1:
        return exprs[0]
    return "COALESCE(" + ", ".join(exprs) + ")"


def cid_source_expr(cols: Sequence[str]) -> Optional[str]:
    tests = []
    for col in cols:
        expr = cid_extract_expr_for_col(col)
        tests.append(f"WHEN {expr} IS NOT NULL THEN {qstring(col)}")
    if not tests:
        return None
    return "CASE " + " ".join(tests) + " ELSE NULL END"


def cid_group_from_cid_expr(cid_sql: str) -> str:
    clauses = [
        f"WHEN {cid_sql} LIKE {qstring(rule['prefixo'] + '%')} THEN {qstring(rule['grupo'])}"
        for rule in CID_MENINGITE_RULES
    ]
    return "CASE WHEN {cid} IS NULL THEN 'Sem CID-10 meningite detectado' {clauses} ELSE 'Outro CID capturado' END".format(
        cid=cid_sql,
        clauses=" ".join(clauses),
    )


def agravo_type_from_cid_expr(cid_sql: str) -> str:
    clauses = [
        f"WHEN {cid_sql} LIKE {qstring(rule['prefixo'] + '%')} THEN {qstring(rule['rotulo'])}"
        for rule in CID_MENINGITE_RULES
    ]
    return "CASE WHEN {cid} IS NULL THEN 'Sem CID-10 meningite detectado' {clauses} ELSE 'Outro CID capturado' END".format(
        cid=cid_sql,
        clauses=" ".join(clauses),
    )


def pretty_label(col: Optional[str]) -> str:
    return col if col else "(não selecionado)"


def safe_alias(text: str) -> str:
    alias = normalize_name(text).lower()
    if not alias:
        return "campo"
    if alias[0].isdigit():
        alias = "c_" + alias
    return alias[:50]


# -----------------------------------------------------------------------------
# Leitura dos arquivos
# -----------------------------------------------------------------------------

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


def dataset_metrics(paths: Sequence[str], dt_expr: Optional[str], where_sql: str = "") -> Dict[str, object]:
    ref = parquet_ref(paths)
    metrics: Dict[str, object] = {}
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
    return run_query(paths, f"SELECT * FROM {ref} LIMIT {int(limit)}")


# -----------------------------------------------------------------------------
# Filtros
# -----------------------------------------------------------------------------

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
    LIMIT {int(limit)}
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
    agravo_type_expression: Optional[str],
    selected_agravo_types: Sequence[str],
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
    if agravo_type_expression and selected_agravo_types:
        agravo_list = ", ".join(qstring(x) for x in selected_agravo_types)
        clauses.append(f"{agravo_type_expression} IN ({agravo_list})")
    if not clauses:
        return ""
    return "WHERE " + " AND ".join(clauses)


# -----------------------------------------------------------------------------
# Consultas / Gráficos
# -----------------------------------------------------------------------------

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


def query_time_series_by_category(
    paths: Sequence[str],
    dt_expr: str,
    category_expr: str,
    where_sql: str,
    freq_sql: str,
) -> pd.DataFrame:
    ref = parquet_ref(paths)
    sql = f"""
    WITH base AS (
        SELECT {dt_expr} AS dt, {category_expr} AS categoria
        FROM {ref}
        {where_sql}
    )
    SELECT date_trunc('{freq_sql}', dt) AS periodo, categoria, COUNT(*) AS n
    FROM base
    WHERE dt IS NOT NULL AND categoria IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 2
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
    if where_sql:
        sql = f"""
        SELECT {cat_expr} AS categoria, COUNT(*) AS n
        FROM {ref}
        {where_sql}
        AND {cat_expr} IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC, 1
        LIMIT {int(top_n)}
        """
    else:
        sql = f"""
        SELECT {cat_expr} AS categoria, COUNT(*) AS n
        FROM {ref}
        WHERE {cat_expr} IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC, 1
        LIMIT {int(top_n)}
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


def query_agravo_distribution(paths: Sequence[str], cid_sql: str, where_sql: str) -> pd.DataFrame:
    ref = parquet_ref(paths)
    grupo_expr = cid_group_from_cid_expr("cid")
    tipo_expr = agravo_type_from_cid_expr("cid")
    sql = f"""
    WITH base AS (
        SELECT {cid_sql} AS cid
        FROM {ref}
        {where_sql}
    )
    SELECT
        {grupo_expr} AS grupo_cid,
        {tipo_expr} AS tipo_agravo_cid10,
        COUNT(*) AS n,
        COUNT(DISTINCT cid) AS cids_distintos_n,
        string_agg(DISTINCT cid, ', ') FILTER (WHERE cid IS NOT NULL) AS cids_encontrados
    FROM base
    GROUP BY 1, 2
    ORDER BY n DESC, grupo_cid
    """
    df = run_query(paths, sql)
    total = df["n"].sum() if not df.empty else 0
    if total:
        df["pct"] = (df["n"] / total * 100).round(2)
    else:
        df["pct"] = 0.0
    return df


def query_cid_values(paths: Sequence[str], cid_sql: str, where_sql: str, top_n: int = 30) -> pd.DataFrame:
    ref = parquet_ref(paths)
    tipo_expr = agravo_type_from_cid_expr("cid")
    sql = f"""
    WITH base AS (
        SELECT {cid_sql} AS cid
        FROM {ref}
        {where_sql}
    )
    SELECT cid AS cid_detectado, {tipo_expr} AS tipo_agravo_cid10, COUNT(*) AS n
    FROM base
    WHERE cid IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 3 DESC, 1
    LIMIT {int(top_n)}
    """
    df = run_query(paths, sql)
    total = df["n"].sum() if not df.empty else 0
    if total:
        df["pct_entre_cids_detectados"] = (df["n"] / total * 100).round(2)
    return df


def query_cid_source(paths: Sequence[str], cid_source_sql: str, where_sql: str, top_n: int = 20) -> pd.DataFrame:
    ref = parquet_ref(paths)
    sql = f"""
    SELECT {cid_source_sql} AS coluna_origem_cid, COUNT(*) AS n
    FROM {ref}
    {where_sql}
    {('AND' if where_sql else 'WHERE')} {cid_source_sql} IS NOT NULL
    GROUP BY 1
    ORDER BY 2 DESC, 1
    LIMIT {int(top_n)}
    """
    return run_query(paths, sql)


def query_enriched_preview(
    paths: Sequence[str],
    selections: ColumnSelection,
    dt_expr: Optional[str],
    sex_sql: Optional[str],
    age_sql: Optional[str],
    municipality_sql: Optional[str],
    outcome_sql: Optional[str],
    cid_sql: Optional[str],
    cid_source_sql: Optional[str],
    where_sql: str,
    limit: int = 200,
) -> pd.DataFrame:
    ref = parquet_ref(paths)
    select_items: List[str] = []
    if dt_expr:
        select_items.append(f"{dt_expr} AS data_analise")
    if sex_sql:
        select_items.append(f"{sex_sql} AS sexo_recodificado")
    if age_sql:
        select_items.append(f"ROUND({age_sql}, 2) AS idade_anos")
    if municipality_sql:
        select_items.append(f"{municipality_sql} AS municipio")
    if outcome_sql:
        select_items.append(f"{outcome_sql} AS desfecho_classificacao")
    if cid_sql:
        select_items.append(f"{cid_sql} AS cid_meningite_detectado")
        select_items.append(f"{cid_group_from_cid_expr(cid_sql)} AS grupo_cid")
        select_items.append(f"{agravo_type_from_cid_expr(cid_sql)} AS tipo_agravo_cid10")
    if cid_source_sql:
        select_items.append(f"{cid_source_sql} AS coluna_origem_cid")

    raw_cols: List[str] = []
    for col in [
        selections.date_col,
        selections.sex_col,
        selections.age_col,
        selections.age_unit_col,
        selections.municipality_col,
        selections.outcome_col,
        selections.diagnosis_col,
        *selections.cid_cols,
    ]:
        if col and col not in raw_cols:
            raw_cols.append(col)
    for col in raw_cols:
        select_items.append(f"{qident(col)} AS raw_{safe_alias(col)}")

    if not select_items:
        select_items.append("*")

    sql = f"""
    SELECT {', '.join(select_items)}
    FROM {ref}
    {where_sql}
    LIMIT {int(limit)}
    """
    return run_query(paths, sql)


def query_missingness(
    paths: Sequence[str],
    selections: ColumnSelection,
    dt_expr: Optional[str],
    where_sql: str,
    cid_sql: Optional[str],
) -> pd.DataFrame:
    ref = parquet_ref(paths)
    checks = []
    if selections.date_col and dt_expr:
        checks.append(("Data", f"SUM(CASE WHEN {dt_expr} IS NULL THEN 1 ELSE 0 END)"))
    if selections.sex_col:
        checks.append(("Sexo", f"SUM(CASE WHEN {clean_str_expr(selections.sex_col)} IS NULL THEN 1 ELSE 0 END)"))
    if selections.age_col:
        age_expr_sql = build_age_expr(selections.age_col, selections.age_mode, selections.age_unit_col)
        checks.append(("Idade", f"SUM(CASE WHEN {age_expr_sql} IS NULL THEN 1 ELSE 0 END)"))
    if selections.age_unit_col:
        checks.append(("Unidade da idade", f"SUM(CASE WHEN {clean_str_expr(selections.age_unit_col)} IS NULL THEN 1 ELSE 0 END)"))
    if selections.municipality_col:
        checks.append(
            (
                "Município",
                f"SUM(CASE WHEN {clean_str_expr(selections.municipality_col)} IS NULL THEN 1 ELSE 0 END)",
            )
        )
    if selections.outcome_col:
        checks.append(("Desfecho/classificação", f"SUM(CASE WHEN {clean_str_expr(selections.outcome_col)} IS NULL THEN 1 ELSE 0 END)"))
    if selections.diagnosis_col:
        checks.append(("Diagnóstico/CID bruto", f"SUM(CASE WHEN {clean_str_expr(selections.diagnosis_col)} IS NULL THEN 1 ELSE 0 END)"))
    if cid_sql:
        checks.append(("CID-10 de meningite detectado", f"SUM(CASE WHEN {cid_sql} IS NULL THEN 1 ELSE 0 END)"))
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


# -----------------------------------------------------------------------------
# Renderização por fonte
# -----------------------------------------------------------------------------

def render_field_guide(source: str) -> None:
    cfg = SOURCE_CONFIG[source]
    st.markdown(f"**Escopo do dicionário operacional usado para orientar o app:** {cfg['dictionary_scope']}")
    st.dataframe(pd.DataFrame(OPERATIONAL_FIELD_GUIDE[source]), use_container_width=True, hide_index=True)
    if cfg.get("notes"):
        for note in cfg["notes"]:
            st.caption(f"• {note}")


def render_cid_reference() -> None:
    st.dataframe(
        pd.DataFrame(CID_MENINGITE_RULES)[["grupo", "padrao", "rotulo"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "A extração remove pontuação do CID e procura os códigos A170, A390, A87*, G00*, G01*, G02* e G03* nos campos selecionados. "
        "Quando houver mais de um campo, o primeiro CID encontrado segue a ordem exibida em 'Campos usados para identificar tipo de agravo'."
    )


def render_source_tab(source: str) -> Optional[Dict[str, object]]:
    cfg = SOURCE_CONFIG[source]
    st.subheader(f"{source} — {cfg['subtitle']}")

    with st.expander("Dicionário operacional: campos recomendados", expanded=False):
        render_field_guide(source)

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
        left, middle, right = st.columns(3)
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
        with middle:
            age_unit_col = st.selectbox(
                "Coluna de unidade da idade (opcional)",
                options=[None] + columns,
                index=(columns.index(suggestions.age_unit_col) + 1) if suggestions.age_unit_col in columns else 0,
                format_func=lambda x: "(não usar)" if x is None else x,
                key=f"age_unit_col_{source}",
                help="Útil no CIHA quando IDADE deve ser interpretada junto com COD_IDADE.",
            )
            age_mode_options = ["Automático", "Anos diretos", "DATASUS codificada", "DATASUS com coluna de unidade"]
            age_mode = st.selectbox(
                "Como interpretar a idade?",
                options=age_mode_options,
                index=age_mode_options.index(suggestions.age_mode) if suggestions.age_mode in age_mode_options else 0,
                key=f"age_mode_{source}",
                help=(
                    "Use 'DATASUS codificada' para campos como NU_IDADE_N/IDADE no padrão compacto. "
                    "Use 'DATASUS com coluna de unidade' para IDADE + COD_IDADE no CIHA."
                ),
            )
            municipality_col = st.selectbox(
                "Coluna de município",
                options=[None] + columns,
                index=(columns.index(suggestions.municipality_col) + 1)
                if suggestions.municipality_col in columns
                else 0,
                format_func=lambda x: "(não usar)" if x is None else x,
                key=f"municipality_col_{source}",
            )
        with right:
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
                "Coluna principal de diagnóstico / CID / agravo",
                options=[None] + columns,
                index=(columns.index(suggestions.diagnosis_col) + 1)
                if suggestions.diagnosis_col in columns
                else 0,
                format_func=lambda x: "(não usar)" if x is None else x,
                key=f"diagnosis_col_{source}",
            )

        default_cid_cols = [c for c in suggestions.cid_cols if c in columns]
        cid_cols = st.multiselect(
            "Campos usados para identificar tipo de agravo CID-10",
            options=columns,
            default=default_cid_cols,
            key=f"cid_cols_{source}",
            help=(
                "Selecione e ordene conceitualmente os campos que podem conter CID-10. "
                "O app identifica A170, A390, A87*, G00*, G01*, G02* e G03* e usa o primeiro campo preenchido na ordem apresentada."
            ),
        )

        st.markdown(
            f"**Sugestões detectadas** — data: `{pretty_label(suggestions.date_col)}`, sexo: `{pretty_label(suggestions.sex_col)}`, "
            f"idade: `{pretty_label(suggestions.age_col)}`, unidade da idade: `{pretty_label(suggestions.age_unit_col)}`, "
            f"município: `{pretty_label(suggestions.municipality_col)}`, desfecho: `{pretty_label(suggestions.outcome_col)}`, "
            f"diagnóstico principal: `{pretty_label(suggestions.diagnosis_col)}`."
        )
        if default_cid_cols:
            st.caption("Campos CID sugeridos pelo dicionário/nomes das colunas: " + ", ".join(default_cid_cols))

    selections = ColumnSelection(
        date_col=date_col,
        sex_col=sex_col,
        age_col=age_col,
        age_unit_col=age_unit_col,
        municipality_col=municipality_col,
        outcome_col=outcome_col,
        diagnosis_col=diagnosis_col,
        cid_cols=list(cid_cols),
        age_mode=age_mode,
    )

    dt_expr = date_expr(date_col) if date_col else None
    sex_sql = sex_expr(sex_col) if sex_col else None
    age_sql = build_age_expr(age_col, age_mode, age_unit_col) if age_col else None
    municipality_sql = clean_str_expr(municipality_col) if municipality_col else None
    outcome_sql = clean_str_expr(outcome_col) if outcome_col else None
    diagnosis_sql = clean_str_expr(diagnosis_col) if diagnosis_col else None
    cid_code_sql = cid_extract_expr(cid_cols)
    cid_source_sql = cid_source_expr(cid_cols)
    agravo_type_sql = agravo_type_from_cid_expr(cid_code_sql) if cid_code_sql else None

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
        c1, c2, c3, c4 = st.columns(4)
        selected_sex: List[str] = []
        selected_municipalities: List[str] = []
        selected_outcomes: List[str] = []
        selected_agravo_types: List[str] = []
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
        if agravo_type_sql:
            with c4:
                agravo_options = top_values(paths, agravo_type_sql, limit=12)
                selected_agravo_types = st.multiselect(
                    "Tipo de agravo CID-10",
                    options=agravo_options,
                    default=[],
                    key=f"agravo_filter_{source}",
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
        agravo_type_expression=agravo_type_sql,
        selected_agravo_types=selected_agravo_types,
    )

    metrics = dataset_metrics(paths, dt_expr, where_sql)
    m1, m2, m3, m4 = st.columns(4)
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

    if cid_code_sql:
        try:
            agravo_dist_for_metric = query_agravo_distribution(paths, cid_code_sql, where_sql)
            if not agravo_dist_for_metric.empty:
                top_row = agravo_dist_for_metric.iloc[0]
                m4.metric("Tipo CID-10 mais frequente", str(top_row["grupo_cid"]), f"{int(top_row['n'])} registros")
            else:
                m4.metric("Tipo CID-10", "sem dados")
        except Exception:
            m4.metric("Tipo CID-10", "erro na classificação")
    else:
        m4.metric("Tipo CID-10", "não configurado")

    with st.expander("Prévia dos dados brutos e schema", expanded=False):
        st.dataframe(preview_df(paths, 10), use_container_width=True)
        st.dataframe(schema, use_container_width=True)

    overview_tab, temporal_tab, agravo_tab, demo_tab, category_tab, quality_tab, dictionary_tab = st.tabs(
        ["Resumo", "Temporal", "Agravo CID-10", "Demografia", "Categorias", "Qualidade", "Dicionário"]
    )

    with overview_tab:
        st.markdown("**Leitura operacional recomendada para este banco**")
        render_field_guide(source)
        st.markdown("**Classificação CID-10 de meningite usada no painel**")
        render_cid_reference()
        if cid_cols:
            st.success("Tipo de agravo CID-10 habilitado com os campos: " + ", ".join(cid_cols))
        else:
            st.warning("Selecione ao menos um campo CID/agravo para habilitar a classificação por tipo de agravo.")

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
            stratify = False
            if agravo_type_sql:
                stratify = st.checkbox(
                    "Estratificar série temporal por tipo de agravo CID-10",
                    value=False,
                    key=f"time_by_agravo_{source}",
                )
            if stratify and agravo_type_sql:
                ts = query_time_series_by_category(paths, dt_expr, agravo_type_sql, where_sql, freq_sql)
                if not ts.empty:
                    fig = px.line(
                        ts,
                        x="periodo",
                        y="n",
                        color="categoria",
                        markers=True,
                        title=f"{cfg['time_title']} por tipo de agravo CID-10",
                        labels={"periodo": "Período", "n": "Registros", "categoria": "Tipo de agravo"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    add_download_button(ts, f"{source.lower()}_serie_temporal_por_agravo.csv", "Baixar série temporal por agravo (CSV)")
                else:
                    st.info("Não houve registros válidos para montar a série temporal estratificada.")
            else:
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

    with agravo_tab:
        if not cid_code_sql:
            st.warning("Selecione campos CID/agravo na configuração para habilitar esta aba.")
        else:
            st.markdown("**Distribuição do tipo de agravo segundo CID-10**")
            agravo_dist = query_agravo_distribution(paths, cid_code_sql, where_sql)
            if not agravo_dist.empty:
                fig = px.bar(
                    agravo_dist,
                    x="n",
                    y="tipo_agravo_cid10",
                    orientation="h",
                    text="pct",
                    title="Tipo de agravo identificado",
                    labels={"tipo_agravo_cid10": "Tipo de agravo CID-10", "n": "Registros", "pct": "%"},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(agravo_dist, use_container_width=True, hide_index=True)
                add_download_button(agravo_dist, f"{source.lower()}_tipo_agravo_cid10.csv", "Baixar tipos de agravo (CSV)")

            cids = query_cid_values(paths, cid_code_sql, where_sql, top_n=30)
            if not cids.empty:
                st.markdown("**CIDs detectados dentro dos grupos monitorados**")
                fig = px.bar(
                    cids,
                    x="n",
                    y="cid_detectado",
                    color="tipo_agravo_cid10",
                    orientation="h",
                    title="Top CIDs detectados",
                    labels={"cid_detectado": "CID detectado", "n": "Registros", "tipo_agravo_cid10": "Tipo de agravo"},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(cids, use_container_width=True, hide_index=True)
                add_download_button(cids, f"{source.lower()}_cids_detectados.csv", "Baixar CIDs detectados (CSV)")

            if cid_source_sql:
                origem = query_cid_source(paths, cid_source_sql, where_sql)
                if not origem.empty:
                    st.markdown("**Campo de origem do CID detectado**")
                    st.dataframe(origem, use_container_width=True, hide_index=True)
                    add_download_button(origem, f"{source.lower()}_origem_cid_detectado.csv", "Baixar origem do CID (CSV)")

            st.markdown("**Prévia enriquecida para leitura dos registros**")
            preview_limit = st.slider(
                "Quantidade de registros na prévia enriquecida",
                min_value=50,
                max_value=5000,
                value=200,
                step=50,
                key=f"enriched_preview_limit_{source}",
            )
            enriched = query_enriched_preview(
                paths,
                selections,
                dt_expr,
                sex_sql,
                age_sql,
                municipality_sql,
                outcome_sql,
                cid_code_sql,
                cid_source_sql,
                where_sql,
                limit=preview_limit,
            )
            st.dataframe(enriched, use_container_width=True)
            add_download_button(enriched, f"{source.lower()}_previa_enriquecida_cid10.csv", "Baixar prévia enriquecida (CSV)")

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
                        title="Top diagnósticos / CID / agravos brutos",
                        labels={"categoria": "Categoria", "n": "Registros"},
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                    add_download_button(diag_df, f"{source.lower()}_top_diagnosticos_brutos.csv", "Baixar top diagnósticos brutos (CSV)")
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
        miss = query_missingness(paths, selections, dt_expr, where_sql, cid_code_sql)
        if miss.empty:
            st.info("Não há colunas-chave configuradas para avaliar completude.")
        else:
            fig = px.bar(
                miss,
                x="campo",
                y="pct_faltante",
                text="pct_faltante",
                title="Completude dos campos-chave após filtros",
                labels={"campo": "Campo", "pct_faltante": "% faltante"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss, use_container_width=True, hide_index=True)
            add_download_button(miss, f"{source.lower()}_completude.csv", "Baixar completude (CSV)")

    with dictionary_tab:
        render_field_guide(source)
        st.markdown("**Referência CID-10 usada pelo app**")
        render_cid_reference()
        st.markdown("**Schema lido dos Parquets carregados**")
        st.dataframe(schema, use_container_width=True, hide_index=True)

    return {
        "source": source,
        "paths": paths,
        "dt_expr": dt_expr,
        "where_sql": where_sql,
        "cid_sql": cid_code_sql,
        "agravo_type_sql": agravo_type_sql,
    }


# -----------------------------------------------------------------------------
# Comparação entre bases
# -----------------------------------------------------------------------------

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
    stratify = st.checkbox(
        "Separar por tipo de agravo CID-10 quando disponível",
        value=False,
        key="comparison_stratify_agravo",
    )
    freq_sql = {"Ano": "year", "Mês": "month", "Semana": "week"}[freq_label]

    frames = []
    for item in available:
        if item["source"] not in chosen:
            continue
        if stratify and item.get("agravo_type_sql"):
            ts = query_time_series_by_category(
                item["paths"], item["dt_expr"], item["agravo_type_sql"], item["where_sql"], freq_sql
            )
            if ts.empty:
                continue
            ts = ts.rename(columns={"n": "valor"})
            ts["base"] = item["source"]
            ts["serie"] = ts["base"] + " — " + ts["categoria"].astype(str)
        else:
            ts = query_time_series(item["paths"], item["dt_expr"], item["where_sql"], freq_sql)
            if ts.empty:
                continue
            ts = ts.rename(columns={"n": "valor"})
            ts["base"] = item["source"]
            ts["serie"] = item["source"]
        if normalize and not ts.empty:
            ts = ts.sort_values("periodo")
            for serie in ts["serie"].unique():
                idx = ts["serie"].eq(serie)
                first_valid = ts.loc[idx & ts["valor"].notna() & ts["valor"].ne(0), "valor"]
                if not first_valid.empty:
                    base0 = first_valid.iloc[0]
                    ts.loc[idx, "valor"] = ts.loc[idx, "valor"] / base0 * 100
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
        color="serie",
        markers=True,
        title="Comparação de séries temporais entre bancos",
        labels={"periodo": "Período", "valor": ylabel, "serie": "Série"},
    )
    st.plotly_chart(fig, use_container_width=True)
    add_download_button(comp, "comparacao_bases_serie_temporal.csv", "Baixar comparação (CSV)")

    if any(item.get("cid_sql") for item in available if item["source"] in chosen):
        st.markdown("**Distribuição comparada do tipo de agravo CID-10**")
        dist_frames = []
        for item in available:
            if item["source"] not in chosen or not item.get("cid_sql"):
                continue
            dist = query_agravo_distribution(item["paths"], item["cid_sql"], item["where_sql"])
            if dist.empty:
                continue
            dist["base"] = item["source"]
            dist_frames.append(dist)
        if dist_frames:
            dist_comp = pd.concat(dist_frames, ignore_index=True)
            fig = px.bar(
                dist_comp,
                x="base",
                y="n",
                color="tipo_agravo_cid10",
                title="Tipo de agravo CID-10 por base",
                labels={"base": "Base", "n": "Registros", "tipo_agravo_cid10": "Tipo de agravo"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dist_comp, use_container_width=True, hide_index=True)
            add_download_button(dist_comp, "comparacao_bases_tipo_agravo_cid10.csv", "Baixar comparação por agravo (CSV)")

    st.caption(
        "Leitura recomendada: SINAN reflete notificações/casos, SIM reflete óbitos e CIHA reflete internações/atendimentos. "
        "Comparações diretas são mais úteis quando o mesmo território e a mesma janela temporal foram aplicados às três bases. "
        "No SINAN, o CID do agravo pode estar constante como G039 no recorte; use as variáveis de classificação do SINAN para análises etiológicas mais finas."
    )


# -----------------------------------------------------------------------------
# Página principal
# -----------------------------------------------------------------------------

def main() -> None:
    st.title("Painel Streamlit para meningite — SINAN, SIM e CIHA")
    st.markdown(
        "Carregue Parquets anuais ou uma pasta local com vários arquivos, ajuste as colunas principais e gere gráficos epidemiológicos interativos. "
        "Esta versão acrescenta a classificação do tipo de agravo segundo grupos CID-10 de meningite e usa os dicionários operacionais para sugerir campos prioritários."
    )
    st.info(
        "Dica: para bases maiores, prefira o modo de pasta/glob local. O app usa DuckDB para agregar os Parquets sem precisar materializar tudo em memória a cada gráfico."
    )

    with st.expander("Como o tipo de agravo CID-10 é identificado", expanded=True):
        render_cid_reference()

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
        "a aba Agravo CID-10 para verificar a composição clínica do recorte; a pirâmide etária para perfis demográficos; "
        "top diagnósticos/desfechos para caracterização complementar; e completude para avaliar qualidade dos dados."
    )


if __name__ == "__main__":
    main()
