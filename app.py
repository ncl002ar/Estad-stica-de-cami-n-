import glob
import inspect
import io
import os
import re
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

# -----------------------------------------------------------------------------
# CONFIGURACION DE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Estadística de Camión | Retail WMS vs TMS",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #e6e9ef;
        padding: 12px 16px;
        border-radius: 10px;
    }
    section[data-testid="stSidebar"] div.block-container {
        padding-top: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# CONEXION CON REPOSITORIO / GOOGLE DRIVE
# -----------------------------------------------------------------------------
def find_repo_excel_files():
    candidates = glob.glob("data/*.xlsx") + glob.glob("*.xlsx")
    amrepa_f = sorted([f for f in candidates if "amrepa" in os.path.basename(f).lower()])
    unigis_f = sorted(
        [
            f
            for f in candidates
            if "seguimiento" in os.path.basename(f).lower()
            or "unigis" in os.path.basename(f).lower()
            or "supply" in os.path.basename(f).lower()
        ]
    )
    if amrepa_f and unigis_f:
        return amrepa_f[-1], unigis_f[-1]
    return None, None


def _extract_folder_id(url: str) -> str:
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url or "")
    return m.group(1) if m else ""


@st.cache_data(ttl=1800, show_spinner="Sincronizando reportes desde Google Drive...")
def fetch_from_drive_folder(folder_url: str):
    if not HAS_GDOWN:
        raise RuntimeError("La librería 'gdown' no está instalada en el entorno.")
    if not folder_url or "drive.google.com" not in folder_url:
        raise ValueError("Ingresá un enlace válido de carpeta compartida de Google Drive.")

    folder_id = _extract_folder_id(folder_url.strip())
    clean_url = (
        f"https://drive.google.com/drive/folders/{folder_id}"
        if folder_id
        else folder_url.strip()
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        sig = inspect.signature(gdown.download_folder)
        kwargs = {
            "url": clean_url,
            "output": tmpdir,
            "quiet": True,
        }
        if "use_cookies" in sig.parameters:
            kwargs["use_cookies"] = False

        gdown.download_folder(**kwargs)

        all_xlsx = glob.glob(os.path.join(tmpdir, "**", "*.xlsx"), recursive=True)
        amrepa_files = sorted(
            [f for f in all_xlsx if "amrepa" in os.path.basename(f).lower()]
        )
        unigis_files = sorted(
            [
                f
                for f in all_xlsx
                if "seguimiento" in os.path.basename(f).lower()
                or "unigis" in os.path.basename(f).lower()
                or "supply" in os.path.basename(f).lower()
            ]
        )

        if not amrepa_files or not unigis_files:
            found_names = [os.path.basename(f) for f in all_xlsx]
            raise FileNotFoundError(
                f"No se encontraron ambos archivos Excel en la carpeta de Drive. "
                f"Detectados: {found_names if found_names else 'ninguno'}."
            )

        with open(amrepa_files[-1], "rb") as fa:
            a_bytes = fa.read()
        with open(unigis_files[-1], "rb") as fu:
            u_bytes = fu.read()

        return (
            a_bytes,
            u_bytes,
            os.path.basename(amrepa_files[-1]),
            os.path.basename(unigis_files[-1]),
        )


# -----------------------------------------------------------------------------
# FUNCIONES DE LIMPIEZA Y PROCESAMIENTO DE DATOS (BACKEND)
# -----------------------------------------------------------------------------
def _clean_int_str(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    return num.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)


@st.cache_data(show_spinner="Procesando y conciliando bases AMREPA y UNIGIS...")
def load_and_process_data(amrepa_bytes: bytes, unigis_bytes: bytes):
    # 1. Lectura de archivos Excel
    df_a_raw = pd.read_excel(io.BytesIO(amrepa_bytes), sheet_name=0)
    xls_u = pd.ExcelFile(io.BytesIO(unigis_bytes))
    sheet_u = (
        "Supply - Reporte Seguimiento De"
        if "Supply - Reporte Seguimiento De" in xls_u.sheet_names
        else xls_u.sheet_names[0]
    )
    df_u_raw = pd.read_excel(io.BytesIO(unigis_bytes), sheet_name=sheet_u)

    # 2. Limpieza AMREPA (WMS)
    df_a = df_a_raw.copy()
    df_a["Operacion"] = pd.to_numeric(df_a["[Codigo almacen]"], errors="coerce").astype("Int64")
    df_a["Contador"] = pd.to_numeric(df_a["[Contador transporte]"], errors="coerce").astype("Int64")
    df_a["Transportista"] = _clean_int_str(df_a["[Codigo de Transportista]"]).fillna("Sin dato")
    df_a["Tienda_Cod"] = _clean_int_str(df_a["[Codigo de tienda]"]).fillna("Sin dato")
    df_a["Fecha_Reparto"] = pd.to_datetime(
        dict(
            year=df_a["[Reparto - Año]"],
            month=df_a["[Reparto - Mes]"],
            day=df_a["[Reparto - Dia]"],
        ),
        errors="coerce",
    )
    df_a["Ref_Clean"] = df_a["Operacion"].astype(str) + " - " + df_a["Contador"].astype(str)

    # 3. Limpieza UNIGIS (TMS)
    df_u = df_u_raw.copy()
    df_u["Operacion"] = pd.to_numeric(df_u["Operacion"], errors="coerce").astype("Int64")
    df_u["Fecha_Reparto"] = pd.to_datetime(df_u["Fecha Reparto"], format="%d/%m/%Y", errors="coerce")

    # Validacion estricta de Ref Externa Viaje: debe ser "Operacion - Numero"
    ref_str = df_u["Ref Externa Viaje"].fillna("").astype(str).str.strip()
    op_str = df_u["Operacion"].astype(str).str.strip()
    parts = ref_str.str.split(" - ")
    is_two_parts = parts.str.len() == 2
    p0 = parts.str[0].fillna("").str.strip()
    p1 = parts.str[1].fillna("").str.strip()
    df_u["ref_valida"] = is_two_parts & (p0 == op_str) & (p1.str.isdigit())
    df_u["Ref_Clean"] = np.where(df_u["ref_valida"], ref_str, np.nan)

    # Correccion de Tienda_Cod (incluye los casos donde Tienda es nulo y el codigo esta en Zona)
    is_store_row = df_u["Tipo de Zona"].isin(["Tienda", "Sin zona"])
    tienda_num = pd.to_numeric(df_u["Tienda"], errors="coerce")
    df_u["Tienda_Cod"] = np.where(
        is_store_row,
        np.where(
            tienda_num.notna(),
            tienda_num.apply(lambda x: str(int(x)) if pd.notna(x) else ""),
            df_u["Zona"].fillna("").astype(str).str.strip(),
        ),
        np.nan,
    )

    # Separar UNIGIS valido vs descartado por Ref Externa invalida
    u_valid = df_u[df_u["ref_valida"]].copy()
    u_invalid = df_u[~df_u["ref_valida"]].copy()

    # Deteccion de Ref Externa duplicada en distintos Id Viaje (conservar ambos como alerta)
    ref_id_counts = u_valid.groupby("Ref_Clean")["Id Viaje"].nunique()
    dup_refs_set = set(ref_id_counts[ref_id_counts > 1].index)
    u_valid["Alerta_Duplicado"] = u_valid["Ref_Clean"].isin(dup_refs_set)

    # 4. Tablas maestras a nivel Viaje (Ref_Clean) para Solapa 1
    a_trips = (
        df_a.groupby(["Operacion", "Ref_Clean"], as_index=False)
        .agg(
            Fecha_Reparto=("Fecha_Reparto", "min"),
            Transportista=("Transportista", lambda s: " / ".join(sorted(s.unique()))),
            Contador=("Contador", "first"),
            Tiendas_WMS=("Tienda_Cod", "nunique"),
            Set_Tiendas_WMS=("Tienda_Cod", lambda s: tuple(sorted(set(s)))),
        )
    )

    u_stores_only = u_valid[u_valid["Tipo de Zona"].isin(["Tienda", "Sin zona"])].copy()
    u_store_agg = (
        u_stores_only.groupby(["Operacion", "Ref_Clean"], as_index=False)
        .agg(
            Tiendas_TMS=("Tienda_Cod", "nunique"),
            Set_Tiendas_TMS=("Tienda_Cod", lambda s: tuple(sorted(set(s)))),
            Tiendas_Visitadas_GPS=("Observacion", lambda s: int(s.isna().sum())),
        )
    )

    u_trips = (
        u_valid.groupby(["Operacion", "Ref_Clean"], as_index=False)
        .agg(
            Fecha_Reparto_TMS=("Fecha_Reparto", "min"),
            Cant_Id_Viaje=("Id Viaje", "nunique"),
            Ids_Viaje=("Id Viaje", lambda s: ", ".join(map(str, sorted(s.unique())))),
            Dominio=("Dominio", lambda s: " / ".join(sorted(s.dropna().astype(str).unique()))),
            Tipo_Camion=("Tipo de Camion", lambda s: " / ".join(sorted(s.dropna().astype(str).unique()))),
            Alerta_Duplicado=("Alerta_Duplicado", "max"),
        )
        .merge(u_store_agg, on=["Operacion", "Ref_Clean"], how="left")
    )
    u_trips["Tiendas_TMS"] = u_trips["Tiendas_TMS"].fillna(0).astype(int)
    u_trips["Tiendas_Visitadas_GPS"] = u_trips["Tiendas_Visitadas_GPS"].fillna(0).astype(int)
    u_trips["Set_Tiendas_TMS"] = u_trips["Set_Tiendas_TMS"].apply(
        lambda x: x if isinstance(x, tuple) else tuple()
    )

    master_trips = pd.merge(
        a_trips,
        u_trips,
        on=["Operacion", "Ref_Clean"],
        how="outer",
        indicator=True,
    )
    master_trips["Estado_Conciliacion"] = master_trips["_merge"].map(
        {
            "both": "Match (WMS + TMS)",
            "left_only": "Solo en AMREPA (Gap)",
            "right_only": "Solo en UNIGIS",
        }
    )
    master_trips["Fecha_Base"] = master_trips["Fecha_Reparto"].fillna(master_trips["Fecha_Reparto_TMS"])
    master_trips["Mes_Label"] = master_trips["Fecha_Base"].dt.strftime("%Y-%m").fillna("Sin fecha")
    master_trips["Semana_Sort"] = master_trips["Fecha_Base"].dt.to_period("W").apply(
        lambda p: p.start_time if pd.notna(p) else pd.NaT
    )
    master_trips["Semana_Label"] = np.where(
        master_trips["Semana_Sort"].notna(),
        "Sem " + master_trips["Semana_Sort"].dt.strftime("%d/%m"),
        "Sin semana",
    )

    master_trips["Transportista"] = master_trips["Transportista"].fillna("Sin dato WMS")
    master_trips["Tiendas_WMS"] = master_trips["Tiendas_WMS"].fillna(0).astype(int)
    master_trips["Tiendas_TMS"] = master_trips["Tiendas_TMS"].fillna(0).astype(int)
    master_trips["Alerta_Duplicado"] = master_trips["Alerta_Duplicado"].fillna(False)
    master_trips["Set_Tiendas_WMS"] = master_trips["Set_Tiendas_WMS"].apply(
        lambda x: x if isinstance(x, tuple) else tuple()
    )
    master_trips["Set_Tiendas_TMS"] = master_trips["Set_Tiendas_TMS"].apply(
        lambda x: x if isinstance(x, tuple) else tuple()
    )

    def calc_store_diffs(row):
        sw = set(row["Set_Tiendas_WMS"])
        st_set = set(row["Set_Tiendas_TMS"])
        en_ambos = len(sw & st_set)
        faltan_tms = sorted(sw - st_set)
        extra_tms = sorted(st_set - sw)
        return pd.Series(
            [
                en_ambos,
                len(faltan_tms),
                len(extra_tms),
                ", ".join(faltan_tms),
                ", ".join(extra_tms),
            ]
        )

    master_trips[
        [
            "Tiendas_Match",
            "Cant_Tiendas_Faltan_TMS",
            "Cant_Tiendas_Extra_TMS",
            "Lista_Faltan_TMS",
            "Lista_Extra_TMS",
        ]
    ] = master_trips.apply(calc_store_diffs, axis=1)

    dup_detail = (
        u_valid[u_valid["Alerta_Duplicado"]]
        .groupby(
            ["Operacion", "Fecha_Reparto", "Hora Reparto", "Ref_Clean", "Id Viaje", "Dominio", "Tipo de Camion"],
            as_index=False,
        )
        .agg(
            Tiendas_Planificadas=("Tipo de Zona", lambda s: int(s.isin(["Tienda", "Sin zona"]).sum())),
            Tiendas_Con_Visita_GPS=(
                "Observacion",
                lambda s: int(
                    (
                        s.isna()
                        & u_valid.loc[s.index, "Tipo de Zona"].isin(["Tienda", "Sin zona"])
                    ).sum()
                ),
            ),
        )
        .sort_values(["Operacion", "Ref_Clean", "Id Viaje"])
    )

    invalid_detail = (
        u_invalid.groupby(
            ["Operacion", "Fecha_Reparto", "Hora Reparto", "Id Viaje", "Dominio", "Tipo de Camion"],
            as_index=False,
        )
        .agg(
            Ref_Externa_Original=(
                "Ref Externa Viaje",
                lambda s: str(s.iloc[0]) if pd.notna(s.iloc[0]) else "(Vacío)",
            ),
            Tiendas_Planificadas=("Tipo de Zona", lambda s: int(s.isin(["Tienda", "Sin zona"]).sum())),
            Observacion_Principal=("Observacion", lambda s: ", ".join(sorted(s.dropna().unique())) or "OK"),
        )
        .sort_values(["Operacion", "Fecha_Reparto", "Id Viaje"])
    )

    return {
        "master_trips": master_trips,
        "dup_detail": dup_detail,
        "invalid_detail": invalid_detail,
    }


def build_audit_excel(df_gap, df_store_diff, df_dup, df_inv) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_gap.to_excel(writer, sheet_name="1_Viajes_Solo_WMS_Gap", index=False)
        df_store_diff.to_excel(writer, sheet_name="2_Diferencias_Tiendas", index=False)
        df_dup.to_excel(writer, sheet_name="3_Alerta_Ref_Duplicadas", index=False)
        df_inv.to_excel(writer, sheet_name="4_Alerta_Ref_Invalidas", index=False)
    return output.getvalue()


# -----------------------------------------------------------------------------
# CARGA AUTOMATICA DE DATOS (SILENCIOSA PARA PRIORIZAR FILTROS EN SIDEBAR)
# -----------------------------------------------------------------------------
default_drive_url = ""
try:
    default_drive_url = st.secrets.get("DRIVE_FOLDER_URL", "")
except Exception:
    default_drive_url = ""

repo_a, repo_u = find_repo_excel_files()
has_repo_files = repo_a is not None and repo_u is not None

amrepa_bytes = None
unigis_bytes = None
fuente_label = ""

# Prioridad 1: Base persistente en GitHub | Prioridad 2: Google Drive Secret | Prioridad 3: Manual
if has_repo_files:
    with open(repo_a, "rb") as fa:
        amrepa_bytes = fa.read()
    with open(repo_u, "rb") as fu:
        unigis_bytes = fu.read()
    fuente_label = f"`{os.path.basename(repo_a)}` + `{os.path.basename(repo_u)}`"
elif default_drive_url:
    try:
        amrepa_bytes, unigis_bytes, name_a, name_u = fetch_from_drive_folder(default_drive_url)
        fuente_label = f"Drive (`{name_a}` + `{name_u}`)"
    except Exception:
        pass

# Si se subieron archivos manuales en el expander inferior, toman prioridad temporal
if "manual_a" in st.session_state and st.session_state["manual_a"] is not None:
    amrepa_bytes = st.session_state["manual_a"]
if "manual_u" in st.session_state and st.session_state["manual_u"] is not None:
    unigis_bytes = st.session_state["manual_u"]

if amrepa_bytes is None or unigis_bytes is None:
    st.title("🚚 Estadística de Camión")
    st.warning("No se detectaron archivos base. Cargalos desde el menú lateral.")
    up_a = st.sidebar.file_uploader("Archivo WMS (AMREPA.xlsx)", type=["xlsx"])
    up_u = st.sidebar.file_uploader("Reporte TMS (UNIGIS Detalle.xlsx)", type=["xlsx"])
    if up_a and up_u:
        st.session_state["manual_a"] = up_a.getvalue()
        st.session_state["manual_u"] = up_u.getvalue()
        st.rerun()
    st.stop()

data = load_and_process_data(amrepa_bytes, unigis_bytes)
master = data["master_trips"].copy()
dup_detail = data["dup_detail"].copy()
invalid_detail = data["invalid_detail"].copy()

# -----------------------------------------------------------------------------
# SIDEBAR: FILTROS GLOBALES ARRIBA (ACTUALIZAN TODAS LAS TABLAS Y GRAFICOS)
# -----------------------------------------------------------------------------
st.sidebar.header("🔎 Filtros Globales")

# 1. Filtro por Mes
all_months = sorted([m for m in master["Mes_Label"].unique() if m != "Sin fecha"])
selected_months = st.sidebar.multiselect(
    "Mes",
    options=all_months,
    default=[],
    placeholder="Todos los meses",
)

# 2. Filtro por Semana (dinamico segun mes seleccionado)
df_for_weeks = master[master["Mes_Label"].isin(selected_months)] if selected_months else master
weeks_df = (
    df_for_weeks[df_for_weeks["Semana_Label"] != "Sin semana"][["Semana_Sort", "Semana_Label"]]
    .drop_duplicates()
    .sort_values("Semana_Sort")
)
all_weeks = weeks_df["Semana_Label"].tolist()
selected_weeks = st.sidebar.multiselect(
    "Semana",
    options=all_weeks,
    default=[],
    placeholder="Todas las semanas",
)

# 3. Filtro por Rango de Fechas
min_date = master["Fecha_Base"].dropna().min().date()
max_date = master["Fecha_Base"].dropna().max().date()

date_range = st.sidebar.date_input(
    "Rango Fecha Reparto",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    d_start, d_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    d_start = d_end = pd.to_datetime(date_range)

# 4. Filtro por Operacion (CD)
all_ops = sorted(master["Operacion"].dropna().unique().tolist())
selected_ops = st.sidebar.multiselect(
    "Operación (CD)",
    options=all_ops,
    default=all_ops,
)

# Aplicar filtros globales sobre master
mask_base = (
    (master["Fecha_Base"] >= d_start)
    & (master["Fecha_Base"] <= d_end)
    & (master["Operacion"].isin(selected_ops))
)
if selected_months:
    mask_base = mask_base & (master["Mes_Label"].isin(selected_months))
if selected_weeks:
    mask_base = mask_base & (master["Semana_Label"].isin(selected_weeks))

df_filt = master[mask_base].copy()

# 5. Filtro por Transportista
all_trans = sorted(df_filt["Transportista"].dropna().unique().tolist())
selected_trans = st.sidebar.multiselect(
    "Transportista (opcional)",
    options=all_trans,
    default=[],
    placeholder="Todos los transportistas",
)
if selected_trans:
    df_filt = df_filt[df_filt["Transportista"].isin(selected_trans)].copy()

# Aplicar mismos filtros temporales y de CD a alertas
mask_dup = (
    (dup_detail["Fecha_Reparto"] >= d_start)
    & (dup_detail["Fecha_Reparto"] <= d_end)
    & (dup_detail["Operacion"].isin(selected_ops))
)
mask_inv = (
    (invalid_detail["Fecha_Reparto"] >= d_start)
    & (invalid_detail["Fecha_Reparto"] <= d_end)
    & (invalid_detail["Operacion"].isin(selected_ops))
)
if selected_months:
    mask_dup = mask_dup & (dup_detail["Fecha_Reparto"].dt.strftime("%Y-%m").isin(selected_months))
    mask_inv = mask_inv & (invalid_detail["Fecha_Reparto"].dt.strftime("%Y-%m").isin(selected_months))
if selected_weeks:
    dup_sem = "Sem " + dup_detail["Fecha_Reparto"].dt.to_period("W").apply(lambda p: p.start_time).dt.strftime("%d/%m")
    inv_sem = "Sem " + invalid_detail["Fecha_Reparto"].dt.to_period("W").apply(lambda p: p.start_time).dt.strftime("%d/%m")
    mask_dup = mask_dup & (dup_sem.isin(selected_weeks))
    mask_inv = mask_inv & (inv_sem.isin(selected_weeks))

dup_filt = dup_detail[mask_dup].copy()
inv_filt = invalid_detail[mask_inv].copy()

# Seccion colapsada al final del sidebar para carga manual / info de base
st.sidebar.divider()
with st.sidebar.expander("⚙️ Base de Datos / Carga Manual", expanded=False):
    st.caption(f"Fuente activa: {fuente_label}")
    up_a_alt = st.file_uploader("Reemplazar AMREPA (.xlsx)", type=["xlsx"], key="up_a_alt")
    up_u_alt = st.file_uploader("Reemplazar UNIGIS (.xlsx)", type=["xlsx"], key="up_u_alt")
    if up_a_alt is not None:
        st.session_state["manual_a"] = up_a_alt.getvalue()
    if up_u_alt is not None:
        st.session_state["manual_u"] = up_u_alt.getvalue()
    if st.button("Restaurar base original"):
        st.session_state["manual_a"] = None
        st.session_state["manual_u"] = None
        st.rerun()

# -----------------------------------------------------------------------------
# CABECERA PRINCIPAL Y NAVEGACION POR SOLAPAS
# -----------------------------------------------------------------------------
st.title("🚚 Estadística de Camión")
st.caption(
    f"Período activo: **{d_start.strftime('%d/%m/%Y')}** al **{d_end.strftime('%d/%m/%Y')}** | "
    f"CDs: **{', '.join(map(str, selected_ops))}** | Todos los filtros laterales actualizan todas las vistas."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 1. Conciliación Macro (WMS vs. TMS)",
        "📍 2. Visitas GPS y Geocercas",
        "⏱️ 3. Estadística de Tiempos",
        "🧹 4. Data Quality y Anomalías",
    ]
)

# =============================================================================
# SOLAPA 1: VISION MACRO - CONCILIACION WMS VS TMS
# =============================================================================
with tab1:
    st.subheader("Conciliación Macro: Despachos WMS (AMREPA) vs. Rutas TMS (UNIGIS)")

    viajes_wms = int((df_filt["_merge"].isin(["both", "left_only"])).sum())
    viajes_tms = int((df_filt["_merge"].isin(["both", "right_only"])).sum())
    viajes_match = int((df_filt["_merge"] == "both").sum())
    viajes_gap = int((df_filt["_merge"] == "left_only").sum())

    pct_cob_viajes = (viajes_match / viajes_wms * 100) if viajes_wms > 0 else 0.0

    wms_only_trips = df_filt[df_filt["_merge"].isin(["both", "left_only"])]
    matched_only = df_filt[df_filt["_merge"] == "both"]
    gap_only_trips = df_filt[df_filt["_merge"] == "left_only"]

    tiendas_wms_total = int(wms_only_trips["Tiendas_WMS"].sum())
    tiendas_ambos_match = int(matched_only["Tiendas_Match"].sum())
    tiendas_faltan_viaje = int(gap_only_trips["Tiendas_WMS"].sum())
    tiendas_faltan_match = int(matched_only["Cant_Tiendas_Faltan_TMS"].sum())
    tiendas_faltan_total = tiendas_faltan_viaje + tiendas_faltan_match

    tiendas_wms_en_match = int(matched_only["Tiendas_WMS"].sum())
    pct_cob_tiendas_global = (
        (tiendas_ambos_match / tiendas_wms_total * 100) if tiendas_wms_total > 0 else 0.0
    )
    pct_cob_tiendas_match = (
        (tiendas_ambos_match / tiendas_wms_en_match * 100) if tiendas_wms_en_match > 0 else 0.0
    )

    cant_dup_refs = int(dup_filt["Ref_Clean"].nunique())
    cant_inv_trips = int(inv_filt["Id Viaje"].nunique())

    def _fmt_int(val: int) -> str:
        return f"{val:,}".replace(",", ".")

    def _fmt_pct(val: float) -> str:
        return f"{val:.2f}%".replace(".", ",")

    def _kpi_card_html(title: str, value: str, line1: str, line1_color: str = "#475569", line2: str = "") -> str:
        line2_html = (
            f'<div style="font-size:12px; color:#64748b; margin-top:2px; line-height:1.3;">{line2}</div>'
            if line2
            else ""
        )
        return f"""
        <div style="background-color:#f8f9fb; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px; height:100%;">
            <div style="font-size:12.5px; font-weight:600; color:#475569; margin-bottom:4px;">{title}</div>
            <div style="font-size:28px; font-weight:700; color:#0f172a; line-height:1.15; margin-bottom:6px;">{value}</div>
            <div style="font-size:12.5px; font-weight:600; color:{line1_color}; line-height:1.3;">{line1}</div>
            {line2_html}
        </div>
        """

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.15, 1.45, 1.05])
    with c1:
        st.markdown(
            _kpi_card_html(
                "Viajes Despachados (WMS)",
                _fmt_int(viajes_wms),
                "Base AMREPA",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _kpi_card_html(
                "Viajes Válidos (TMS)",
                _fmt_int(viajes_tms),
                f"{_fmt_int(viajes_match)} matcheados",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _kpi_card_html(
                "% Cobertura Viajes",
                _fmt_pct(pct_cob_viajes),
                f"Faltan {_fmt_int(viajes_gap)} viajes en TMS",
                "#dc2626" if viajes_gap > 0 else "#16a34a",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _kpi_card_html(
                "% Cobertura Tiendas",
                _fmt_pct(pct_cob_tiendas_global),
                f"Faltan {_fmt_int(tiendas_faltan_total)} tiendas en TMS:",
                "#dc2626" if tiendas_faltan_total > 0 else "#16a34a",
                f"• <b>{_fmt_int(tiendas_faltan_viaje)}</b> por viaje caído &nbsp;|&nbsp; <b>{_fmt_int(tiendas_faltan_match)}</b> en viaje activo",
            ),
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            _kpi_card_html(
                "Alertas Integridad TMS",
                _fmt_int(cant_dup_refs + cant_inv_trips),
                f"{cant_dup_refs} duplicadas | {cant_inv_trips} inválidas",
                "#d97706" if (cant_dup_refs + cant_inv_trips) > 0 else "#475569",
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE A: TABLA RESUMEN POR OPERACION (ALINEADA CON TARJETA UNIFICADA)
    # -------------------------------------------------------------------------
    st.markdown("### 1. Resumen por Operación (Centro de Distribución)")

    rows_op = []
    for op in sorted(df_filt["Operacion"].dropna().unique()):
        sub = df_filt[df_filt["Operacion"] == op]
        v_wms = int(sub["_merge"].isin(["both", "left_only"]).sum())
        v_tms = int(sub["_merge"].isin(["both", "right_only"]).sum())
        v_match = int((sub["_merge"] == "both").sum())
        v_gap = int((sub["_merge"] == "left_only").sum())
        cob_v = (v_match / v_wms * 100) if v_wms > 0 else 0.0

        sub_wms_all = sub[sub["_merge"].isin(["both", "left_only"])]
        sub_m = sub[sub["_merge"] == "both"]
        sub_g = sub[sub["_merge"] == "left_only"]

        t_wms_tot = int(sub_wms_all["Tiendas_WMS"].sum())
        t_ambos = int(sub_m["Tiendas_Match"].sum())
        t_faltan_v = int(sub_g["Tiendas_WMS"].sum())
        t_faltan_m = int(sub_m["Cant_Tiendas_Faltan_TMS"].sum())
        cob_t_glob = (t_ambos / t_wms_tot * 100) if t_wms_tot > 0 else 0.0

        rows_op.append(
            {
                "CD": str(int(op)),
                "Viajes WMS": v_wms,
                "Viajes TMS": v_tms,
                "Match": v_match,
                "Gap Viajes": v_gap,
                "% Cob. Viajes": round(cob_v, 2),
                "Tiendas WMS": t_wms_tot,
                "En TMS": t_ambos,
                "Faltan x Viaje": t_faltan_v,
                "Faltan x Parada": t_faltan_m,
                "% Cob. Tiendas": round(cob_t_glob, 2),
            }
        )

    df_resumen_op = pd.DataFrame(rows_op)
    if not df_resumen_op.empty:
        total_row = pd.DataFrame(
            [
                {
                    "CD": "TOTAL",
                    "Viajes WMS": df_resumen_op["Viajes WMS"].sum(),
                    "Viajes TMS": df_resumen_op["Viajes TMS"].sum(),
                    "Match": df_resumen_op["Match"].sum(),
                    "Gap Viajes": df_resumen_op["Gap Viajes"].sum(),
                    "% Cob. Viajes": round(pct_cob_viajes, 2),
                    "Tiendas WMS": df_resumen_op["Tiendas WMS"].sum(),
                    "En TMS": df_resumen_op["En TMS"].sum(),
                    "Faltan x Viaje": df_resumen_op["Faltan x Viaje"].sum(),
                    "Faltan x Parada": df_resumen_op["Faltan x Parada"].sum(),
                    "% Cob. Tiendas": round(pct_cob_tiendas_global, 2),
                }
            ]
        )
        df_resumen_op_display = pd.concat([df_resumen_op, total_row], ignore_index=True)

        st.dataframe(
            df_resumen_op_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "CD": st.column_config.TextColumn("CD", width="small"),
                "Viajes WMS": st.column_config.NumberColumn("Viajes WMS", format="%d", width="small"),
                "Viajes TMS": st.column_config.NumberColumn("Viajes TMS", format="%d", width="small"),
                "Match": st.column_config.NumberColumn("Match", format="%d", width="small"),
                "Gap Viajes": st.column_config.NumberColumn("Gap Viajes", format="%d", width="small"),
                "% Cob. Viajes": st.column_config.ProgressColumn(
                    "% Cob. Viajes",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                    width="small",
                ),
                "Tiendas WMS": st.column_config.NumberColumn("Tiendas WMS", format="%d", width="small"),
                "En TMS": st.column_config.NumberColumn("En TMS", format="%d", width="small"),
                "Faltan x Viaje": st.column_config.NumberColumn("Faltan x Viaje", format="%d", width="small"),
                "Faltan x Parada": st.column_config.NumberColumn("Faltan x Parada", format="%d", width="small"),
                "% Cob. Tiendas": st.column_config.ProgressColumn(
                    "% Cob. Tiendas",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                    width="small",
                ),
            },
        )

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE B: EVOLUCION TEMPORAL DINAMICA (DIA / SEMANA / MES Y ALMACEN)
    # -------------------------------------------------------------------------
    st.markdown("### 2. Evolución Temporal de Cobertura y Brechas")

    ctrl1, ctrl2 = st.columns([1, 1.5])
    with ctrl1:
        gran_tiempo = st.radio(
            "Agrupar eje temporal por:",
            options=["Día", "Semana", "Mes"],
            index=1,
            horizontal=True,
        )
    with ctrl2:
        modo_vista = st.radio(
            "Modo de visualización:",
            options=[
                "Consolidado (Barras + Línea %)",
                "Abierto por Almacén (Mapa de Calor %)",
            ],
            index=0,
            horizontal=True,
        )

    df_time = df_filt[df_filt["_merge"].isin(["both", "left_only"])].copy()

    # Drill-down inteligente cuando se elige vista por Dia
    if gran_tiempo == "Día" and not df_time.empty:
        semanas_disp = (
            df_time[["Semana_Sort", "Semana_Label"]]
            .drop_duplicates()
            .sort_values("Semana_Sort")["Semana_Label"]
            .tolist()
        )
        default_sems = semanas_disp[-2:] if len(semanas_disp) >= 2 else semanas_disp

        d_col1, d_col2 = st.columns([2, 1])
        with d_col1:
            sems_elegidas = st.multiselect(
                "Drill-down — Seleccioná semana(s) para abrir por día (por defecto últimas 2):",
                options=semanas_disp,
                default=default_sems,
                placeholder="Todas las semanas",
            )
        with d_col2:
            st.write("")
            excluir_guardias = st.checkbox(
                "Ocultar domingos/feriados (<30 viajes)",
                value=False,
                help="Evita caídas visuales artificiales en días de guardia con muy pocos despachos.",
            )

        if sems_elegidas:
            df_time = df_time[df_time["Semana_Label"].isin(sems_elegidas)].copy()

        if excluir_guardias and not df_time.empty:
            day_vols = df_time.groupby(df_time["Fecha_Base"].dt.floor("D"))["Ref_Clean"].transform("count")
            df_time = df_time[day_vols >= 30].copy()

    dias_es = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
    if gran_tiempo == "Día":
        df_time["Periodo_Sort"] = df_time["Fecha_Base"].dt.floor("D")
        df_time["Periodo"] = (
            df_time["Periodo_Sort"].dt.dayofweek.map(dias_es)
            + " "
            + df_time["Periodo_Sort"].dt.strftime("%d/%m")
        )
    elif gran_tiempo == "Semana":
        df_time["Periodo_Sort"] = df_time["Semana_Sort"]
        df_time["Periodo"] = df_time["Semana_Label"]
    else:
        df_time["Periodo_Sort"] = df_time["Fecha_Base"].dt.to_period("M").apply(lambda p: p.start_time)
        df_time["Periodo"] = df_time["Periodo_Sort"].dt.strftime("%Y-%m")

    if df_time.empty:
        st.info("No hay viajes despachados en el rango y filtros seleccionados.")
    elif modo_vista == "Consolidado (Barras + Línea %)":
        agg_t = (
            df_time.groupby(["Periodo_Sort", "Periodo"], as_index=False)
            .agg(
                Viajes_WMS=("Ref_Clean", "count"),
                Match=("_merge", lambda s: int((s == "both").sum())),
                Gap_Solo_WMS=("_merge", lambda s: int((s == "left_only").sum())),
            )
            .sort_values("Periodo_Sort")
        )
        agg_t["Pct_Cobertura"] = np.where(
            agg_t["Viajes_WMS"] > 0,
            (agg_t["Match"] / agg_t["Viajes_WMS"] * 100).round(2),
            0.0,
        )

        n_pts = len(agg_t)
        show_text_labels = n_pts <= 16

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=agg_t["Periodo"],
                y=agg_t["Match"],
                name="Viajes En Ambos (WMS + TMS)",
                marker_color="#1d4ed8",
                hovertemplate="<b>%{x}</b><br>En Ambos: %{y:,}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=agg_t["Periodo"],
                y=agg_t["Gap_WMS"] if "Gap_WMS" in agg_t else agg_t["Gap_Solo_WMS"],
                name="Gap Solo WMS (Sin UNIGIS)",
                marker_color="#dc2626",
                hovertemplate="<b>%{x}</b><br>Gap Solo WMS: %{y:,}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=agg_t["Periodo"],
                y=agg_t["Pct_Cobertura"],
                name="% Cobertura TMS",
                mode="lines+markers",
                line=dict(color="#f59e0b", width=3.5),
                marker=dict(
                    size=10 if show_text_labels else 7,
                    color="#f59e0b",
                    line=dict(width=2, color="#0f172a"),
                ),
                hovertemplate="<b>%{x}</b><br>Cobertura: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=True,
        )

        # Etiquetas con fondo blanco de alto contraste sobre la linea
        if show_text_labels:
            for row_t in agg_t.itertuples(index=False):
                fig.add_annotation(
                    x=row_t.Periodo,
                    y=row_t.Pct_Cobertura,
                    yref="y2",
                    text=f"<b>{row_t.Pct_Cobertura:.1f}%</b>",
                    showarrow=False,
                    yshift=18,
                    font=dict(color="#0f172a", size=11),
                    bgcolor="rgba(255, 255, 255, 0.92)",
                    bordercolor="#f59e0b",
                    borderwidth=1.2,
                    borderpad=3,
                )

        # Linea objetivo 98% correctamente ubicada en y=98 del eje derecho
        fig.add_shape(
            type="line",
            xref="paper",
            x0=0,
            x1=1,
            yref="y2",
            y0=98,
            y1=98,
            line=dict(color="#334155", width=1.5, dash="dot"),
        )
        fig.add_annotation(
            xref="paper",
            x=0.99,
            yref="y2",
            y=98,
            text="Objetivo 98%",
            showarrow=False,
            yshift=-12,
            xanchor="right",
            font=dict(color="#334155", size=11),
            bgcolor="rgba(255, 255, 255, 0.85)",
        )

        fig.update_layout(
            barmode="stack",
            height=450 if gran_tiempo == "Día" else 420,
            margin=dict(l=20, r=20, t=35, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            xaxis=dict(
                rangeslider=dict(visible=(gran_tiempo == "Día" and n_pts > 14), thickness=0.06),
                tickangle=-35 if n_pts > 14 else 0,
            ),
        )
        fig.update_yaxes(title_text="Cantidad de Viajes", secondary_y=False)
        fig.update_yaxes(
            title_text="% Cobertura TMS",
            range=[0, 114],
            dtick=20,
            showgrid=False,
            secondary_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        agg_op_t = (
            df_time.groupby(["Operacion", "Periodo_Sort", "Periodo"], as_index=False)
            .agg(
                Viajes_WMS=("Ref_Clean", "count"),
                Match=("_merge", lambda s: int((s == "both").sum())),
                Gap=("_merge", lambda s: int((s == "left_only").sum())),
            )
            .sort_values(["Operacion", "Periodo_Sort"])
        )
        agg_op_t["Pct_Cobertura"] = np.where(
            agg_op_t["Viajes_WMS"] > 0,
            (agg_op_t["Match"] / agg_op_t["Viajes_WMS"] * 100).round(1),
            np.nan,
        )
        agg_op_t["Operacion_Str"] = "CD " + agg_op_t["Operacion"].astype(str)

        ordered_periods = (
            agg_op_t[["Periodo_Sort", "Periodo"]]
            .drop_duplicates()
            .sort_values("Periodo_Sort")["Periodo"]
            .tolist()
        )
        ordered_ops = sorted(agg_op_t["Operacion_Str"].unique().tolist())

        piv_pct = agg_op_t.pivot(index="Operacion_Str", columns="Periodo", values="Pct_Cobertura").reindex(
            index=ordered_ops, columns=ordered_periods
        )
        piv_wms = agg_op_t.pivot(index="Operacion_Str", columns="Periodo", values="Viajes_WMS").reindex(
            index=ordered_ops, columns=ordered_periods
        )
        piv_gap = agg_op_t.pivot(index="Operacion_Str", columns="Periodo", values="Gap").reindex(
            index=ordered_ops, columns=ordered_periods
        )

        show_cell_text = len(ordered_periods) <= 22
        text_matrix = np.where(
            piv_pct.notna(),
            piv_pct.map(lambda v: f"{v:.1f}%" if pd.notna(v) else ""),
            "S/D",
        )
        customdata = np.dstack([piv_wms.fillna(0).values, piv_gap.fillna(0).values])

        fig_hm = go.Figure(
            data=go.Heatmap(
                z=piv_pct.values,
                x=ordered_periods,
                y=ordered_ops,
                text=text_matrix if show_cell_text else None,
                texttemplate="%{text}" if show_cell_text else None,
                textfont=dict(size=11),
                xgap=2,
                ygap=2,
                colorscale="RdYlGn",
                zmin=40,
                zmax=100,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y} — %{x}</b><br>"
                    "Cobertura TMS: %{z:.1f}%<br>"
                    "Viajes WMS: %{customdata[0]:,.0f}<br>"
                    "Gap sin TMS: %{customdata[1]:,.0f}<extra></extra>"
                ),
                colorbar=dict(title="% Cob."),
            )
        )
        fig_hm.update_layout(
            height=390,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title=f"Período ({gran_tiempo})",
            yaxis_title="Almacén / Operación",
            xaxis=dict(tickangle=-35 if len(ordered_periods) > 14 else 0),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE C: DETALLE DE AUDITORIA Y ALERTAS DE INTEGRIDAD (FECHAS SIN 00:00)
    # -------------------------------------------------------------------------
    st.markdown("### 3. Auditoría de Brechas y Alertas de Integridad")

    df_gap_export = (
        df_filt[df_filt["_merge"] == "left_only"][
            ["Operacion", "Fecha_Base", "Ref_Clean", "Transportista", "Tiendas_WMS", "Set_Tiendas_WMS"]
        ]
        .copy()
        .sort_values(["Operacion", "Fecha_Base", "Ref_Clean"])
    )
    df_gap_export["Fecha Reparto"] = df_gap_export["Fecha_Base"].dt.strftime("%d/%m/%Y").fillna("")
    df_gap_export["Tiendas Despachadas WMS"] = df_gap_export["Set_Tiendas_WMS"].apply(lambda x: ", ".join(x))
    df_gap_export = df_gap_export[
        ["Operacion", "Fecha Reparto", "Ref_Clean", "Transportista", "Tiendas_WMS", "Tiendas Despachadas WMS"]
    ].rename(
        columns={
            "Ref_Clean": "Ref Externa Viaje",
            "Tiendas_WMS": "Cant. Tiendas WMS",
        }
    )

    df_store_diff_export = (
        df_filt[
            (df_filt["_merge"] == "both")
            & ((df_filt["Cant_Tiendas_Faltan_TMS"] > 0) | (df_filt["Cant_Tiendas_Extra_TMS"] > 0))
        ][
            [
                "Operacion",
                "Fecha_Base",
                "Ref_Clean",
                "Transportista",
                "Tiendas_WMS",
                "Tiendas_TMS",
                "Lista_Faltan_TMS",
                "Lista_Extra_TMS",
            ]
        ]
        .copy()
        .sort_values(["Operacion", "Fecha_Base", "Ref_Clean"])
    )
    df_store_diff_export["Fecha Reparto"] = df_store_diff_export["Fecha_Base"].dt.strftime("%d/%m/%Y").fillna("")
    df_store_diff_export = df_store_diff_export[
        [
            "Operacion",
            "Fecha Reparto",
            "Ref_Clean",
            "Transportista",
            "Tiendas_WMS",
            "Tiendas_TMS",
            "Lista_Faltan_TMS",
            "Lista_Extra_TMS",
        ]
    ].rename(
        columns={
            "Ref_Clean": "Ref Externa Viaje",
            "Tiendas_WMS": "Cant. Tiendas WMS",
            "Tiendas_TMS": "Cant. Tiendas TMS",
            "Lista_Faltan_TMS": "Tiendas en WMS pero NO en TMS",
            "Lista_Extra_TMS": "Tiendas en TMS pero NO en WMS",
        }
    )

    dup_display = dup_filt.copy()
    dup_display["Fecha Reparto"] = dup_display["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")
    dup_display = dup_display[
        [
            "Operacion",
            "Fecha Reparto",
            "Hora Reparto",
            "Ref_Clean",
            "Id Viaje",
            "Dominio",
            "Tipo de Camion",
            "Tiendas_Planificadas",
            "Tiendas_Con_Visita_GPS",
        ]
    ].rename(
        columns={
            "Ref_Clean": "Ref Externa Viaje",
            "Tiendas_Planificadas": "Tiendas Plan.",
            "Tiendas_Con_Visita_GPS": "Con Visita GPS",
        }
    )

    inv_display = inv_filt.copy()
    inv_display["Fecha Reparto"] = inv_display["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")
    inv_display = inv_display[
        [
            "Operacion",
            "Fecha Reparto",
            "Hora Reparto",
            "Id Viaje",
            "Dominio",
            "Tipo de Camion",
            "Ref_Externa_Original",
            "Tiendas_Planificadas",
            "Observacion_Principal",
        ]
    ].rename(
        columns={
            "Ref_Externa_Original": "Ref Externa Original",
            "Tiendas_Planificadas": "Tiendas Plan.",
            "Observacion_Principal": "Observación",
        }
    )

    excel_bytes = build_audit_excel(df_gap_export, df_store_diff_export, dup_display, inv_display)
    st.download_button(
        label="📥 Descargar Auditoría Completa en Excel (.xlsx)",
        data=excel_bytes,
        file_name="Auditoria_Conciliacion_WMS_TMS.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    sub_t1, sub_t2, sub_t3 = st.tabs(
        [
            f"🚨 Viajes Faltantes en TMS ({len(df_gap_export)})",
            f"🏪 Diferencias de Tiendas ({len(df_store_diff_export)})",
            f"⚠️ Alertas Calidad UNIGIS ({cant_dup_refs} dup | {cant_inv_trips} inv.)",
        ]
    )

    with sub_t1:
        st.caption("Viajes despachados en AMREPA (WMS) que no ingresaron o no matchearon en UNIGIS.")
        st.dataframe(df_gap_export, use_container_width=True, hide_index=True)

    with sub_t2:
        st.caption("Viajes que sí existen en ambos sistemas, pero difieren en las tiendas asignadas.")
        st.dataframe(df_store_diff_export, use_container_width=True, hide_index=True)

    with sub_t3:
        st.markdown("**A. Ref Externa Viaje duplicadas en múltiples `Id Viaje` (ambos conservados como alerta)**")
        st.dataframe(dup_display, use_container_width=True, hide_index=True)

        st.markdown("**B. Viajes descartados de UNIGIS por `Ref Externa Viaje` vacía o con formato inválido**")
        st.dataframe(inv_display, use_container_width=True, hide_index=True)

# =============================================================================
# PLACEHOLDERS PARA PROXIMAS SOLAPAS
# =============================================================================
with tab2:
    st.info(
        "🚧 **Próximo paso (Solapa 2):** Auditoría de Visitas GPS y Geocercas — "
        "Análisis de tiendas 'No visitada' vs. visitadas, ranking de tiendas con geocerca defectuosa "
        "y efectividad GPS por transportista y tipo de camión."
    )

with tab3:
    st.info(
        "🚧 **Solapa 3:** Estadística de Tiempos y Flota — "
        "Tiempos de estadía en tienda (mediana, p75, p90), permanencia en CD origen, "
        "tiempos de tránsito y comparativa Cercana vs. Lejana (>100 km)."
    )

with tab4:
    st.info(
        "🚧 **Solapa 4:** Data Quality y Anomalías — "
        "Detección de estadías extremas (>8h), kilómetros en 0, velocidades irreales y tiempos de retorno abiertos."
    )
