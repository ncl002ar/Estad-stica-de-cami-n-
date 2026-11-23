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
    nomina_f = sorted([f for f in candidates if "nomina" in os.path.basename(f).lower()])
    nom_path = nomina_f[-1] if nomina_f else None
    if amrepa_f and unigis_f:
        return amrepa_f[-1], unigis_f[-1], nom_path
    return None, None, nom_path


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

        nomina_files = sorted(
            [f for f in all_xlsx if "nomina" in os.path.basename(f).lower()]
        )
        n_bytes = None
        name_n = None
        if nomina_files:
            with open(nomina_files[-1], "rb") as fn:
                n_bytes = fn.read()
            name_n = os.path.basename(nomina_files[-1])

        with open(amrepa_files[-1], "rb") as fa:
            a_bytes = fa.read()
        with open(unigis_files[-1], "rb") as fu:
            u_bytes = fu.read()

        return (
            a_bytes,
            u_bytes,
            n_bytes,
            os.path.basename(amrepa_files[-1]),
            os.path.basename(unigis_files[-1]),
            name_n,
        )


# -----------------------------------------------------------------------------
# FUNCIONES DE LIMPIEZA Y PROCESAMIENTO DE DATOS (BACKEND)
# -----------------------------------------------------------------------------
def _clean_int_str(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    return num.apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)


def _to_float_safe(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def _build_nomina_lookup(nomina_bytes: bytes | None) -> pd.DataFrame:
    if nomina_bytes is None:
        return pd.DataFrame(columns=["Tipo_Vehiculo_Nom", "Satelital_Nom", "Empresa_Nom"])
    try:
        df_n = pd.read_excel(io.BytesIO(nomina_bytes), sheet_name=0)
    except Exception:
        return pd.DataFrame(columns=["Tipo_Vehiculo_Nom", "Satelital_Nom", "Empresa_Nom"])

    if "Patente" not in df_n.columns:
        return pd.DataFrame(columns=["Tipo_Vehiculo_Nom", "Satelital_Nom", "Empresa_Nom"])

    df_n = df_n.copy()
    df_n["Patente_Clean"] = (
        df_n["Patente"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "", regex=False)
    )
    df_n = df_n[~df_n["Patente_Clean"].isin(["", "0", "SINPATENTE", "NAN"])].copy()

    sit_rank = {
        "ACTIVO": 1,
        "ACTIVO EVENTUAL": 2,
        "REEMPLAZADO X BACKUP": 3,
        "TRACTOR": 4,
        "BACKUP INACTIVO": 5,
        "INACTIVO": 6,
        "BAJA": 7,
    }
    if "SITUACION" in df_n.columns:
        df_n["_rank"] = df_n["SITUACION"].astype(str).str.strip().str.upper().map(sit_rank).fillna(99)
    else:
        df_n["_rank"] = 50

    df_n = df_n.sort_values("_rank").drop_duplicates("Patente_Clean", keep="first").set_index("Patente_Clean")

    out = pd.DataFrame(index=df_n.index)
    out["Tipo_Vehiculo_Nom"] = (
        df_n["Tipo Camion"].fillna("").astype(str).str.strip().str.upper()
        if "Tipo Camion" in df_n.columns
        else ""
    )
    out["Satelital_Nom"] = (
        df_n["Satelital"].fillna("Sin dato").astype(str).str.strip()
        if "Satelital" in df_n.columns
        else "Sin dato"
    )
    out["Empresa_Nom"] = (
        df_n["Empresa"].fillna("Sin dato").astype(str).str.strip()
        if "Empresa" in df_n.columns
        else "Sin dato"
    )
    return out


@st.cache_data(show_spinner="Procesando y conciliando bases AMREPA, UNIGIS y Nómina...")
def load_and_process_data(amrepa_bytes: bytes, unigis_bytes: bytes, nomina_bytes: bytes | None = None):
    # 1. Lectura de archivos Excel
    df_a_raw = pd.read_excel(io.BytesIO(amrepa_bytes), sheet_name=0)
    xls_u = pd.ExcelFile(io.BytesIO(unigis_bytes))
    sheet_u = (
        "Supply - Reporte Seguimiento De"
        if "Supply - Reporte Seguimiento De" in xls_u.sheet_names
        else xls_u.sheet_names[0]
    )
    df_u_raw = pd.read_excel(io.BytesIO(unigis_bytes), sheet_name=sheet_u)
    nomina_lut = _build_nomina_lookup(nomina_bytes)

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

    # Parseo de Distancia en Activacion (si viene en el reporte)
    if "Distancia en Activacion" in df_u.columns:
        df_u["Dist_Activacion_km"] = _to_float_safe(df_u["Distancia en Activacion"])
    else:
        df_u["Dist_Activacion_km"] = np.nan

    # Parseo de etiquetas multiples en Observacion (soporta formato simple y compuesto con '|')
    obs_str = df_u["Observacion"].fillna("").astype(str)
    df_u["obs_no_visitada"] = obs_str.str.contains("No visitada", case=False, regex=False)
    df_u["obs_sin_coord"] = (df_u["Tipo de Zona"] == "Sin zona") | obs_str.str.contains(
        "Sin coordenadas", case=False, regex=False
    )
    df_u["obs_perdida_gps"] = obs_str.str.contains("Perdida de GPS", case=False, regex=False)
    df_u["obs_act_tarde"] = obs_str.str.contains("Activacion tarde", case=False, regex=False) | (
        df_u["Dist_Activacion_km"].fillna(0) > 5.0
    )
    df_u["obs_err_fin"] = obs_str.str.contains("Error en finalizacion", case=False, regex=False)
    df_u["visita_efectiva"] = (~df_u["obs_no_visitada"]) & (~df_u["obs_sin_coord"])

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
            Tiendas_Visitadas_GPS=("visita_efectiva", lambda s: int(s.sum())),
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
                "visita_efectiva",
                lambda s: int(
                    (
                        s
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

    # 5. Tabla maestra a nivel Parada en Tienda para Solapa 2 (Visitas GPS y Geocercas)
    a_trans_map = (
        df_a.groupby(["Operacion", "Ref_Clean"], as_index=False)
        .agg(Transportista=("Transportista", lambda s: " / ".join(sorted(s.unique()))))
    )

    stores_gps = u_stores_only.copy()
    stores_gps = stores_gps.merge(a_trans_map, on=["Operacion", "Ref_Clean"], how="left")
    stores_gps["Transportista"] = stores_gps["Transportista"].fillna("Sin dato WMS")
    stores_gps["Dominio"] = stores_gps["Dominio"].fillna("Sin dominio").astype(str).str.strip()
    stores_gps["Dominio_Clean"] = stores_gps["Dominio"].str.upper().str.replace(" ", "", regex=False)
    stores_gps["Interno"] = stores_gps["Interno"].fillna("S/D").astype(str).str.strip()
    stores_gps["Tipo de Camion"] = stores_gps["Tipo de Camion"].fillna("Sin dato").astype(str).str.strip()
    if not nomina_lut.empty:
        mapped_tipo = stores_gps["Dominio_Clean"].map(nomina_lut["Tipo_Vehiculo_Nom"]).fillna("")
        stores_gps["Tipo_Vehiculo"] = np.where(mapped_tipo != "", mapped_tipo, stores_gps["Tipo de Camion"])
        stores_gps["Satelital"] = stores_gps["Dominio_Clean"].map(nomina_lut["Satelital_Nom"]).fillna("Sin dato")
    else:
        stores_gps["Tipo_Vehiculo"] = stores_gps["Tipo de Camion"]
        stores_gps["Satelital"] = "Sin dato"

    stores_gps["Mes_Label"] = stores_gps["Fecha_Reparto"].dt.strftime("%Y-%m").fillna("Sin fecha")
    stores_gps["Semana_Sort"] = stores_gps["Fecha_Reparto"].dt.to_period("W").apply(
        lambda p: p.start_time if pd.notna(p) else pd.NaT
    )
    stores_gps["Semana_Label"] = np.where(
        stores_gps["Semana_Sort"].notna(),
        "Sem " + stores_gps["Semana_Sort"].dt.strftime("%d/%m"),
        "Sin semana",
    )

    # Mantenemos nuestra logica exacta:
    # - 0 tiendas visitadas en todo el viaje => Falla GPS / Unidad (Viaje Ciego)
    # - >=1 tienda visitada en el viaje pero esta parada no => Falla Geozona (Parcial)
    stores_gps["Visitadas_En_IdViaje"] = stores_gps.groupby("Id Viaje")["visita_efectiva"].transform("sum").astype(int)

    conds_gps = [
        stores_gps["visita_efectiva"],
        stores_gps["obs_sin_coord"],
        stores_gps["obs_no_visitada"] & (stores_gps["Visitadas_En_IdViaje"] == 0),
        stores_gps["obs_no_visitada"] & (stores_gps["Visitadas_En_IdViaje"] > 0),
    ]
    choices_gps = [
        "Visitada (OK)",
        "Sin coordenadas",
        "Viaje Ciego (GPS/Unidad)",
        "Falla Geocerca (Parcial)",
    ]
    stores_gps["Diagnostico_GPS"] = np.select(conds_gps, choices_gps, default="Otro")

    keep_store_cols = [
        "Operacion",
        "Fecha_Reparto",
        "Hora Reparto",
        "Mes_Label",
        "Semana_Sort",
        "Semana_Label",
        "Id Viaje",
        "Ref_Clean",
        "Transportista",
        "Dominio",
        "Interno",
        "Tipo de Camion",
        "Tipo_Vehiculo",
        "Satelital",
        "Tienda_Cod",
        "Tipo de Zona",
        "Categoria",
        "Tiempo de Estadia",
        "Cantidad Visitas",
        "Dist_Activacion_km",
        "obs_perdida_gps",
        "obs_act_tarde",
        "obs_err_fin",
        "Observacion",
        "Visitadas_En_IdViaje",
        "Diagnostico_GPS",
    ]
    stores_gps = stores_gps[keep_store_cols].copy()

    # 6. Preparacion de tablas base para Solapa 3 (Data Quality) y Solapa 4 (Tiempos)
    # IMPORTANTE: Ignoramos por completo la segunda fila de CD - Carga (retorno) y NO usamos
    # tiempos ni kms de retorno para excluir viajes de la base limpia.
    u_clean_num = u_valid.copy()
    for col_num in ["KMS Parada", "Velocidad Real Calculada", "KMS Total Viaje", "KMS Retorno"]:
        u_clean_num[col_num + "_num"] = _to_float_safe(u_clean_num[col_num])
    for col_time in ["Tiempo de Estadia", "Tiempo en Viaje", "Tiempo Retorno", "Cantidad Visitas"]:
        if col_time in u_clean_num.columns:
            u_clean_num[col_time] = _to_float_safe(u_clean_num[col_time])

    # Identificar secuencia de CD - Carga: 1 = CD Origen (Carga), >1 = Fila retorno (se descarta por completo)
    u_clean_num["cd_seq"] = np.where(
        u_clean_num["Tipo de Zona"] == "CD - Carga",
        u_clean_num.groupby(["Id Viaje", "Tipo de Zona"]).cumcount() + 1,
        0,
    )
    u_clean_num = u_clean_num[~((u_clean_num["Tipo de Zona"] == "CD - Carga") & (u_clean_num["cd_seq"] > 1))].copy()

    u_clean_num = u_clean_num.merge(a_trans_map, on=["Operacion", "Ref_Clean"], how="left")
    u_clean_num["Transportista"] = u_clean_num["Transportista"].fillna("Sin dato WMS")
    u_clean_num["Dominio"] = u_clean_num["Dominio"].fillna("Sin dominio").astype(str).str.strip()
    u_clean_num["Dominio_Clean"] = u_clean_num["Dominio"].str.upper().str.replace(" ", "", regex=False)
    u_clean_num["Interno"] = u_clean_num["Interno"].fillna("S/D").astype(str).str.strip()
    u_clean_num["Tipo de Camion"] = u_clean_num["Tipo de Camion"].fillna("Sin dato").astype(str).str.strip()
    if not nomina_lut.empty:
        mapped_tv = u_clean_num["Dominio_Clean"].map(nomina_lut["Tipo_Vehiculo_Nom"]).fillna("")
        u_clean_num["Tipo_Vehiculo"] = np.where(mapped_tv != "", mapped_tv, u_clean_num["Tipo de Camion"])
        u_clean_num["Satelital"] = u_clean_num["Dominio_Clean"].map(nomina_lut["Satelital_Nom"]).fillna("Sin dato")
        u_clean_num["Empresa_Flota"] = u_clean_num["Dominio_Clean"].map(nomina_lut["Empresa_Nom"]).fillna("Sin dato")
    else:
        u_clean_num["Tipo_Vehiculo"] = u_clean_num["Tipo de Camion"]
        u_clean_num["Satelital"] = "Sin dato"
        u_clean_num["Empresa_Flota"] = "Sin dato"

    for col_vol in ["Total Bultos", "Formato Equivalentes", "Apertura Puerta Trasera", "Cierre Puerta Trasera", "Minutos Apertura y Cierre"]:
        if col_vol in u_clean_num.columns:
            u_clean_num[col_vol + "_num"] = _to_float_safe(u_clean_num[col_vol])
        else:
            u_clean_num[col_vol + "_num"] = np.nan
    u_clean_num["Mes_Label"] = u_clean_num["Fecha_Reparto"].dt.strftime("%Y-%m").fillna("Sin fecha")
    u_clean_num["Semana_Sort"] = u_clean_num["Fecha_Reparto"].dt.to_period("W").apply(
        lambda p: p.start_time if pd.notna(p) else pd.NaT
    )
    u_clean_num["Semana_Label"] = np.where(
        u_clean_num["Semana_Sort"].notna(),
        "Sem " + u_clean_num["Semana_Sort"].dt.strftime("%d/%m"),
        "Sin semana",
    )

    # Parseo de timestamps reales para detectar solapamientos horarios entre paradas consecutivas
    u_clean_num["dt_ini_real"] = pd.to_datetime(
        u_clean_num["Inicio Horario Visita Real"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    u_clean_num["dt_fin_real"] = pd.to_datetime(
        u_clean_num["Fecha Hora Salida"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    u_clean_num["prev_dt_fin_real"] = u_clean_num.groupby("Id Viaje")["dt_fin_real"].shift(1)

    # Completitud de CD Origen (1a fila CD - Carga) y de Todas las Tiendas por Id Viaje
    cd_orig_rows = u_clean_num[u_clean_num["Tipo de Zona"] == "CD - Carga"].copy()
    cd_orig_rows["_cd_ok"] = cd_orig_rows["visita_efectiva"] & cd_orig_rows["Tiempo de Estadia"].notna()
    cd_orig_map = (
        cd_orig_rows.groupby("Id Viaje", as_index=False)
        .agg(
            CD_Origen_OK=("_cd_ok", "first"),
            Estadia_CD_Origen=("Tiempo de Estadia", "first"),
            Ini_CD_Origen=("Inicio Horario Visita Real", "first"),
            Fin_CD_Origen=("Fecha Hora Salida", "first"),
        )
    )

    is_st = u_clean_num["Tipo de Zona"].isin(["Tienda", "Sin zona"])
    st_comp_map = (
        u_clean_num[is_st]
        .groupby("Id Viaje", as_index=False)
        .agg(
            Tiendas_Plan_Viaje=("Tienda_Cod", "count"),
            Tiendas_OK_Viaje=("visita_efectiva", lambda s: int(s.sum())),
            Sum_KMS_Paradas=("KMS Parada_num", "sum"),
        )
    )
    st_comp_map["Todas_Tiendas_OK"] = (
        (st_comp_map["Tiendas_Plan_Viaje"] > 0)
        & (st_comp_map["Tiendas_Plan_Viaje"] == st_comp_map["Tiendas_OK_Viaje"])
    )

    # Tabla a nivel Viaje TMS (1 fila por Id Viaje)
    dq_trips = (
        u_clean_num.groupby("Id Viaje", as_index=False)
        .agg(
            Operacion=("Operacion", "first"),
            Fecha_Reparto=("Fecha_Reparto", "first"),
            Hora_Reparto=("Hora Reparto", "first"),
            Mes_Label=("Mes_Label", "first"),
            Semana_Sort=("Semana_Sort", "first"),
            Semana_Label=("Semana_Label", "first"),
            Ref_Clean=("Ref_Clean", "first"),
            Transportista=("Transportista", "first"),
            Dominio=("Dominio", "first"),
            Tipo_Camion=("Tipo de Camion", "first"),
            Categoria=("Categoria", "first"),
            KMS_Total=("KMS Total Viaje_num", "first"),
            KMS_Retorno=("KMS Retorno_num", "first"),
            Tiempo_Retorno=("Tiempo Retorno", "first"),
            Dist_Activacion_km=("Dist_Activacion_km", "first"),
            obs_act_tarde=("obs_act_tarde", "max"),
            obs_err_fin=("obs_err_fin", "max"),
        )
        .merge(cd_orig_map, on="Id Viaje", how="left")
        .merge(st_comp_map, on="Id Viaje", how="left")
    )
    dq_trips["CD_Origen_OK"] = dq_trips["CD_Origen_OK"].fillna(False)
    dq_trips["Todas_Tiendas_OK"] = dq_trips["Todas_Tiendas_OK"].fillna(False)
    dq_trips["Tiendas_Plan_Viaje"] = dq_trips["Tiendas_Plan_Viaje"].fillna(0).astype(int)
    dq_trips["Tiendas_OK_Viaje"] = dq_trips["Tiendas_OK_Viaje"].fillna(0).astype(int)
    dq_trips["Viaje_Completo_Base"] = dq_trips["CD_Origen_OK"] & dq_trips["Todas_Tiendas_OK"]

    # Clasificar Tipo de Viaje segun cantidad de tiendas planificadas en Id Viaje
    dq_trips["Tipo_Viaje"] = np.select(
        [
            dq_trips["Tiendas_Plan_Viaje"] == 1,
            dq_trips["Tiendas_Plan_Viaje"] == 2,
            dq_trips["Tiendas_Plan_Viaje"] >= 3,
        ],
        [
            "Directo (1 tienda)",
            "Compartido (2 tiendas)",
            "Rutas (3+ tiendas)",
        ],
        default="Sin tiendas",
    )

    trip_type_lut = dq_trips.set_index("Id Viaje")[["Tiendas_Plan_Viaje", "Tipo_Viaje"]]

    # Subtabla de Paradas en Tienda
    dq_stores = u_clean_num[u_clean_num["Tipo de Zona"] == "Tienda"][
        [
            "Operacion",
            "Fecha_Reparto",
            "Hora Reparto",
            "Mes_Label",
            "Semana_Sort",
            "Semana_Label",
            "Id Viaje",
            "Ref_Clean",
            "Transportista",
            "Dominio",
            "Interno",
            "Tipo de Camion",
            "Tipo_Vehiculo",
            "Satelital",
            "Tienda_Cod",
            "Orden Visita Tienda",
            "Categoria",
            "Inicio Horario Visita Real",
            "Fecha Hora Salida",
            "dt_ini_real",
            "dt_fin_real",
            "prev_dt_fin_real",
            "Tiempo de Estadia",
            "Tiempo en Viaje",
            "KMS Parada_num",
            "Velocidad Real Calculada_num",
            "Cantidad Visitas",
            "Total Bultos_num",
            "Formato Equivalentes_num",
            "Apertura Puerta Trasera_num",
            "Cierre Puerta Trasera_num",
            "Minutos Apertura y Cierre_num",
            "visita_efectiva",
        ]
    ].copy()
    dq_stores["Tiendas_Plan_Viaje"] = dq_stores["Id Viaje"].map(trip_type_lut["Tiendas_Plan_Viaje"]).fillna(1).astype(int)
    dq_stores["Tipo_Viaje"] = dq_stores["Id Viaje"].map(trip_type_lut["Tipo_Viaje"]).fillna("Directo (1 tienda)")

    dq_cd_orig = dq_trips[dq_trips["CD_Origen_OK"]].copy()

    return {
        "master_trips": master_trips,
        "dup_detail": dup_detail,
        "invalid_detail": invalid_detail,
        "stores_gps": stores_gps,
        "dq_stores": dq_stores,
        "dq_cd_orig": dq_cd_orig,
        "dq_trips": dq_trips,
    }


def build_audit_excel(df_gap, df_solo_tms, df_store_diff, df_dup, df_inv) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_gap.to_excel(writer, sheet_name="1_Viajes_Solo_WMS_Gap", index=False)
        df_solo_tms.to_excel(writer, sheet_name="2_Viajes_Solo_TMS", index=False)
        df_store_diff.to_excel(writer, sheet_name="3_Diferencias_Tiendas", index=False)
        df_dup.to_excel(writer, sheet_name="4_Alerta_Ref_Duplicadas", index=False)
        df_inv.to_excel(writer, sheet_name="5_Alerta_Ref_Invalidas", index=False)
    return output.getvalue()


def build_gps_audit_excel(df_geo, df_fleet, df_sin_coord, df_raw_fails) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_geo.to_excel(writer, sheet_name="1_Tiendas_Geocerca", index=False)
        df_fleet.to_excel(writer, sheet_name="2_Unidades_Viajes_Ciegos", index=False)
        df_sin_coord.to_excel(writer, sheet_name="3_Tiendas_Sin_Coordenadas", index=False)
        df_raw_fails.to_excel(writer, sheet_name="4_Detalle_Paradas_Sin_GPS", index=False)
    return output.getvalue()


def build_estadias_excel(df_seg, df_mat, df_rank, df_clean_st) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_seg.to_excel(writer, sheet_name="1_Resumen_Segmentos", index=False)
        df_mat.to_excel(writer, sheet_name="2_Matriz_CD_x_Vehiculo", index=False)
        df_rank.to_excel(writer, sheet_name="3_Ranking_Tiendas", index=False)
        df_clean_st.to_excel(writer, sheet_name="4_Base_Limpia_Estadias", index=False)
    return output.getvalue()


# -----------------------------------------------------------------------------
# CARGA AUTOMATICA DE DATOS (SILENCIOSA PARA PRIORIZAR FILTROS EN SIDEBAR)
# -----------------------------------------------------------------------------
default_drive_url = ""
try:
    default_drive_url = st.secrets.get("DRIVE_FOLDER_URL", "")
except Exception:
    default_drive_url = ""

repo_a, repo_u, repo_n = find_repo_excel_files()
has_repo_files = repo_a is not None and repo_u is not None

amrepa_bytes = None
unigis_bytes = None
nomina_bytes = None
fuente_label = ""

# Prioridad 1: Base persistente en GitHub | Prioridad 2: Google Drive Secret | Prioridad 3: Manual
if has_repo_files:
    with open(repo_a, "rb") as fa:
        amrepa_bytes = fa.read()
    with open(repo_u, "rb") as fu:
        unigis_bytes = fu.read()
    fuente_label = f"`{os.path.basename(repo_a)}` + `{os.path.basename(repo_u)}`"
    if repo_n is not None:
        with open(repo_n, "rb") as fn:
            nomina_bytes = fn.read()
        fuente_label += f" + `{os.path.basename(repo_n)}`"
elif default_drive_url:
    try:
        amrepa_bytes, unigis_bytes, nomina_bytes, name_a, name_u, name_n = fetch_from_drive_folder(default_drive_url)
        fuente_label = f"Drive (`{name_a}` + `{name_u}`" + (f" + `{name_n}`)" if name_n else ")")
    except Exception:
        pass

# Si se subieron archivos manuales en el expander inferior, toman prioridad temporal
if "manual_a" in st.session_state and st.session_state["manual_a"] is not None:
    amrepa_bytes = st.session_state["manual_a"]
if "manual_u" in st.session_state and st.session_state["manual_u"] is not None:
    unigis_bytes = st.session_state["manual_u"]
if "manual_n" in st.session_state and st.session_state["manual_n"] is not None:
    nomina_bytes = st.session_state["manual_n"]

if amrepa_bytes is None or unigis_bytes is None:
    st.title("🚚 Estadística de Camión")
    st.warning("No se detectaron archivos base. Cargalos desde el menú lateral.")
    up_a = st.sidebar.file_uploader("Archivo WMS (AMREPA.xlsx)", type=["xlsx"])
    up_u = st.sidebar.file_uploader("Reporte TMS (UNIGIS Detalle.xlsx)", type=["xlsx"])
    up_n = st.sidebar.file_uploader("Nómina Flota (Nomina.xlsx - opcional)", type=["xlsx"])
    if up_a and up_u:
        st.session_state["manual_a"] = up_a.getvalue()
        st.session_state["manual_u"] = up_u.getvalue()
        if up_n:
            st.session_state["manual_n"] = up_n.getvalue()
        st.rerun()
    st.stop()

data = load_and_process_data(amrepa_bytes, unigis_bytes, nomina_bytes)
master = data["master_trips"].copy()
dup_detail = data["dup_detail"].copy()
invalid_detail = data["invalid_detail"].copy()
stores_gps = data["stores_gps"].copy()
dq_stores = data["dq_stores"].copy()
dq_cd_orig = data["dq_cd_orig"].copy()
dq_trips = data["dq_trips"].copy()

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

mask_gps = (
    (stores_gps["Fecha_Reparto"] >= d_start)
    & (stores_gps["Fecha_Reparto"] <= d_end)
    & (stores_gps["Operacion"].isin(selected_ops))
)
if selected_months:
    mask_gps = mask_gps & (stores_gps["Mes_Label"].isin(selected_months))
if selected_weeks:
    mask_gps = mask_gps & (stores_gps["Semana_Label"].isin(selected_weeks))
if selected_trans:
    mask_gps = mask_gps & (stores_gps["Transportista"].isin(selected_trans))

gps_filt = stores_gps[mask_gps].copy()


def _apply_global_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    m = (
        (df_in["Fecha_Reparto"] >= d_start)
        & (df_in["Fecha_Reparto"] <= d_end)
        & (df_in["Operacion"].isin(selected_ops))
    )
    if selected_months:
        m = m & (df_in["Mes_Label"].isin(selected_months))
    if selected_weeks:
        m = m & (df_in["Semana_Label"].isin(selected_weeks))
    if selected_trans:
        m = m & (df_in["Transportista"].isin(selected_trans))
    return df_in[m].copy()


dq_stores_filt = _apply_global_filters(dq_stores)
dq_cd_filt = _apply_global_filters(dq_cd_orig)
dq_trips_filt = _apply_global_filters(dq_trips)

# Seccion colapsada al final del sidebar para carga manual / info de base
st.sidebar.divider()
with st.sidebar.expander("⚙️ Base de Datos / Carga Manual", expanded=False):
    st.caption(f"Fuente activa: {fuente_label}")
    up_a_alt = st.file_uploader("Reemplazar AMREPA (.xlsx)", type=["xlsx"], key="up_a_alt")
    up_u_alt = st.file_uploader("Reemplazar UNIGIS (.xlsx)", type=["xlsx"], key="up_u_alt")
    up_n_alt = st.file_uploader("Cargar / Reemplazar Nómina Flota (.xlsx)", type=["xlsx"], key="up_n_alt")
    if up_a_alt is not None:
        st.session_state["manual_a"] = up_a_alt.getvalue()
    if up_u_alt is not None:
        st.session_state["manual_u"] = up_u_alt.getvalue()
    if up_n_alt is not None:
        st.session_state["manual_n"] = up_n_alt.getvalue()
    if st.button("Restaurar base original"):
        st.session_state["manual_a"] = None
        st.session_state["manual_u"] = None
        st.session_state["manual_n"] = None
        st.rerun()

# -----------------------------------------------------------------------------
# CABECERA PRINCIPAL Y NAVEGACION POR SOLAPAS
# -----------------------------------------------------------------------------
st.title("🚚 Estadística de Camión")
st.caption(
    f"Período activo: **{d_start.strftime('%d/%m/%Y')}** al **{d_end.strftime('%d/%m/%Y')}** | "
    f"CDs: **{', '.join(map(str, selected_ops))}** | Todos los filtros laterales actualizan todas las vistas."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 1. Conciliación Macro (WMS vs. TMS)",
        "📍 2. Visitas GPS y Geocercas",
        "🏪 3. Estadías y Productividad en Tienda",
        "🛣️ 4. Velocidades (Parametrización UNIGIS)",
        "🚛 5. Internos, CD Origen y Jornada Diaria",
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
        v_solo_tms = int((sub["_merge"] == "right_only").sum())
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
                "Gap WMS": v_gap,
                "Solo TMS": v_solo_tms,
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
                    "Gap WMS": df_resumen_op["Gap WMS"].sum(),
                    "Solo TMS": df_resumen_op["Solo TMS"].sum(),
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
                "Gap WMS": st.column_config.NumberColumn("Gap WMS", format="%d", width="small"),
                "Solo TMS": st.column_config.NumberColumn("Solo TMS", format="%d", width="small"),
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

    df_solo_tms_export = (
        df_filt[df_filt["_merge"] == "right_only"][
            ["Operacion", "Fecha_Base", "Ref_Clean", "Ids_Viaje", "Dominio", "Tipo_Camion", "Tiendas_TMS", "Set_Tiendas_TMS"]
        ]
        .copy()
        .sort_values(["Operacion", "Fecha_Base", "Ref_Clean"])
    )
    df_solo_tms_export["Fecha Reparto"] = df_solo_tms_export["Fecha_Base"].dt.strftime("%d/%m/%Y").fillna("")
    df_solo_tms_export["Tiendas en TMS"] = df_solo_tms_export["Set_Tiendas_TMS"].apply(lambda x: ", ".join(x))
    df_solo_tms_export = df_solo_tms_export[
        ["Operacion", "Fecha Reparto", "Ref_Clean", "Ids_Viaje", "Dominio", "Tipo_Camion", "Tiendas_TMS", "Tiendas en TMS"]
    ].rename(
        columns={
            "Ref_Clean": "Ref Externa Viaje",
            "Ids_Viaje": "Id Viaje UNIGIS",
            "Tipo_Camion": "Tipo Camión",
            "Tiendas_TMS": "Cant. Tiendas TMS",
        }
    )

    excel_bytes = build_audit_excel(
        df_gap_export, df_solo_tms_export, df_store_diff_export, dup_display, inv_display
    )
    st.download_button(
        label="📥 Descargar Auditoría Completa en Excel (.xlsx)",
        data=excel_bytes,
        file_name="Auditoria_Conciliacion_WMS_TMS.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    sub_t1, sub_t_solo_tms, sub_t2, sub_t3 = st.tabs(
        [
            f"🚨 Viajes Faltantes en TMS ({len(df_gap_export)})",
            f"🚛 Viajes Solo en TMS ({len(df_solo_tms_export)})",
            f"🏪 Diferencias de Tiendas ({len(df_store_diff_export)})",
            f"⚠️ Alertas Calidad UNIGIS ({cant_dup_refs} dup | {cant_inv_trips} inv.)",
        ]
    )

    with sub_t1:
        st.caption("Viajes despachados en AMREPA (WMS) que no ingresaron o no matchearon en UNIGIS.")
        st.dataframe(df_gap_export, use_container_width=True, hide_index=True)

    with sub_t_solo_tms:
        st.caption("Viajes con referencia válida presentes en UNIGIS (TMS) que no figuran en el corte de AMREPA (WMS).")
        st.dataframe(df_solo_tms_export, use_container_width=True, hide_index=True)

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
    st.subheader("Auditoría de Visitas GPS y Geocercas (Paradas en Tienda TMS)")

    tot_paradas = len(gps_filt)
    paradas_ok = int((gps_filt["Diagnostico_GPS"] == "Visitada (OK)").sum())
    paradas_ciego = int((gps_filt["Diagnostico_GPS"] == "Viaje Ciego (GPS/Unidad)").sum())
    paradas_geo = int((gps_filt["Diagnostico_GPS"] == "Falla Geocerca (Parcial)").sum())
    paradas_sin_coord = int((gps_filt["Diagnostico_GPS"] == "Sin coordenadas").sum())
    paradas_sin_dato = paradas_ciego + paradas_geo + paradas_sin_coord

    viajes_gps_tot = int(gps_filt["Id Viaje"].nunique())
    viajes_ciegos_cnt = int(
        gps_filt[gps_filt["Diagnostico_GPS"] == "Viaje Ciego (GPS/Unidad)"]["Id Viaje"].nunique()
    )
    viajes_100_ok_cnt = int(
        gps_filt.groupby("Id Viaje")["Diagnostico_GPS"]
        .apply(lambda s: int((s == "Visitada (OK)").sum()) == len(s))
        .sum()
    )

    act_tarde_df = gps_filt[gps_filt["obs_act_tarde"]]
    viajes_act_tarde_cnt = int(act_tarde_df["Id Viaje"].nunique())
    viajes_act_tarde_ok_cnt = int(
        act_tarde_df.groupby("Id Viaje")["Diagnostico_GPS"]
        .apply(lambda s: int((s == "Visitada (OK)").sum()) == len(s))
        .sum()
    )
    viajes_act_tarde_fail_cnt = viajes_act_tarde_cnt - viajes_act_tarde_ok_cnt
    pct_viajes_act_tarde = (viajes_act_tarde_cnt / viajes_gps_tot * 100) if viajes_gps_tot > 0 else 0.0

    geo_only_df = gps_filt[gps_filt["Diagnostico_GPS"] == "Falla Geocerca (Parcial)"]
    paradas_geo_act_ok = int((~geo_only_df["obs_act_tarde"]).sum())
    paradas_geo_act_tarde = int(geo_only_df["obs_act_tarde"].sum())

    pct_efec_gps = (paradas_ok / tot_paradas * 100) if tot_paradas > 0 else 0.0
    pct_ciego = (paradas_ciego / tot_paradas * 100) if tot_paradas > 0 else 0.0
    pct_geo = (paradas_geo / tot_paradas * 100) if tot_paradas > 0 else 0.0

    g1, g2, g3, g4, g5 = st.columns([1.0, 1.45, 1.25, 1.30, 1.20])
    with g1:
        st.markdown(
            _kpi_card_html(
                "Paradas Planificadas",
                _fmt_int(tot_paradas),
                f"En {_fmt_int(viajes_gps_tot)} viajes TMS",
                "#475569",
                f"<b>{_fmt_int(viajes_100_ok_cnt)}</b> viajes con 100% GPS",
            ),
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            _kpi_card_html(
                "% Efectividad GPS (Paradas)",
                _fmt_pct(pct_efec_gps),
                f"{_fmt_int(paradas_ok)} visitas OK | {_fmt_int(paradas_sin_dato)} sin dato:",
                "#16a34a" if pct_efec_gps >= 90 else "#d97706",
                f"• <b>{_fmt_int(paradas_ciego)}</b> viaje ciego | <b>{_fmt_int(paradas_geo)}</b> geozona | <b>{_fmt_int(paradas_sin_coord)}</b> s/coord.",
            ),
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            _kpi_card_html(
                "Falla GPS / Unidad (Paradas)",
                _fmt_int(paradas_ciego),
                f"{_fmt_pct(pct_ciego)} del total de paradas",
                "#dc2626" if paradas_ciego > 0 else "#16a34a",
                f"En <b>{_fmt_int(viajes_ciegos_cnt)}</b> viajes con 0% tiendas GPS",
            ),
            unsafe_allow_html=True,
        )
    with g4:
        st.markdown(
            _kpi_card_html(
                "Falla Geozona (Paradas)",
                _fmt_int(paradas_geo),
                f"{_fmt_pct(pct_geo)} del total de paradas",
                "#d97706" if paradas_geo > 0 else "#16a34a",
                f"<b>{_fmt_int(paradas_geo_act_ok)}</b> activ. OK | <b>{_fmt_int(paradas_geo_act_tarde)}</b> activ. tarde",
            ),
            unsafe_allow_html=True,
        )
    with g5:
        st.markdown(
            _kpi_card_html(
                "Activación Tarde (Viajes)",
                _fmt_int(viajes_act_tarde_cnt),
                f"{_fmt_pct(pct_viajes_act_tarde)} de viajes (>5 km CD)",
                "#475569",
                f"<b>{_fmt_int(viajes_act_tarde_ok_cnt)}</b> con visitas OK | <b>{_fmt_int(viajes_act_tarde_fail_cnt)}</b> c/ pérdida",
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE 1: EFECTIVIDAD GPS POR OPERACION (CD)
    # -------------------------------------------------------------------------
    st.markdown("### 1. Efectividad GPS, Geozonas e Interfaz por Operación (CD)")

    rows_gps_op = []
    for op in sorted(gps_filt["Operacion"].dropna().unique()):
        sub = gps_filt[gps_filt["Operacion"] == op]
        v_tot = int(sub["Id Viaje"].nunique())
        v_ciegos = int(sub[sub["Diagnostico_GPS"] == "Viaje Ciego (GPS/Unidad)"]["Id Viaje"].nunique())
        v_act = int(sub[sub["obs_act_tarde"]]["Id Viaje"].nunique())
        v_err = int(sub[sub["obs_err_fin"]]["Id Viaje"].nunique())

        p_tot = len(sub)
        p_ok = int((sub["Diagnostico_GPS"] == "Visitada (OK)").sum())
        p_ciego = int((sub["Diagnostico_GPS"] == "Viaje Ciego (GPS/Unidad)").sum())
        p_geo = int((sub["Diagnostico_GPS"] == "Falla Geocerca (Parcial)").sum())
        p_sc = int((sub["Diagnostico_GPS"] == "Sin coordenadas").sum())
        efec = (p_ok / p_tot * 100) if p_tot > 0 else 0.0

        rows_gps_op.append(
            {
                "CD": str(int(op)),
                "Viajes TMS": v_tot,
                "Viajes Ciegos": v_ciegos,
                "Activ. Tarde": v_act,
                "Error Cierre": v_err,
                "Paradas Plan.": p_tot,
                "Visitadas OK": p_ok,
                "Falla GPS": p_ciego,
                "Falla Geozona": p_geo,
                "Sin Coord.": p_sc,
                "% Efectividad GPS": round(efec, 2),
            }
        )

    df_gps_op = pd.DataFrame(rows_gps_op)
    if not df_gps_op.empty:
        tot_gps_row = pd.DataFrame(
            [
                {
                    "CD": "TOTAL",
                    "Viajes TMS": df_gps_op["Viajes TMS"].sum(),
                    "Viajes Ciegos": df_gps_op["Viajes Ciegos"].sum(),
                    "Activ. Tarde": df_gps_op["Activ. Tarde"].sum(),
                    "Error Cierre": df_gps_op["Error Cierre"].sum(),
                    "Paradas Plan.": df_gps_op["Paradas Plan."].sum(),
                    "Visitadas OK": df_gps_op["Visitadas OK"].sum(),
                    "Falla GPS": df_gps_op["Falla GPS"].sum(),
                    "Falla Geozona": df_gps_op["Falla Geozona"].sum(),
                    "Sin Coord.": df_gps_op["Sin Coord."].sum(),
                    "% Efectividad GPS": round(pct_efec_gps, 2),
                }
            ]
        )
        df_gps_op_disp = pd.concat([df_gps_op, tot_gps_row], ignore_index=True)

        st.dataframe(
            df_gps_op_disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "CD": st.column_config.TextColumn("CD", width="small"),
                "Viajes TMS": st.column_config.NumberColumn("Viajes TMS", format="%d", width="small"),
                "Viajes Ciegos": st.column_config.NumberColumn("Viajes Ciegos", format="%d", width="small"),
                "Activ. Tarde": st.column_config.NumberColumn("Activ. Tarde", format="%d", width="small"),
                "Error Cierre": st.column_config.NumberColumn("Error Cierre", format="%d", width="small"),
                "Paradas Plan.": st.column_config.NumberColumn("Paradas Plan.", format="%d", width="small"),
                "Visitadas OK": st.column_config.NumberColumn("Visitadas OK", format="%d", width="small"),
                "Falla GPS": st.column_config.NumberColumn("Falla GPS", format="%d", width="small"),
                "Falla Geozona": st.column_config.NumberColumn("Falla Geozona", format="%d", width="small"),
                "Sin Coord.": st.column_config.NumberColumn("Sin Coord.", format="%d", width="small"),
                "% Efectividad GPS": st.column_config.ProgressColumn(
                    "% Efectividad GPS",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                    width="medium",
                ),
            },
        )

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE 2: EVOLUCION TEMPORAL DE EFECTIVIDAD GPS
    # -------------------------------------------------------------------------
    st.markdown("### 2. Evolución Temporal de Efectividad GPS y Fallas")

    gctrl1, gctrl2 = st.columns([1, 1.5])
    with gctrl1:
        gran_gps = st.radio(
            "Agrupar eje temporal por:",
            options=["Día", "Semana", "Mes"],
            index=1,
            horizontal=True,
            key="gran_gps",
        )
    with gctrl2:
        modo_gps = st.radio(
            "Modo de visualización:",
            options=[
                "Consolidado (Barras por Causa + Línea %)",
                "Abierto por Almacén (Mapa de Calor %)",
            ],
            index=0,
            horizontal=True,
            key="modo_gps",
        )

    df_gps_time = gps_filt.copy()

    if gran_gps == "Día" and not df_gps_time.empty:
        sems_gps_disp = (
            df_gps_time[["Semana_Sort", "Semana_Label"]]
            .drop_duplicates()
            .sort_values("Semana_Sort")["Semana_Label"]
            .tolist()
        )
        def_sems_gps = sems_gps_disp[-2:] if len(sems_gps_disp) >= 2 else sems_gps_disp

        gd1, gd2 = st.columns([2, 1])
        with gd1:
            sems_gps_sel = st.multiselect(
                "Drill-down — Seleccioná semana(s) para abrir por día (por defecto últimas 2):",
                options=sems_gps_disp,
                default=def_sems_gps,
                placeholder="Todas las semanas",
                key="sems_gps_sel",
            )
        with gd2:
            st.write("")
            exc_guardias_gps = st.checkbox(
                "Ocultar domingos/feriados (<50 paradas)",
                value=False,
                key="exc_guardias_gps",
            )

        if sems_gps_sel:
            df_gps_time = df_gps_time[df_gps_time["Semana_Label"].isin(sems_gps_sel)].copy()
        if exc_guardias_gps and not df_gps_time.empty:
            d_cnts = df_gps_time.groupby(df_gps_time["Fecha_Reparto"].dt.floor("D"))["Id Viaje"].transform("count")
            df_gps_time = df_gps_time[d_cnts >= 50].copy()

    if gran_gps == "Día":
        df_gps_time["Periodo_Sort"] = df_gps_time["Fecha_Reparto"].dt.floor("D")
        df_gps_time["Periodo"] = (
            df_gps_time["Periodo_Sort"].dt.dayofweek.map(dias_es)
            + " "
            + df_gps_time["Periodo_Sort"].dt.strftime("%d/%m")
        )
    elif gran_gps == "Semana":
        df_gps_time["Periodo_Sort"] = df_gps_time["Semana_Sort"]
        df_gps_time["Periodo"] = df_gps_time["Semana_Label"]
    else:
        df_gps_time["Periodo_Sort"] = df_gps_time["Fecha_Reparto"].dt.to_period("M").apply(lambda p: p.start_time)
        df_gps_time["Periodo"] = df_gps_time["Periodo_Sort"].dt.strftime("%Y-%m")

    if df_gps_time.empty:
        st.info("No hay paradas registradas en el rango y filtros seleccionados.")
    elif modo_gps == "Consolidado (Barras por Causa + Línea %)":
        agg_gt = (
            df_gps_time.groupby(["Periodo_Sort", "Periodo"], as_index=False)
            .agg(
                Total=("Diagnostico_GPS", "count"),
                OK=("Diagnostico_GPS", lambda s: int((s == "Visitada (OK)").sum())),
                Geocerca=("Diagnostico_GPS", lambda s: int((s == "Falla Geocerca (Parcial)").sum())),
                Ciego=("Diagnostico_GPS", lambda s: int((s == "Viaje Ciego (GPS/Unidad)").sum())),
                SinCoord=("Diagnostico_GPS", lambda s: int((s == "Sin coordenadas").sum())),
            )
            .sort_values("Periodo_Sort")
        )
        agg_gt["Pct_OK"] = np.where(
            agg_gt["Total"] > 0,
            (agg_gt["OK"] / agg_gt["Total"] * 100).round(2),
            0.0,
        )

        n_pts_g = len(agg_gt)
        show_lbl_g = n_pts_g <= 16

        fig_g = make_subplots(specs=[[{"secondary_y": True}]])
        fig_g.add_trace(
            go.Bar(
                x=agg_gt["Periodo"],
                y=agg_gt["OK"],
                name="Visitadas OK",
                marker_color="#1d4ed8",
                hovertemplate="<b>%{x}</b><br>Visitadas OK: %{y:,}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig_g.add_trace(
            go.Bar(
                x=agg_gt["Periodo"],
                y=agg_gt["Geocerca"],
                name="Falla Geocerca (Parcial)",
                marker_color="#f59e0b",
                hovertemplate="<b>%{x}</b><br>Falla Geocerca: %{y:,}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig_g.add_trace(
            go.Bar(
                x=agg_gt["Periodo"],
                y=agg_gt["Ciego"],
                name="Viaje Ciego (GPS/Unidad)",
                marker_color="#dc2626",
                hovertemplate="<b>%{x}</b><br>Viaje Ciego: %{y:,}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig_g.add_trace(
            go.Bar(
                x=agg_gt["Periodo"],
                y=agg_gt["SinCoord"],
                name="Sin Coordenadas",
                marker_color="#64748b",
                hovertemplate="<b>%{x}</b><br>Sin Coordenadas: %{y:,}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig_g.add_trace(
            go.Scatter(
                x=agg_gt["Periodo"],
                y=agg_gt["Pct_OK"],
                name="% Efectividad GPS",
                mode="lines+markers",
                line=dict(color="#16a34a", width=3.5),
                marker=dict(
                    size=10 if show_lbl_g else 7,
                    color="#16a34a",
                    line=dict(width=2, color="#0f172a"),
                ),
                hovertemplate="<b>%{x}</b><br>Efectividad GPS: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=True,
        )

        if show_lbl_g:
            for rt in agg_gt.itertuples(index=False):
                fig_g.add_annotation(
                    x=rt.Periodo,
                    y=rt.Pct_OK,
                    yref="y2",
                    text=f"<b>{rt.Pct_OK:.1f}%</b>",
                    showarrow=False,
                    yshift=18,
                    font=dict(color="#0f172a", size=11),
                    bgcolor="rgba(255, 255, 255, 0.92)",
                    bordercolor="#16a34a",
                    borderwidth=1.2,
                    borderpad=3,
                )

        fig_g.add_shape(
            type="line",
            xref="paper",
            x0=0,
            x1=1,
            yref="y2",
            y0=95,
            y1=95,
            line=dict(color="#334155", width=1.5, dash="dot"),
        )
        fig_g.add_annotation(
            xref="paper",
            x=0.99,
            yref="y2",
            y=95,
            text="Objetivo 95%",
            showarrow=False,
            yshift=-12,
            xanchor="right",
            font=dict(color="#334155", size=11),
            bgcolor="rgba(255, 255, 255, 0.85)",
        )

        fig_g.update_layout(
            barmode="stack",
            height=450 if gran_gps == "Día" else 420,
            margin=dict(l=20, r=20, t=35, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            xaxis=dict(
                rangeslider=dict(visible=(gran_gps == "Día" and n_pts_g > 14), thickness=0.06),
                tickangle=-35 if n_pts_g > 14 else 0,
            ),
        )
        fig_g.update_yaxes(title_text="Cantidad de Paradas", secondary_y=False)
        fig_g.update_yaxes(
            title_text="% Efectividad GPS",
            range=[0, 114],
            dtick=20,
            showgrid=False,
            secondary_y=True,
        )
        st.plotly_chart(fig_g, use_container_width=True)

    else:
        agg_gop = (
            df_gps_time.groupby(["Operacion", "Periodo_Sort", "Periodo"], as_index=False)
            .agg(
                Total=("Diagnostico_GPS", "count"),
                OK=("Diagnostico_GPS", lambda s: int((s == "Visitada (OK)").sum())),
                Fallas=("Diagnostico_GPS", lambda s: int((s != "Visitada (OK)").sum())),
            )
            .sort_values(["Operacion", "Periodo_Sort"])
        )
        agg_gop["Pct_OK"] = np.where(
            agg_gop["Total"] > 0,
            (agg_gop["OK"] / agg_gop["Total"] * 100).round(1),
            np.nan,
        )
        agg_gop["Operacion_Str"] = "CD " + agg_gop["Operacion"].astype(str)

        ord_per_g = (
            agg_gop[["Periodo_Sort", "Periodo"]]
            .drop_duplicates()
            .sort_values("Periodo_Sort")["Periodo"]
            .tolist()
        )
        ord_ops_g = sorted(agg_gop["Operacion_Str"].unique().tolist())

        piv_g_pct = agg_gop.pivot(index="Operacion_Str", columns="Periodo", values="Pct_OK").reindex(
            index=ord_ops_g, columns=ord_per_g
        )
        piv_g_tot = agg_gop.pivot(index="Operacion_Str", columns="Periodo", values="Total").reindex(
            index=ord_ops_g, columns=ord_per_g
        )
        piv_g_fal = agg_gop.pivot(index="Operacion_Str", columns="Periodo", values="Fallas").reindex(
            index=ord_ops_g, columns=ord_per_g
        )

        show_txt_hm_g = len(ord_per_g) <= 22
        txt_mat_g = np.where(
            piv_g_pct.notna(),
            piv_g_pct.map(lambda v: f"{v:.1f}%" if pd.notna(v) else ""),
            "S/D",
        )
        cdata_g = np.dstack([piv_g_tot.fillna(0).values, piv_g_fal.fillna(0).values])

        fig_hm_g = go.Figure(
            data=go.Heatmap(
                z=piv_g_pct.values,
                x=ord_per_g,
                y=ord_ops_g,
                text=txt_mat_g if show_txt_hm_g else None,
                texttemplate="%{text}" if show_txt_hm_g else None,
                textfont=dict(size=11),
                xgap=2,
                ygap=2,
                colorscale="RdYlGn",
                zmin=50,
                zmax=100,
                customdata=cdata_g,
                hovertemplate=(
                    "<b>%{y} — %{x}</b><br>"
                    "Efectividad GPS: %{z:.1f}%<br>"
                    "Paradas Plan.: %{customdata[0]:,.0f}<br>"
                    "Sin Dato GPS: %{customdata[1]:,.0f}<extra></extra>"
                ),
                colorbar=dict(title="% GPS"),
            )
        )
        fig_hm_g.update_layout(
            height=390,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title=f"Período ({gran_gps})",
            yaxis_title="Almacén / Operación",
            xaxis=dict(tickangle=-35 if len(ord_per_g) > 14 else 0),
        )
        st.plotly_chart(fig_hm_g, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE 3: RANKINGS ACCIONABLES (GEOCERCAS, UNIDADES CIEGAS, SIN COORD.)
    # -------------------------------------------------------------------------
    st.markdown("### 3. Diagnóstico Accionable: Geocercas vs. Unidades Sin Señal")

    # 3A. Ranking de Tiendas por Falla de Geozona (con apertura de Activacion Tarde)
    gps_tiendas_only = gps_filt[gps_filt["Tipo de Zona"] == "Tienda"].copy()
    gps_tiendas_only["_geo_ok_act"] = (
        (gps_tiendas_only["Diagnostico_GPS"] == "Falla Geocerca (Parcial)")
        & (~gps_tiendas_only["obs_act_tarde"])
    )
    gps_tiendas_only["_geo_tarde_act"] = (
        (gps_tiendas_only["Diagnostico_GPS"] == "Falla Geocerca (Parcial)")
        & (gps_tiendas_only["obs_act_tarde"])
    )

    st_rank_all = (
        gps_tiendas_only.groupby(["Operacion", "Tienda_Cod"], as_index=False)
        .agg(
            Paradas_Plan=("Diagnostico_GPS", "count"),
            Visitadas_OK=("Diagnostico_GPS", lambda s: int((s == "Visitada (OK)").sum())),
            Falla_Geozona_Pura=("_geo_ok_act", "sum"),
            Falla_Con_Act_Tarde=("_geo_tarde_act", "sum"),
            Falla_Geocerca=("Diagnostico_GPS", lambda s: int((s == "Falla Geocerca (Parcial)").sum())),
            Falla_Viaje_Ciego=("Diagnostico_GPS", lambda s: int((s == "Viaje Ciego (GPS/Unidad)").sum())),
            Reentradas_Geocerca=("Cantidad Visitas", lambda s: int((s > 1).sum())),
        )
    )
    st_rank_all["% Falla Geozona (Activ. OK)"] = np.where(
        st_rank_all["Paradas_Plan"] > 0,
        (st_rank_all["Falla_Geozona_Pura"] / st_rank_all["Paradas_Plan"] * 100).round(1),
        0.0,
    )
    st_rank_all["% Efectividad OK"] = np.where(
        st_rank_all["Paradas_Plan"] > 0,
        (st_rank_all["Visitadas_OK"] / st_rank_all["Paradas_Plan"] * 100).round(1),
        0.0,
    )
    st_rank_all = st_rank_all.sort_values(
        ["Falla_Geozona_Pura", "% Falla Geozona (Activ. OK)", "Paradas_Plan"],
        ascending=[False, False, False],
    ).rename(
        columns={
            "Operacion": "CD",
            "Tienda_Cod": "Tienda",
            "Paradas_Plan": "Paradas Plan.",
            "Visitadas_OK": "Visitadas OK",
            "Falla_Geozona_Pura": "Falla Geozona (Activ. OK)",
            "Falla_Con_Act_Tarde": "Falla c/ Activ. Tarde",
            "Falla_Geocerca": "Falla Parcial Total",
            "Falla_Viaje_Ciego": "Falla Viaje Ciego",
            "Reentradas_Geocerca": "Re-entradas (>1)",
        }
    )

    # 3B. Ranking de Unidades (Dominio) por Viajes Ciegos (con Tipo Vehiculo y Satelital de Nomina)
    dom_trip_lvl = (
        gps_filt.groupby(
            ["Operacion", "Dominio", "Interno", "Tipo_Vehiculo", "Satelital", "Transportista", "Id Viaje"],
            as_index=False,
        )
        .agg(
            Paradas=("Diagnostico_GPS", "count"),
            Visitadas=("Diagnostico_GPS", lambda s: int((s == "Visitada (OK)").sum())),
            Es_Ciego=("Visitadas_En_IdViaje", lambda s: int(s.iloc[0] == 0)),
        )
    )
    dom_rank_all = (
        dom_trip_lvl.groupby(
            ["Operacion", "Dominio", "Interno", "Tipo_Vehiculo", "Satelital", "Transportista"],
            as_index=False,
        )
        .agg(
            Viajes_TMS=("Id Viaje", "nunique"),
            Viajes_Ciegos=("Es_Ciego", "sum"),
            Paradas_Plan=("Paradas", "sum"),
            Paradas_OK=("Visitadas", "sum"),
        )
    )
    dom_rank_all["% Viajes Ciegos"] = np.where(
        dom_rank_all["Viajes_TMS"] > 0,
        (dom_rank_all["Viajes_Ciegos"] / dom_rank_all["Viajes_TMS"] * 100).round(1),
        0.0,
    )
    dom_rank_all["% Efectividad GPS"] = np.where(
        dom_rank_all["Paradas_Plan"] > 0,
        (dom_rank_all["Paradas_OK"] / dom_rank_all["Paradas_Plan"] * 100).round(1),
        0.0,
    )
    dom_rank_all = dom_rank_all.sort_values(
        ["Viajes_Ciegos", "% Viajes Ciegos", "Viajes_TMS"], ascending=[False, False, False]
    ).rename(
        columns={
            "Operacion": "CD",
            "Tipo_Vehiculo": "Tipo Vehículo",
            "Viajes_TMS": "Viajes TMS",
            "Viajes_Ciegos": "Viajes Ciegos",
            "Paradas_Plan": "Paradas Plan.",
            "Paradas_OK": "Paradas OK",
        }
    )

    # 3C. Tiendas Sin Coordenadas (agrupado por Tienda única para coincidir 1:1 con KPI)
    sin_coord_tbl = (
        gps_filt[gps_filt["Diagnostico_GPS"] == "Sin coordenadas"]
        .groupby("Tienda_Cod", as_index=False)
        .agg(
            CDs=("Operacion", lambda s: " / ".join(sorted(s.dropna().astype(int).astype(str).unique()))),
            Paradas_Afectadas=("Id Viaje", "count"),
            Viajes_Unicos=("Ref_Clean", "nunique"),
            Primera_Fecha=("Fecha_Reparto", "min"),
            Ultima_Fecha=("Fecha_Reparto", "max"),
        )
        .sort_values(["Paradas_Afectadas", "Tienda_Cod"], ascending=[False, True])
    )
    if not sin_coord_tbl.empty:
        sin_coord_tbl["Primera Fecha"] = sin_coord_tbl["Primera_Fecha"].dt.strftime("%d/%m/%Y").fillna("")
        sin_coord_tbl["Última Fecha"] = sin_coord_tbl["Ultima_Fecha"].dt.strftime("%d/%m/%Y").fillna("")
        sin_coord_tbl = sin_coord_tbl[
            ["Tienda_Cod", "CDs", "Paradas_Afectadas", "Viajes_Unicos", "Primera Fecha", "Última Fecha"]
        ].rename(
            columns={
                "Tienda_Cod": "Tienda",
                "CDs": "CD(s)",
                "Paradas_Afectadas": "Paradas Afectadas",
                "Viajes_Unicos": "Viajes Únicos",
            }
        )
    else:
        sin_coord_tbl = pd.DataFrame(
            columns=["Tienda", "CD(s)", "Paradas Afectadas", "Viajes Únicos", "Primera Fecha", "Última Fecha"]
        )

    # 3D. Detalle de paradas sin GPS para exportacion
    raw_fails_export = (
        gps_filt[gps_filt["Diagnostico_GPS"] != "Visitada (OK)"][
            [
                "Operacion",
                "Fecha_Reparto",
                "Hora Reparto",
                "Ref_Clean",
                "Id Viaje",
                "Transportista",
                "Dominio",
                "Tipo de Camion",
                "Tienda_Cod",
                "Diagnostico_GPS",
            ]
        ]
        .copy()
        .sort_values(["Operacion", "Fecha_Reparto", "Ref_Clean"])
    )
    raw_fails_export["Fecha Reparto"] = raw_fails_export["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")
    raw_fails_export = raw_fails_export[
        [
            "Operacion",
            "Fecha Reparto",
            "Hora Reparto",
            "Ref_Clean",
            "Id Viaje",
            "Transportista",
            "Dominio",
            "Tipo de Camion",
            "Tienda_Cod",
            "Diagnostico_GPS",
        ]
    ].rename(
        columns={
            "Operacion": "CD",
            "Ref_Clean": "Ref Externa Viaje",
            "Tipo de Camion": "Tipo Camión",
            "Tienda_Cod": "Tienda",
            "Diagnostico_GPS": "Diagnóstico GPS",
        }
    )

    gps_excel_bytes = build_gps_audit_excel(
        st_rank_all, dom_rank_all, sin_coord_tbl, raw_fails_export
    )
    st.download_button(
        label="📥 Descargar Auditoría GPS y Geocercas en Excel (.xlsx)",
        data=gps_excel_bytes,
        file_name="Auditoria_GPS_y_Geocercas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    cant_geo_rec_pairs = int((st_rank_all["Falla Geozona (Activ. OK)"] >= 2).sum())
    cant_dom_ciegos = int((dom_rank_all["Viajes Ciegos"] > 0).sum())

    gtab1, gtab2, gtab3 = st.tabs(
        [
            f"🗺️ Tiendas a Revisar Geozona ({cant_geo_rec_pairs} recurrentes)",
            f"📡 Unidades Sin Señal GPS ({cant_dom_ciegos})",
            f"📌 Sin Coordenadas ({len(sin_coord_tbl)}) e Interfaz TMS",
        ]
    )

    with gtab1:
        st.caption(
            "Apertura completa de cada tienda: Paradas Plan. = Visitadas OK + Falla Geozona (polígono) + "
            "Por Activ. Tarde (>5 km CD) + Falla Viaje Ciego (unidad sin GPS en todo el viaje)."
        )
        gc1, gc2 = st.columns(2)
        with gc1:
            min_fallas_geo = st.number_input(
                "Mínimo de fallas de geozona (2 = recurrentes):",
                min_value=1,
                max_value=50,
                value=2,
                step=1,
                key="min_fallas_geo",
            )
        with gc2:
            min_paradas_geo = st.number_input(
                "Mínimo de paradas planificadas en la tienda:",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
                key="min_paradas_geo",
            )

        st_geo_sub = st_rank_all[
            (st_rank_all["Falla Geozona (Activ. OK)"] >= min_fallas_geo)
            & (st_rank_all["Paradas Plan."] >= min_paradas_geo)
        ].copy()
        st_geo_sub["% Falla Geozona"] = np.where(
            st_geo_sub["Paradas Plan."] > 0,
            (st_geo_sub["Falla Geozona (Activ. OK)"] / st_geo_sub["Paradas Plan."] * 100).round(1),
            0.0,
        )
        st_geo_view = (
            st_geo_sub[
                [
                    "CD",
                    "Tienda",
                    "Paradas Plan.",
                    "Visitadas OK",
                    "Falla Geozona (Activ. OK)",
                    "Falla c/ Activ. Tarde",
                    "Falla Viaje Ciego",
                    "Re-entradas (>1)",
                    "% Falla Geozona",
                ]
            ]
            .sort_values(["Falla Geozona (Activ. OK)", "% Falla Geozona"], ascending=[False, False])
            .rename(
                columns={
                    "Falla Geozona (Activ. OK)": "Falla Geozona",
                    "Falla c/ Activ. Tarde": "Por Activ. Tarde",
                }
            )
        )

        st.dataframe(
            st_geo_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "CD": st.column_config.TextColumn("CD", width="small"),
                "Tienda": st.column_config.TextColumn("Tienda", width="small"),
                "Paradas Plan.": st.column_config.NumberColumn("Paradas Plan.", format="%d", width="small"),
                "Visitadas OK": st.column_config.NumberColumn("Visitadas OK", format="%d", width="small"),
                "Falla Geozona": st.column_config.NumberColumn("Falla Geozona", format="%d", width="small"),
                "Por Activ. Tarde": st.column_config.NumberColumn("Por Activ. Tarde", format="%d", width="small"),
                "Falla Viaje Ciego": st.column_config.NumberColumn("Falla Viaje Ciego", format="%d", width="small"),
                "Re-entradas (>1)": st.column_config.NumberColumn("Re-entradas (>1)", format="%d", width="small"),
                "% Falla Geozona": st.column_config.ProgressColumn(
                    "% Falla Geozona",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                    width="medium",
                ),
            },
        )

    with gtab2:
        st.caption(
            "Unidades (patentes) con viajes donde ninguna tienda registró entrada a geocerca (0% señal GPS en el recorrido)."
        )
        min_viajes_gps = st.number_input(
            "Mínimo de viajes realizados por la patente:",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            key="min_viajes_gps",
        )
        fleet_view = dom_rank_all[
            (dom_rank_all["Viajes TMS"] >= min_viajes_gps) & (dom_rank_all["Viajes Ciegos"] > 0)
        ]

        st.dataframe(
            fleet_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "% Viajes Ciegos": st.column_config.ProgressColumn(
                    "% Viajes Ciegos",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                    width="small",
                ),
                "% Efectividad GPS": st.column_config.ProgressColumn(
                    "% Efectividad GPS",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                    width="small",
                ),
            },
        )

    with gtab3:
        st.markdown("**A. Tiendas Sin Coordenadas en Maestro UNIGIS**")
        st.dataframe(sin_coord_tbl, use_container_width=True, hide_index=True)

        st.markdown("**B. Resumen de Activación Tardía (>5 km del CD) y Error en Finalización por CD**")
        trips_int_df = (
            gps_filt.groupby(["Operacion", "Id Viaje"], as_index=False)
            .agg(
                Dist_Act=("Dist_Activacion_km", "first"),
                Act_Tarde=("obs_act_tarde", "max"),
                Err_Fin=("obs_err_fin", "max"),
            )
        )
        int_cd_summary = (
            trips_int_df.groupby("Operacion", as_index=False)
            .agg(
                Viajes_TMS=("Id Viaje", "count"),
                Viajes_Activ_Tarde=("Act_Tarde", "sum"),
                Dist_Mediana_Tarde_km=("Dist_Act", lambda s: round(float(s[s > 5].median()), 2) if (s > 5).any() else 0.0),
                Dist_Max_km=("Dist_Act", lambda s: round(float(s.max()), 1) if s.notna().any() else 0.0),
                Viajes_Error_Cierre=("Err_Fin", "sum"),
            )
            .rename(
                columns={
                    "Operacion": "CD",
                    "Viajes_TMS": "Viajes TMS",
                    "Viajes_Activ_Tarde": "Viajes Activ. Tarde (>5 km)",
                    "Dist_Mediana_Tarde_km": "Mediana Dist. Activ. Tarde (km)",
                    "Dist_Max_km": "Máx. Dist. Activ. (km)",
                    "Viajes_Error_Cierre": "Viajes Error Cierre",
                }
            )
        )
        st.dataframe(int_cd_summary, use_container_width=True, hide_index=True)

# =============================================================================
# SOLAPA 3: ESTADIAS Y PRODUCTIVIDAD EN TIENDA (LIMPIEZA PROPIA + BOXPLOTS)
# =============================================================================
with tab3:
    st.subheader("Estadías y Productividad de Descarga en Tienda")
    st.caption(
        "Muestra independiente sobre todas las paradas en tienda con visita GPS efectiva "
        "(no descarta paradas si faltó estadía en el CD o en otra parada del viaje)."
    )

    with st.expander("🎛️ Parámetros de Depuración de Estadías en Tienda", expanded=False):
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            min_est_tienda = st.number_input(
                "Estadía mínima válida (min):",
                min_value=0,
                max_value=30,
                value=5,
                step=1,
                help="Descarta pasos rápidos por la puerta de la tienda sin descarga real (<5 min).",
                key="s3_min_est",
            )
        with sc2:
            max_est_cerc = st.number_input(
                "Estadía máx. Cercana (min):",
                min_value=60,
                max_value=720,
                value=240,
                step=30,
                help="Excluye relojes abiertos o pernoctes en tiendas Cercana (default: 240 min = 4h).",
                key="s3_max_cerc",
            )
        with sc3:
            max_est_lej = st.number_input(
                "Estadía máx. Lejana (min):",
                min_value=60,
                max_value=960,
                value=360,
                step=30,
                help="Excluye relojes abiertos en tiendas Lejana >100 km (default: 360 min = 6h).",
                key="s3_max_lej",
            )
        with sc4:
            st.write("")
            excl_solape_s3 = st.checkbox(
                "Excluir solapamiento horario",
                value=True,
                help="Descarta paradas cuya hora de entrada ocurre antes de haber salido de la parada anterior.",
                key="s3_excl_solape",
            )

    # Construccion de la Base Limpia de Estadias en Tienda
    st_all_vis = dq_stores_filt[dq_stores_filt["visita_efectiva"]].copy()
    is_lej_s3 = st_all_vis["Categoria"] == "Lejana"
    max_est_vec_s3 = np.where(is_lej_s3, max_est_lej, max_est_cerc)

    r_corta_s3 = st_all_vis["Tiempo de Estadia"].isna() | (st_all_vis["Tiempo de Estadia"] < min_est_tienda)
    r_larga_s3 = st_all_vis["Tiempo de Estadia"] > max_est_vec_s3
    r_solape_s3 = (
        (
            st_all_vis["dt_ini_real"].notna()
            & st_all_vis["prev_dt_fin_real"].notna()
            & (st_all_vis["dt_ini_real"] < st_all_vis["prev_dt_fin_real"] - pd.Timedelta(minutes=1))
        )
        if excl_solape_s3
        else pd.Series(False, index=st_all_vis.index)
    )

    st_clean_s3 = st_all_vis[(~r_corta_s3) & (~r_larga_s3) & (~r_solape_s3)].copy()

    # Etiquetas de orden de parada en ruta
    st_clean_s3["Orden_Parada_Grupo"] = np.select(
        [
            st_clean_s3["Orden Visita Tienda"] == 1,
            st_clean_s3["Orden Visita Tienda"] == 2,
            st_clean_s3["Orden Visita Tienda"] == 3,
            st_clean_s3["Orden Visita Tienda"] >= 4,
        ],
        ["1ª Parada", "2ª Parada", "3ª Parada", "4ª+ Parada"],
        default="Sin orden",
    )

    # KPIs de Limpieza + Estadía + Productividad + Fases de Puerta
    tot_vis_s3 = len(st_all_vis)
    ok_vis_s3 = len(st_clean_s3)
    excl_vis_s3 = tot_vis_s3 - ok_vis_s3
    pct_ret_s3 = (ok_vis_s3 / tot_vis_s3 * 100) if tot_vis_s3 > 0 else 0.0

    est_series = st_clean_s3["Tiempo de Estadia"].dropna()
    med_est_s3 = float(est_series.median()) if len(est_series) > 0 else 0.0
    p25_est_s3 = float(np.percentile(est_series, 25)) if len(est_series) > 0 else 0.0
    p75_est_s3 = float(np.percentile(est_series, 75)) if len(est_series) > 0 else 0.0
    p90_est_s3 = float(np.percentile(est_series, 90)) if len(est_series) > 0 else 0.0

    # Productividad por Bultos (paradas con >= 20 bultos para evitar distorsiones de 1-2 bultos)
    prod_s3 = st_clean_s3[st_clean_s3["Total Bultos_num"].fillna(0) >= 20].copy()
    prod_s3["Min_100_Bultos"] = prod_s3["Tiempo de Estadia"] / (prod_s3["Total Bultos_num"] / 100.0)
    med_min_100b = float(prod_s3["Min_100_Bultos"].median()) if len(prod_s3) > 0 else 0.0
    med_bultos_s3 = float(st_clean_s3["Total Bultos_num"].median()) if ok_vis_s3 > 0 else 0.0
    bultos_hora_s3 = (6000.0 / med_min_100b) if med_min_100b > 0 else 0.0

    # Sensor de puerta trasera (pre-conectado para cuando UNIGIS active el dato)
    puerta_activa_cnt = int((st_clean_s3["Minutos Apertura y Cierre_num"].fillna(0) > 0).sum())

    k3_1, k3_2, k3_3, k3_4 = st.columns([1.15, 1.25, 1.25, 1.25])
    with k3_1:
        st.markdown(
            _kpi_card_html(
                "Base Limpia Estadías",
                _fmt_int(ok_vis_s3),
                f"{_fmt_pct(pct_ret_s3)} de {_fmt_int(tot_vis_s3)} visitas OK",
                "#16a34a" if pct_ret_s3 >= 90 else "#d97706",
                f"Excluidas: <b>{int(r_corta_s3.sum())}</b> <{min_est_tienda}m | <b>{int(r_larga_s3.sum())}</b> >máx | <b>{int(r_solape_s3.sum())}</b> solape",
            ),
            unsafe_allow_html=True,
        )
    with k3_2:
        st.markdown(
            _kpi_card_html(
                "Mediana Estadía en Tienda",
                f"{med_est_s3:.0f} min",
                f"P25: {p25_est_s3:.0f} min  |  P75: {p75_est_s3:.0f} min",
                "#1d4ed8",
                f"Percentil 90 (crítico): <b>{p90_est_s3:.0f} min</b>",
            ),
            unsafe_allow_html=True,
        )
    with k3_3:
        st.markdown(
            _kpi_card_html(
                "Productividad de Descarga",
                f"{med_min_100b:.1f} min",
                "Cada 100 bultos entregados",
                "#0f172a",
                f"Ritmo: <b>{bultos_hora_s3:.0f} bultos/h</b> | Mediana: <b>{med_bultos_s3:.0f} bultos</b>",
            ),
            unsafe_allow_html=True,
        )
    with k3_4:
        if puerta_activa_cnt > 0:
            med_desc_efectiva = float(
                st_clean_s3[st_clean_s3["Minutos Apertura y Cierre_num"] > 0]["Minutos Apertura y Cierre_num"].median()
            )
            st.markdown(
                _kpi_card_html(
                    "Descarga Efectiva (Puerta)",
                    f"{med_desc_efectiva:.0f} min",
                    f"Con sensor activo en {_fmt_int(puerta_activa_cnt)} paradas",
                    "#16a34a",
                    "Fase 2: Apertura a Cierre puerta trasera",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                _kpi_card_html(
                    "Fases Puerta Trasera",
                    "Sensor en 0",
                    "Pendiente activación en UNIGIS",
                    "#64748b",
                    "Pre-conectado: Espera + Descarga + Egreso",
                ),
                unsafe_allow_html=True,
            )

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE 1: DIAGRAMAS DE CAJAS (BOXPLOTS) Y PERCENTILES POR SEGMENTO
    # -------------------------------------------------------------------------
    st.markdown("### 1. Distribución de Estadías (Diagrama de Cajas y Percentiles)")

    dim_map_s3 = {
        "Tipo de Viaje (Directo / Compartido / Rutas)": "Tipo_Viaje",
        "Tipo de Vehículo (Nómina)": "Tipo_Vehiculo",
        "Operación (CD)": "Operacion",
        "Categoría (Cercana vs. Lejana)": "Categoria",
        "Orden de Parada en Ruta": "Orden_Parada_Grupo",
    }
    sub_map_s3 = {
        "Sin sub-corte": None,
        "Categoría (Cercana vs. Lejana)": "Categoria",
        "Tipo de Viaje": "Tipo_Viaje",
        "Tipo de Vehículo (Nómina)": "Tipo_Vehiculo",
    }

    bx_c1, bx_c2 = st.columns([1.3, 1.2])
    with bx_c1:
        dim_label_s3 = st.selectbox(
            "Agrupar eje principal por:",
            options=list(dim_map_s3.keys()),
            index=0,
            key="s3_dim_sel",
        )
    with bx_c2:
        sub_label_s3 = st.selectbox(
            "Sub-dividir cajas por color (opcional):",
            options=list(sub_map_s3.keys()),
            index=0,
            key="s3_sub_sel",
        )

    dim_col = dim_map_s3[dim_label_s3]
    sub_col = sub_map_s3[sub_label_s3]
    if sub_col == dim_col:
        sub_col = None

    if st_clean_s3.empty:
        st.info("No hay paradas limpias en el rango y filtros seleccionados.")
    else:
        df_plot_s3 = st_clean_s3.copy()
        if dim_col == "Operacion":
            df_plot_s3[dim_col] = "CD " + df_plot_s3[dim_col].astype(str)
        else:
            df_plot_s3[dim_col] = df_plot_s3[dim_col].fillna("Sin dato").astype(str)

        # Ordenar categorias por mediana descendente (excepto Orden de Parada o Tipo de Viaje)
        if dim_col == "Tipo_Viaje":
            cat_order = [
                c
                for c in ["Directo (1 tienda)", "Compartido (2 tiendas)", "Rutas (3+ tiendas)"]
                if c in df_plot_s3[dim_col].unique()
            ]
        elif dim_col == "Orden_Parada_Grupo":
            cat_order = [
                c
                for c in ["1ª Parada", "2ª Parada", "3ª Parada", "4ª+ Parada"]
                if c in df_plot_s3[dim_col].unique()
            ]
        else:
            cat_order = (
                df_plot_s3.groupby(dim_col)["Tiempo de Estadia"]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )

        fig_box = go.Figure()
        palette_box = ["#1d4ed8", "#d97706", "#16a34a", "#7c3aed", "#dc2626", "#0891b2", "#475569"]

        if sub_col is None:
            for idx_c, cat_val in enumerate(cat_order):
                vals = df_plot_s3[df_plot_s3[dim_col] == cat_val]["Tiempo de Estadia"]
                fig_box.add_trace(
                    go.Box(
                        y=vals,
                        name=f"{cat_val} (n={len(vals):,})",
                        boxmean=True,
                        boxpoints=False,
                        marker_color=palette_box[idx_c % len(palette_box)],
                        hovertemplate=(
                            f"<b>{cat_val}</b><br>"
                            "Estadía: %{y:.0f} min<extra></extra>"
                        ),
                    )
                )
        else:
            df_plot_s3[sub_col] = df_plot_s3[sub_col].fillna("Sin dato").astype(str)
            sub_vals = sorted(df_plot_s3[sub_col].unique().tolist())
            for idx_s, sv in enumerate(sub_vals):
                sub_df = df_plot_s3[df_plot_s3[sub_col] == sv]
                fig_box.add_trace(
                    go.Box(
                        x=sub_df[dim_col],
                        y=sub_df["Tiempo de Estadia"],
                        name=sv,
                        boxmean=True,
                        boxpoints=False,
                        marker_color=palette_box[idx_s % len(palette_box)],
                    )
                )
            fig_box.update_layout(boxmode="group", xaxis=dict(categoryorder="array", categoryarray=cat_order))

        fig_box.update_layout(
            height=410,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Tiempo de Estadía en Tienda (minutos)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            showlegend=True,
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Tabla resumen estadistico del segmento elegido
        grp_cols = [dim_col] if sub_col is None else [dim_col, sub_col]
        seg_rows = []
        for keys, grp in df_plot_s3.groupby(grp_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            s_est = grp["Tiempo de Estadia"]
            g_prod = grp[grp["Total Bultos_num"].fillna(0) >= 20]
            m100 = (
                float((g_prod["Tiempo de Estadia"] / (g_prod["Total Bultos_num"] / 100.0)).median())
                if len(g_prod) > 0
                else np.nan
            )
            row_dict = {grp_cols[i]: keys[i] for i in range(len(grp_cols))}
            row_dict.update(
                {
                    "Paradas Limpias": int(len(s_est)),
                    "Mediana Bultos": round(float(grp["Total Bultos_num"].median()), 0),
                    "P25 (min)": round(float(np.percentile(s_est, 25)), 0),
                    "Mediana (min)": round(float(s_est.median()), 0),
                    "P75 (min)": round(float(np.percentile(s_est, 75)), 0),
                    "P90 (min)": round(float(np.percentile(s_est, 90)), 0),
                    "Promedio (min)": round(float(s_est.mean()), 1),
                    "Min / 100 Bultos": round(m100, 1) if pd.notna(m100) else None,
                }
            )
            seg_rows.append(row_dict)

        df_seg_summary = pd.DataFrame(seg_rows)
        if sub_col is None and dim_col in cat_order:
            df_seg_summary["_ord"] = df_seg_summary[dim_col].map({v: i for i, v in enumerate(cat_order)})
            df_seg_summary = df_seg_summary.sort_values("_ord").drop(columns=["_ord"])

        st.dataframe(df_seg_summary, use_container_width=True, hide_index=True)

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE 2: MATRIZ CD x TIPO DE VEHICULO (NOMINA) + FASES PUERTA TRASERA
    # -------------------------------------------------------------------------
    st.markdown("### 2. Matriz por Operación (CD) y Tipo de Vehículo (Nómina)")

    if not st_clean_s3.empty:
        piv_med_veh = (
            st_clean_s3.assign(CD_Str="CD " + st_clean_s3["Operacion"].astype(str))
            .pivot_table(
                index="CD_Str",
                columns="Tipo_Vehiculo",
                values="Tiempo de Estadia",
                aggfunc="median",
            )
            .round(0)
        )
        piv_cnt_veh = (
            st_clean_s3.assign(CD_Str="CD " + st_clean_s3["Operacion"].astype(str))
            .pivot_table(
                index="CD_Str",
                columns="Tipo_Vehiculo",
                values="Tiempo de Estadia",
                aggfunc="count",
            )
            .fillna(0)
            .astype(int)
        )

        # Combinar "42 min (n=1.230)" en una tabla limpia de lectura gerencial
        mat_display = pd.DataFrame(index=piv_med_veh.index)
        for col_v in piv_med_veh.columns:
            mat_display[col_v] = [
                f"{int(piv_med_veh.loc[idx_r, col_v])} min (n={piv_cnt_veh.loc[idx_r, col_v]:,})".replace(",", ".")
                if pd.notna(piv_med_veh.loc[idx_r, col_v]) and piv_cnt_veh.loc[idx_r, col_v] > 0
                else "—"
                for idx_r in piv_med_veh.index
            ]
        mat_display = mat_display.reset_index().rename(columns={"CD_Str": "Operación"})
        st.dataframe(mat_display, use_container_width=True, hide_index=True)

        with st.expander("🚪 Fases de Estadía en Tienda (Sensor Puerta Trasera)", expanded=False):
            st.markdown(
                """
                Cuando UNIGIS active la telemetría de puerta trasera (`Apertura Puerta Trasera` y `Cierre Puerta Trasera`),
                este bloque abrirá automáticamente cada estadía en sus **3 fases operativas**:
                1. **Fase 1 — Espera / Recepción en Tienda:** `Apertura Puerta Trasera − Inicio Horario Visita Real` *(demora hasta que abren cortina/reciben)*.
                2. **Fase 2 — Descarga Efectiva:** `Minutos Apertura y Cierre` *(tiempo real de descarga física de mercadería)*.
                3. **Fase 3 — Cierre Documental y Egreso:** `Fecha Hora Salida − Cierre Puerta Trasera` *(demora de remito/firma y salida de geocerca)*.
                """
            )
            if puerta_activa_cnt == 0:
                st.info("Estado actual del archivo UNIGIS: las 3 columnas de puerta trasera vienen en `0` en el 100% de las paradas.")

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE 3: RANKING DE TIENDAS POR TIEMPO DE ESTADIA + DESCARGA EXCEL
    # -------------------------------------------------------------------------
    st.markdown("### 3. Ranking de Tiendas por Estadía y Productividad")

    rk_c1, rk_c2, rk_c3 = st.columns([1.2, 1.2, 1.2])
    with rk_c1:
        min_vis_rk = st.number_input(
            "Mínimo de visitas limpias en la tienda:",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            key="s3_min_vis_rk",
        )
    with rk_c2:
        filtro_tv_rk = st.selectbox(
            "Filtrar por Tipo de Viaje:",
            options=["Todos", "Directo (1 tienda)", "Compartido (2 tiendas)", "Rutas (3+ tiendas)"],
            index=0,
            key="s3_tv_rk",
        )
    with rk_c3:
        filtro_cat_rk = st.selectbox(
            "Filtrar por Categoría:",
            options=["Todas", "Cercana", "Lejana"],
            index=0,
            key="s3_cat_rk",
        )

    rk_base = st_clean_s3.copy()
    if filtro_tv_rk != "Todos":
        rk_base = rk_base[rk_base["Tipo_Viaje"] == filtro_tv_rk]
    if filtro_cat_rk != "Todas":
        rk_base = rk_base[rk_base["Categoria"] == filtro_cat_rk]

    rk_rows = []
    for (op_r, t_cod), grp in rk_base.groupby(["Operacion", "Tienda_Cod"]):
        if len(grp) < min_vis_rk:
            continue
        s_e = grp["Tiempo de Estadia"]
        g_p = grp[grp["Total Bultos_num"].fillna(0) >= 20]
        m100_t = (
            float((g_p["Tiempo de Estadia"] / (g_p["Total Bultos_num"] / 100.0)).median())
            if len(g_p) > 0
            else np.nan
        )
        rk_rows.append(
            {
                "CD": str(int(op_r)),
                "Tienda": str(t_cod),
                "Categoría": grp["Categoria"].mode().iloc[0] if not grp["Categoria"].mode().empty else "Sin dato",
                "Visitas Limpias": int(len(s_e)),
                "Mediana Bultos": round(float(grp["Total Bultos_num"].median()), 0),
                "P25 (min)": round(float(np.percentile(s_e, 25)), 0),
                "Mediana (min)": round(float(s_e.median()), 0),
                "P75 (min)": round(float(np.percentile(s_e, 75)), 0),
                "P90 (min)": round(float(np.percentile(s_e, 90)), 0),
                "Min / 100 Bultos": round(m100_t, 1) if pd.notna(m100_t) else None,
            }
        )

    df_rk_tiendas = (
        pd.DataFrame(rk_rows).sort_values(["Mediana (min)", "P90 (min)"], ascending=[False, False])
        if rk_rows
        else pd.DataFrame(
            columns=[
                "CD",
                "Tienda",
                "Categoría",
                "Visitas Limpias",
                "Mediana Bultos",
                "P25 (min)",
                "Mediana (min)",
                "P75 (min)",
                "P90 (min)",
                "Min / 100 Bultos",
            ]
        )
    )

    clean_s3_export = st_clean_s3.copy()
    clean_s3_export["Fecha Reparto"] = clean_s3_export["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")
    estadias_excel_bytes = build_estadias_excel(
        df_seg_summary if not st_clean_s3.empty else pd.DataFrame(),
        mat_display if not st_clean_s3.empty else pd.DataFrame(),
        df_rk_tiendas,
        clean_s3_export,
    )
    st.download_button(
        label="📥 Descargar Análisis de Estadías en Tienda (.xlsx)",
        data=estadias_excel_bytes,
        file_name="Analisis_Estadias_y_Productividad_Tienda.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.dataframe(df_rk_tiendas, use_container_width=True, hide_index=True)

# =============================================================================
# PLACEHOLDERS PARA SOLAPAS 4 Y 5
# =============================================================================
with tab4:
    st.info(
        "🚧 **Próximo paso (Solapa 4 — Velocidades y Parametrización UNIGIS):** "
        "Cálculo de las 4 velocidades de UNIGIS (`Velocidad promedio entre paradas`, `Velocidad depósito salida`, "
        "`Velocidad depósito llegada` y `Velocidad máxima`) abiertas por **Operación (CD)** y **Tipo de Vehículo (Nómina)**, "
        "con su bloque de limpieza de tramos y diagramas de cajas."
    )

with tab5:
    st.info(
        "🚧 **Solapa 5 — Internos, CD Origen y Jornada Diaria:** "
        "Permanencia en CD Origen, desvío vs. `Hora Reparto`, análisis de vueltas por día (`1` vs `2` vs `3+ vueltas`) "
        "y jornada del chofer por interno/patente."
    )
