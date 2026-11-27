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

_ui_fragment = getattr(st, "fragment", getattr(st, "experimental_fragment", lambda f: f))
dias_es = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}

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
    out = pd.Series(np.nan, index=series.index, dtype="object")
    mask = num.notna()
    out.loc[mask] = num.loc[mask].astype("int64").astype(str)
    return out


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
        return pd.DataFrame(columns=["Interno_Nom", "Tipo_Vehiculo_Nom", "Satelital_Nom", "Empresa_Nom"])
    try:
        df_n = pd.read_excel(io.BytesIO(nomina_bytes), sheet_name=0)
    except Exception:
        return pd.DataFrame(columns=["Interno_Nom", "Tipo_Vehiculo_Nom", "Satelital_Nom", "Empresa_Nom"])

    if "Patente" not in df_n.columns:
        return pd.DataFrame(columns=["Interno_Nom", "Tipo_Vehiculo_Nom", "Satelital_Nom", "Empresa_Nom"])

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
    out["Interno_Nom"] = (
        _clean_int_str(df_n["Interno"]).fillna(df_n["Interno"].fillna("").astype(str).str.strip())
        if "Interno" in df_n.columns
        else ""
    )
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
    # Excluir internos/transportistas administrativos o especiales (1331 Transferencias, 1333 Dev. picking, 50000 Salta)
    codigos_excl_especiales = {"1331", "1333", "50000"}
    for col_a in df_a.columns:
        c_low = col_a.lower()
        if "interno" in c_low or "transportista" in c_low:
            val_clean_a = _clean_int_str(df_a[col_a]).fillna(
                df_a[col_a].fillna("").astype(str).str.strip()
            )
            df_a = df_a[~val_clean_a.isin(codigos_excl_especiales)].copy()

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
    if "Interno" in df_u.columns:
        int_clean_u = _clean_int_str(df_u["Interno"]).fillna(
            df_u["Interno"].fillna("").astype(str).str.strip()
        )
        df_u = df_u[~int_clean_u.isin(codigos_excl_especiales)].copy()

    # Tambien excluir en UNIGIS cualquier viaje cuya Ref Externa corresponda a un viaje excluido de AMREPA
    refs_excluidas_amrepa = set()
    for col_a_raw in df_a_raw.columns:
        c_low_raw = col_a_raw.lower()
        if "interno" in c_low_raw or "transportista" in c_low_raw:
            v_raw = _clean_int_str(df_a_raw[col_a_raw]).fillna(
                df_a_raw[col_a_raw].fillna("").astype(str).str.strip()
            )
            mask_ex = v_raw.isin(codigos_excl_especiales)
            if mask_ex.any():
                op_ex = pd.to_numeric(df_a_raw.loc[mask_ex, "[Codigo almacen]"], errors="coerce").astype("Int64")
                cnt_ex = pd.to_numeric(df_a_raw.loc[mask_ex, "[Contador transporte]"], errors="coerce").astype("Int64")
                refs_excluidas_amrepa.update((op_ex.astype(str) + " - " + cnt_ex.astype(str)).dropna().tolist())

    if refs_excluidas_amrepa and "Ref Externa Viaje" in df_u.columns:
        df_u = df_u[~df_u["Ref Externa Viaje"].fillna("").astype(str).str.strip().isin(refs_excluidas_amrepa)].copy()

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
    master_trips["Semana_Sort"] = master_trips["Fecha_Base"].dt.to_period("W").dt.start_time
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

    en_ambos_l, cant_f_l, cant_e_l, str_f_l, str_e_l = [], [], [], [], []
    for sw_t, st_t in zip(master_trips["Set_Tiendas_WMS"], master_trips["Set_Tiendas_TMS"]):
        sw = set(sw_t)
        st_set = set(st_t)
        en_ambos_l.append(len(sw & st_set))
        f_tms = sorted(sw - st_set)
        e_tms = sorted(st_set - sw)
        cant_f_l.append(len(f_tms))
        cant_e_l.append(len(e_tms))
        str_f_l.append(", ".join(f_tms))
        str_e_l.append(", ".join(e_tms))

    master_trips["Tiendas_Match"] = en_ambos_l
    master_trips["Cant_Tiendas_Faltan_TMS"] = cant_f_l
    master_trips["Cant_Tiendas_Extra_TMS"] = cant_e_l
    master_trips["Lista_Faltan_TMS"] = str_f_l
    master_trips["Lista_Extra_TMS"] = str_e_l

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

    def _merge_trans_nomina(trans_series: pd.Series, nom_int_series: pd.Series) -> pd.Series:
        t_clean = trans_series.fillna("Sin dato WMS").astype(str).str.strip()
        n_clean = nom_int_series.fillna("").astype(str).str.strip()
        # Si Nomina tiene interno valido y NO coincide con AMREPA (ni esta ya contenido en AMREPA)
        diff_mask = (
            (n_clean != "")
            & (n_clean != "0")
            & (n_clean.str.lower() != "nan")
            & (t_clean != n_clean)
            & (~t_clean.isin(["Sin dato", "Sin dato WMS"]))
        )
        # Verificar tambien si t_clean tiene multiples transportistas separados por " / "
        already_in = pd.Series(
            [n in [p.strip() for p in t.split("/")] for t, n in zip(t_clean, n_clean)],
            index=trans_series.index,
        )
        show_both = diff_mask & (~already_in)
        return pd.Series(
            np.where(show_both, t_clean + " / " + n_clean + " (Nóm.)", t_clean),
            index=trans_series.index,
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
        mapped_int_nom = stores_gps["Dominio_Clean"].map(nomina_lut["Interno_Nom"]).fillna("")
        stores_gps["Transportista"] = _merge_trans_nomina(stores_gps["Transportista"], mapped_int_nom)
    else:
        stores_gps["Tipo_Vehiculo"] = stores_gps["Tipo de Camion"]
        stores_gps["Satelital"] = "Sin dato"

    stores_gps["Mes_Label"] = stores_gps["Fecha_Reparto"].dt.strftime("%Y-%m").fillna("Sin fecha")
    stores_gps["Semana_Sort"] = stores_gps["Fecha_Reparto"].dt.to_period("W").dt.start_time
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
        mapped_int_u = u_clean_num["Dominio_Clean"].map(nomina_lut["Interno_Nom"]).fillna("")
        u_clean_num["Transportista"] = _merge_trans_nomina(u_clean_num["Transportista"], mapped_int_u)
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
    u_clean_num["Semana_Sort"] = u_clean_num["Fecha_Reparto"].dt.to_period("W").dt.start_time
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
            Interno=("Interno", "first"),
            Tipo_Camion=("Tipo de Camion", "first"),
            Tipo_Vehiculo=("Tipo_Vehiculo", "first"),
            Satelital=("Satelital", "first"),
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

    # Pre-cálculo inicial de bases limpias por defecto para que el cambio de filtros sea instantáneo (<5 ms)
    # A. Pre-cálculo Estadías limpias por defecto (Solapa 3)
    st_vis_def = dq_stores[dq_stores["visita_efectiva"]].copy()
    if not st_vis_def.empty:
        min_v_def = np.select(
            [st_vis_def["Tipo_Viaje"] == "Directo (1 tienda)", st_vis_def["Tipo_Viaje"] == "Compartido (2 tiendas)"],
            [15.0, 10.0],
            default=5.0,
        )
        is_lej_def = st_vis_def["Categoria"] == "Lejana"
        max_v_def = np.select(
            [
                (st_vis_def["Tipo_Viaje"] == "Directo (1 tienda)") & (~is_lej_def),
                (st_vis_def["Tipo_Viaje"] == "Directo (1 tienda)") & is_lej_def,
                (st_vis_def["Tipo_Viaje"] == "Compartido (2 tiendas)") & (~is_lej_def),
                (st_vis_def["Tipo_Viaje"] == "Compartido (2 tiendas)") & is_lej_def,
                (st_vis_def["Tipo_Viaje"] == "Rutas (3+ tiendas)") & (~is_lej_def),
                (st_vis_def["Tipo_Viaje"] == "Rutas (3+ tiendas)") & is_lej_def,
            ],
            [240.0, 360.0, 180.0, 300.0, 150.0, 240.0],
            default=240.0,
        )
        b_def = st_vis_def["Total Bultos_num"].fillna(0)
        m100_def = np.where(b_def > 0, st_vis_def["Tiempo de Estadia"] / (b_def / 100.0), np.nan)
        sol_def = (
            st_vis_def["dt_ini_real"].notna()
            & st_vis_def["prev_dt_fin_real"].notna()
            & (st_vis_def["dt_ini_real"] < st_vis_def["prev_dt_fin_real"] - pd.Timedelta(minutes=1))
        )
        ok_st_def = (
            (~((b_def >= 50) & (m100_def < 3.0)))
            & (~(st_vis_def["Tiempo de Estadia"].isna() | (st_vis_def["Tiempo de Estadia"] < min_v_def)))
            & (~(st_vis_def["Tiempo de Estadia"] > max_v_def))
            & (~sol_def)
        )
        st_clean_default = st_vis_def[ok_st_def].copy()
        st_clean_default["Orden_Parada_Grupo"] = np.select(
            [
                st_clean_default["Orden Visita Tienda"] == 1,
                st_clean_default["Orden Visita Tienda"] == 2,
                st_clean_default["Orden Visita Tienda"] == 3,
                st_clean_default["Orden Visita Tienda"] >= 4,
            ],
            ["1ª Parada", "2ª Parada", "3ª Parada", "4ª+ Parada"],
            default="Sin orden",
        )
    else:
        st_clean_default = st_vis_def

    # B. Pre-cálculo Viajes y Jornadas por defecto (Solapa 5)
    has_sol_s5 = (
        dq_stores["dt_ini_real"].notna()
        & dq_stores["prev_dt_fin_real"].notna()
        & (dq_stores["dt_ini_real"] < dq_stores["prev_dt_fin_real"] - pd.Timedelta(minutes=1))
    )
    st_tt_def = (
        dq_stores.assign(_solape=has_sol_s5)
        .groupby("Id Viaje", as_index=False)
        .agg(
            Ini_Primera_Tienda=("dt_ini_real", "min"),
            Fin_Ultima_Tienda=("dt_fin_real", "max"),
            Suma_Estadia_Tiendas_min=("Tiempo de Estadia", "sum"),
            Min_Estadia_Tienda=("Tiempo de Estadia", "min"),
            Max_Estadia_Tienda=("Tiempo de Estadia", "max"),
            Tiene_Solape=("_solape", "any"),
        )
    )
    trips_pre_s5 = dq_trips.merge(st_tt_def, on="Id Viaje", how="left")
    trips_pre_s5["dt_ini_cd"] = pd.to_datetime(trips_pre_s5["Ini_CD_Origen"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    trips_pre_s5["dt_fin_cd"] = pd.to_datetime(trips_pre_s5["Fin_CD_Origen"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    ic_pre = _clean_int_str(trips_pre_s5["Interno"]).fillna(trips_pre_s5["Interno"].fillna("").astype(str).str.strip())
    dom_pre = trips_pre_s5["Dominio"].fillna("S/D").astype(str).str.strip().str.upper()
    trips_pre_s5["Interno_Clean"] = np.where(ic_pre.isin(["", "0", "S/D", "NAN", "None"]), "S/D (" + dom_pre + ")", ic_pre)
    trips_pre_s5["Dominio_Clean"] = trips_pre_s5["Dominio"].fillna("").astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    trips_pre_s5["Dominio_Valido"] = (~trips_pre_s5["Dominio_Clean"].isin(["", "SINDOMINIO", "XXX000", "XXXXXX", "NAN"])) & (~trips_pre_s5["Dominio_Clean"].str.startswith("XXX"))
    trips_pre_s5["_is_lejana"] = (trips_pre_s5["Categoria"] == "Lejana").astype(int)

    return {
        "master_trips": master_trips,
        "dup_detail": dup_detail,
        "invalid_detail": invalid_detail,
        "stores_gps": stores_gps,
        "dq_stores": dq_stores,
        "dq_cd_orig": dq_cd_orig,
        "dq_trips": dq_trips,
        "st_clean_default": st_clean_default,
        "trips_pre_s5": trips_pre_s5,
    }



def _add_clean_box_traces(
    fig: go.Figure,
    x_labels: list,
    series_list: list[pd.Series],
    colors: list[str],
    unit_suffix: str,
    use_tukey: bool = True,
    group_name: str = "",
    show_legend: bool = False,
):
    """Dibuja cajas limpias con 1 sola tarjeta en español (Bigote Máximo, Q3, Media, Mediana, Q1, Bigote Mínimo)
    al apoyar el mouse en cualquier parte de la caja (tanto simple como agrupada por color)."""
    def _build_card(hi_w: float, q3_v: float, mean_v: float, med_v: float, q1_v: float, lo_w: float, n_cnt: int) -> str:
        header = f"<b>{group_name}</b> (n={n_cnt:,})<br>" if (show_legend and group_name) else f"<b>Muestra: n={n_cnt:,}</b><br>"
        return (
            f"{header}"
            f"Bigote Máximo: <b>{hi_w:.1f} {unit_suffix}</b><br>"
            f"Q3 (75%): <b>{q3_v:.1f} {unit_suffix}</b><br>"
            f"Media: <b>{mean_v:.1f} {unit_suffix}</b><br>"
            f"Mediana (50%): <b>{med_v:.1f} {unit_suffix}</b><br>"
            f"Q1 (25%): <b>{q1_v:.1f} {unit_suffix}</b><br>"
            f"Bigote Mínimo: <b>{lo_w:.1f} {unit_suffix}</b>"
        )

    if show_legend and group_name:
        x_valid, lf_l, q1_l, med_l, q3_l, uf_l, mean_l, spans_l, cards_l = [], [], [], [], [], [], [], [], []
        for x_lbl, vals in zip(x_labels, series_list):
            s = vals.dropna()
            if len(s) == 0:
                continue
            q1_v, med_v, q3_v = np.percentile(s, [25, 50, 75])
            mean_v = float(s.mean())
            if use_tukey:
                iqr = q3_v - q1_v
                lo_w = max(float(s.min()), float(q1_v - 1.5 * iqr))
                hi_w = min(float(s.max()), float(q3_v + 1.5 * iqr))
            else:
                lo_w = float(s.min())
                hi_w = float(s.max())
            x_valid.append(x_lbl)
            lf_l.append(round(lo_w, 1))
            q1_l.append(round(q1_v, 1))
            med_l.append(round(med_v, 1))
            q3_l.append(round(q3_v, 1))
            uf_l.append(round(hi_w, 1))
            mean_l.append(round(mean_v, 1))
            spans_l.append(max(0.2, round(hi_w - lo_w, 1)))
            cards_l.append(_build_card(hi_w, q3_v, mean_v, med_v, q1_v, lo_w, len(s)))

        if not x_valid:
            return
        col_hex = colors[0]
        fig.add_trace(
            go.Box(
                x=x_valid,
                lowerfence=lf_l,
                q1=q1_l,
                median=med_l,
                q3=q3_l,
                upperfence=uf_l,
                mean=mean_l,
                boxpoints=False,
                name=group_name,
                legendgroup=group_name,
                offsetgroup=group_name,
                showlegend=True,
                marker_color=col_hex,
                hoverinfo="skip",
            )
        )
        # Barra transparente alineada 1:1 con cada caja agrupada para desplegar tarjeta 100% en español
        fig.add_trace(
            go.Bar(
                x=x_valid,
                y=spans_l,
                base=lf_l,
                name=group_name,
                legendgroup=group_name,
                offsetgroup=group_name,
                showlegend=False,
                marker=dict(color="rgba(0,0,0,0)", line=dict(width=0)),
                customdata=cards_l,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
        return

    # Caso sin sub-corte (1 caja por posición X, cada una con su color)
    for idx_b, (x_lbl, vals) in enumerate(zip(x_labels, series_list)):
        s = vals.dropna()
        if len(s) == 0:
            continue
        q1_v, med_v, q3_v = np.percentile(s, [25, 50, 75])
        mean_v = float(s.mean())
        if use_tukey:
            iqr = q3_v - q1_v
            lo_w = max(float(s.min()), float(q1_v - 1.5 * iqr))
            hi_w = min(float(s.max()), float(q3_v + 1.5 * iqr))
        else:
            lo_w = float(s.min())
            hi_w = float(s.max())

        col_hex = colors[idx_b % len(colors)]
        card_txt = _build_card(hi_w, q3_v, mean_v, med_v, q1_v, lo_w, len(s))
        fig.add_trace(
            go.Box(
                x=[x_lbl],
                lowerfence=[round(lo_w, 1)],
                q1=[round(q1_v, 1)],
                median=[round(med_v, 1)],
                q3=[round(q3_v, 1)],
                upperfence=[round(hi_w, 1)],
                mean=[round(mean_v, 1)],
                boxpoints=False,
                width=0.52,
                name="",
                showlegend=False,
                marker_color=col_hex,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[x_lbl],
                y=[max(0.2, round(hi_w - lo_w, 1))],
                base=[round(lo_w, 1)],
                width=0.52,
                name="",
                showlegend=False,
                marker=dict(color="rgba(0,0,0,0)", line=dict(width=0)),
                customdata=[card_txt],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )


@st.cache_data(show_spinner=False, max_entries=128)
def _build_grouped_boxplot_cached(
    df_clean: pd.DataFrame,
    df_tot: pd.DataFrame | None,
    val_col: str,
    dim_col: str,
    sub_col: str | None,
    unit_suffix: str,
    y_title: str,
    use_tukey: bool,
    chart_title: str | None = None,
    y_range: tuple | None = None,
    y_dtick: float | None = None,
    fixed_dim_order: tuple | None = None,
    fixed_sub_order: tuple | None = None,
    compute_summary_table: bool = False,
    nice_dim_name: str = "",
    summary_mode: str = "estadias",
):
    """Construye y cachea en RAM (hasta 128 combinaciones) la figura Plotly y su tabla resumen."""
    if df_clean.empty:
        return go.Figure(), pd.DataFrame()

    is_temp = dim_col in {"Evol_Mes", "Evol_Semana", "Evol_Dia"}
    if fixed_dim_order is not None:
        cat_order = [c for c in fixed_dim_order if c in df_clean[dim_col].unique()]
    elif dim_col in {"Evol_Mes", "Evol_Dia"} and "_p_sort" in df_clean.columns:
        cat_order = df_clean[["_p_sort", dim_col]].drop_duplicates().sort_values("_p_sort")[dim_col].tolist()
    elif dim_col == "Evol_Semana" and "Semana_Sort" in df_clean.columns:
        cat_order = df_clean[["Semana_Sort", dim_col]].drop_duplicates().sort_values("Semana_Sort")[dim_col].tolist()
    elif dim_col == "Operacion":
        cat_order = sorted(df_clean[dim_col].unique().tolist())
    else:
        cat_order = (
            df_clean.groupby(dim_col)[val_col]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

    fig = go.Figure()
    palette = ["#1d4ed8", "#d97706", "#16a34a", "#7c3aed", "#dc2626", "#0891b2", "#475569"]

    if sub_col is None:
        x_lbls, s_list, c_list = [], [], []
        for idx_c, cv in enumerate(cat_order):
            vals = df_clean[df_clean[dim_col] == cv][val_col].dropna()
            if len(vals) == 0:
                continue
            x_lbls.append(f"{cv}<br>(n={len(vals):,})" if is_temp else f"{cv} (n={len(vals):,})")
            s_list.append(vals)
            c_list.append("#1d4ed8" if is_temp else palette[idx_c % len(palette)])
        _add_clean_box_traces(
            fig,
            x_lbls,
            s_list,
            c_list,
            unit_suffix=unit_suffix,
            use_tukey=use_tukey,
            show_legend=False,
        )
        fig.update_layout(barmode="overlay", showlegend=False)
    else:
        if fixed_sub_order is not None:
            sub_vals = [s for s in fixed_sub_order if s in df_clean[sub_col].unique()]
        else:
            sub_vals = sorted(df_clean[sub_col].unique().tolist())
        for idx_s, sv in enumerate(sub_vals):
            sub_df = df_clean[df_clean[sub_col] == sv]
            xc, ser_sub = [], []
            for cv in cat_order:
                v_sub = sub_df[sub_df[dim_col] == cv][val_col].dropna()
                if len(v_sub) == 0:
                    continue
                xc.append(cv)
                ser_sub.append(v_sub)
            _add_clean_box_traces(
                fig,
                xc,
                ser_sub,
                [palette[idx_s % len(palette)]],
                unit_suffix=unit_suffix,
                use_tukey=use_tukey,
                group_name=sv,
                show_legend=True,
            )
        fig.update_layout(
            boxmode="group",
            barmode="group",
            xaxis=dict(categoryorder="array", categoryarray=cat_order),
            showlegend=True,
        )

    yaxis_cfg = dict(title=y_title)
    if y_range is not None:
        yaxis_cfg["range"] = list(y_range)
        yaxis_cfg["zeroline"] = True
        yaxis_cfg["zerolinecolor"] = "#cbd5e1"
    if y_dtick is not None:
        yaxis_cfg["dtick"] = y_dtick

    fig.update_layout(
        title=chart_title if chart_title else None,
        boxgap=0.18,
        boxgroupgap=0.10,
        bargap=0.18,
        bargroupgap=0.10,
        height=420,
        margin=dict(l=20, r=20, t=42 if chart_title else 35, b=20),
        yaxis=yaxis_cfg,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickangle=-35 if len(cat_order) > 12 else 0),
        hovermode="closest",
    )

    df_res = pd.DataFrame()
    if compute_summary_table:
        grp_cols = [dim_col] if sub_col is None else [dim_col, sub_col]
        tot_map = df_tot.groupby(grp_cols).size().to_dict() if df_tot is not None else {}
        rows = []
        for keys, grp in df_clean.groupby(grp_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            lk = keys[0] if len(keys) == 1 else keys
            n_tot = int(tot_map.get(lk, len(grp)))
            n_ok = int(len(grp))
            pct_rep = round((n_ok / n_tot * 100), 1) if n_tot > 0 else 100.0
            s_val = grp[val_col]
            rd = {nice_dim_name: keys[0]}
            if sub_col is not None:
                rd[sub_col] = keys[1]

            if summary_mode == "estadias":
                g_prod = grp[grp["Total Bultos_num"].fillna(0) >= 20]
                m100 = (
                    float((g_prod["Tiempo de Estadia"] / (g_prod["Total Bultos_num"] / 100.0)).median())
                    if len(g_prod) > 0
                    else np.nan
                )
                rd.update(
                    {
                        "Paradas Totales": n_tot,
                        "Paradas Limpias": n_ok,
                        "% Represent.": pct_rep,
                        "Mediana Bultos": round(float(grp["Total Bultos_num"].median()), 0),
                        "P25 (min)": round(float(np.percentile(s_val, 25)), 0),
                        "Mediana (min)": round(float(s_val.median()), 0),
                        "P75 (min)": round(float(np.percentile(s_val, 75)), 0),
                        "P90 (min)": round(float(np.percentile(s_val, 90)), 0),
                        "Media (min)": round(float(s_val.mean()), 1),
                        "Min / 100 Bultos": round(m100, 1) if pd.notna(m100) else None,
                    }
                )
            elif summary_mode == "viajes":
                rd.update(
                    {
                        "Viajes Totales": n_tot,
                        "Viajes OK": n_ok,
                        "% Represent.": pct_rep,
                        "Mediana KMS": round(float(grp["KMS_Total"].median()), 1),
                        "Med. Estadía CD (hs)": round(float(grp["Tiempo_CD_hs"].median()), 2),
                        "Med. Tiempo en Viaje (hs)": round(float(grp["Tiempo_Ruta_hs"].median()), 2),
                        "Med. Estadía Tiendas (hs)": round(float(grp["Tiempo_Tiendas_hs"].median()), 2),
                        "P25 Duración Total (hs)": round(float(np.percentile(s_val, 25)), 2),
                        "Mediana Duración Total (hs)": round(float(s_val.median()), 2),
                        "P75 Duración Total (hs)": round(float(np.percentile(s_val, 75)), 2),
                        "P90 Duración Total (hs)": round(float(np.percentile(s_val, 90)), 2),
                    }
                )
            rows.append(rd)
        df_res = pd.DataFrame(rows)
        if not df_res.empty and nice_dim_name in df_res.columns:
            df_res["_ord"] = df_res[nice_dim_name].map({v: i for i, v in enumerate(cat_order)})
            df_res = df_res.sort_values("_ord").drop(columns=["_ord"])

    return fig, df_res


def build_audit_excel(df_gap, df_solo_tms, df_store_diff, df_dup, df_inv) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_gap.to_excel(writer, sheet_name="1_Viajes_Solo_WMS_Gap", index=False)
        df_solo_tms.to_excel(writer, sheet_name="2_Viajes_Solo_TMS", index=False)
        df_store_diff.to_excel(writer, sheet_name="3_Diferencias_Tiendas", index=False)
        df_dup.to_excel(writer, sheet_name="4_Alerta_Ref_Duplicadas", index=False)
        df_inv.to_excel(writer, sheet_name="5_Alerta_Ref_Invalidas", index=False)
    return output.getvalue()


def build_gps_audit_excel(df_geo, df_fleet, df_sin_coord, df_raw_fails, df_cd_sin_dato=None) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_geo.to_excel(writer, sheet_name="1_Tiendas_Geocerca", index=False)
        df_fleet.to_excel(writer, sheet_name="2_Unidades_Viajes_Ciegos", index=False)
        df_sin_coord.to_excel(writer, sheet_name="3_Tiendas_Sin_Coordenadas", index=False)
        df_raw_fails.to_excel(writer, sheet_name="4_Detalle_Paradas_Sin_GPS", index=False)
        if df_cd_sin_dato is not None:
            df_cd_sin_dato.to_excel(writer, sheet_name="5_CD_Origen_Sin_Dato", index=False)
    return output.getvalue()


def build_estadias_excel(df_seg, df_rank, df_clean_st) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_seg.to_excel(writer, sheet_name="1_Resumen_Segmentos", index=False)
        df_rank.to_excel(writer, sheet_name="2_Ranking_Tiendas", index=False)
        df_clean_st.to_excel(writer, sheet_name="3_Base_Limpia_Estadias", index=False)
    return output.getvalue()


def build_camiones_excel(df_vp, df_vs, df_vr, df_vmax, df_mat_est, df_rank_int, df_tramos_clean) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_vp.to_excel(writer, sheet_name="1_Vel_Entre_Paradas", index=False)
        df_vs.to_excel(writer, sheet_name="2_Vel_Deposito_Salida", index=False)
        df_vr.to_excel(writer, sheet_name="3_Vel_Deposito_Llegada", index=False)
        df_vmax.to_excel(writer, sheet_name="4_Vel_Maxima_P95", index=False)
        df_mat_est.to_excel(writer, sheet_name="5_Estadias_CD_x_Camion", index=False)
        df_rank_int.to_excel(writer, sheet_name="6_Ranking_Internos", index=False)
        df_tramos_clean.to_excel(writer, sheet_name="7_Base_Limpia_Tramos", index=False)
    return output.getvalue()


def build_viajes_jornadas_excel(df_v_res, df_j_res, df_v_clean, df_j_clean, df_j_excl) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_v_res.to_excel(writer, sheet_name="1_Resumen_Viajes", index=False)
        df_j_res.to_excel(writer, sheet_name="2_Resumen_Jornadas", index=False)
        df_v_clean.to_excel(writer, sheet_name="3_Base_Limpia_Viajes_OK", index=False)
        df_j_clean.to_excel(writer, sheet_name="4_Base_Limpia_Jornadas", index=False)
        df_j_excl.to_excel(writer, sheet_name="5_Jornadas_Excluidas_Audit", index=False)
    return output.getvalue()



@st.cache_data(show_spinner=False)
def _clean_estadias_cached(
    df_in: pd.DataFrame,
    min_est_directo: int,
    min_est_comp: int,
    min_est_rutas: int,
    min_min100b_phys: float,
    max_est_cerc: int,
    max_est_lej: int,
    excl_solape_s3: bool,
):
    st_all_vis = df_in[df_in["visita_efectiva"]].copy()
    if st_all_vis.empty:
        return st_all_vis, {"corta": 0, "phys": 0, "larga": 0, "solape": 0, "tot": 0, "plan_tot": int(len(df_in))}

    min_est_vec_s3 = np.select(
        [
            st_all_vis["Tipo_Viaje"] == "Directo (1 tienda)",
            st_all_vis["Tipo_Viaje"] == "Compartido (2 tiendas)",
        ],
        [float(min_est_directo), float(min_est_comp)],
        default=float(min_est_rutas),
    )

    is_lej_s3 = st_all_vis["Categoria"] == "Lejana"
    max_est_vec_s3 = np.select(
        [
            (st_all_vis["Tipo_Viaje"] == "Directo (1 tienda)") & (~is_lej_s3),
            (st_all_vis["Tipo_Viaje"] == "Directo (1 tienda)") & is_lej_s3,
            (st_all_vis["Tipo_Viaje"] == "Compartido (2 tiendas)") & (~is_lej_s3),
            (st_all_vis["Tipo_Viaje"] == "Compartido (2 tiendas)") & is_lej_s3,
            (st_all_vis["Tipo_Viaje"] == "Rutas (3+ tiendas)") & (~is_lej_s3),
            (st_all_vis["Tipo_Viaje"] == "Rutas (3+ tiendas)") & is_lej_s3,
        ],
        [
            float(max_est_cerc),
            float(max_est_lej),
            round(float(max_est_cerc) * 0.75, 0),
            round(float(max_est_lej) * 0.8333, 0),
            round(float(max_est_cerc) * 0.625, 0),
            round(float(max_est_lej) * 0.6667, 0),
        ],
        default=float(max_est_cerc),
    )

    bultos_s3 = st_all_vis["Total Bultos_num"].fillna(0)
    min_100b_raw = np.where(
        bultos_s3 > 0,
        st_all_vis["Tiempo de Estadia"] / (bultos_s3 / 100.0),
        np.nan,
    )
    r_phys_s3 = (bultos_s3 >= 50) & (min_100b_raw < float(min_min100b_phys))
    r_corta_s3 = st_all_vis["Tiempo de Estadia"].isna() | (st_all_vis["Tiempo de Estadia"] < min_est_vec_s3)
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

    st_clean = st_all_vis[(~r_corta_s3) & (~r_larga_s3) & (~r_phys_s3) & (~r_solape_s3)].copy()
    st_clean["Orden_Parada_Grupo"] = np.select(
        [
            st_clean["Orden Visita Tienda"] == 1,
            st_clean["Orden Visita Tienda"] == 2,
            st_clean["Orden Visita Tienda"] == 3,
            st_clean["Orden Visita Tienda"] >= 4,
        ],
        ["1ª Parada", "2ª Parada", "3ª Parada", "4ª+ Parada"],
        default="Sin orden",
    )
    stats = {
        "plan_tot": int(len(df_in)),
        "tot": int(len(st_all_vis)),
        "corta": int(r_corta_s3.sum()),
        "phys": int(r_phys_s3.sum()),
        "larga": int(r_larga_s3.sum()),
        "solape": int(r_solape_s3.sum()),
    }
    return st_clean, stats


@st.cache_data(show_spinner=False)
def _rank_tiendas_s3_cached(
    df_all_s3: pd.DataFrame,
    st_clean_s3: pd.DataFrame,
    min_vis_rk: int,
    filtro_cd_rk: str,
    filtro_tv_rk: str,
    filtro_cat_rk: str,
) -> pd.DataFrame:
    cols_out = [
        "CD",
        "Tienda",
        "Tipo Viaje Ppal.",
        "Categoría",
        "Visitas Totales",
        "Visitas Limpias",
        "% Represent.",
        "Mediana Bultos",
        "Mediana Est. (min)",
        "P90 Est. (min)",
        "Mediana del Segmento (min)",
        "Base Segmento (Tiendas / Paradas)",
        "Demora Extra (min)",
        "Min / 100 Bultos",
    ]
    if st_clean_s3.empty:
        return pd.DataFrame(columns=cols_out)

    # Calcular Mediana del Segmento a nivel global del período (antes de filtrar por CD puntual)
    seg_ref = (
        st_clean_s3.assign(_pair=st_clean_s3["Operacion"].astype(str) + "_" + st_clean_s3["Tienda_Cod"].astype(str))
        .groupby(["Tipo_Viaje", "Categoria"], as_index=False)
        .agg(
            Mediana_Seg=("Tiempo de Estadia", "median"),
            Tiendas_Seg=("_pair", "nunique"),
            Paradas_Seg=("Tiempo de Estadia", "count"),
        )
    )
    seg_ref["Base_Seg_Str"] = (
        seg_ref["Tiendas_Seg"].map(lambda x: f"{int(x):,}".replace(",", "."))
        + " tiendas ("
        + seg_ref["Paradas_Seg"].map(lambda x: f"{int(x):,}".replace(",", "."))
        + " par.)"
    )

    # Conteo de Visitas Totales (todas las paradas en el periodo/filtros, antes de limpieza de estadía)
    tot_base = df_all_s3.copy()
    if filtro_cd_rk != "Todos":
        tot_base = tot_base[tot_base["Operacion"].astype(int).astype(str) == str(filtro_cd_rk)]
    if filtro_tv_rk != "Todos":
        tot_base = tot_base[tot_base["Tipo_Viaje"] == filtro_tv_rk]
    if filtro_cat_rk != "Todas":
        tot_base = tot_base[tot_base["Categoria"] == filtro_cat_rk]
    tot_cnt_df = (
        tot_base.groupby(["Operacion", "Tienda_Cod"], as_index=False)
        .size()
        .rename(columns={"size": "Visitas_Totales"})
    )

    rk_base = st_clean_s3.copy()
    if filtro_cd_rk != "Todos":
        rk_base = rk_base[rk_base["Operacion"].astype(int).astype(str) == str(filtro_cd_rk)]
    if filtro_tv_rk != "Todos":
        rk_base = rk_base[rk_base["Tipo_Viaje"] == filtro_tv_rk]
    if filtro_cat_rk != "Todas":
        rk_base = rk_base[rk_base["Categoria"] == filtro_cat_rk]
    if rk_base.empty:
        return pd.DataFrame(columns=cols_out)

    rk_base = rk_base.merge(seg_ref[["Tipo_Viaje", "Categoria", "Mediana_Seg", "Base_Seg_Str"]], on=["Tipo_Viaje", "Categoria"], how="left")

    g_cnt = rk_base.groupby(["Operacion", "Tienda_Cod"]).size()
    valid_pairs = g_cnt[g_cnt >= min_vis_rk].index
    if len(valid_pairs) == 0:
        return pd.DataFrame(columns=cols_out)

    rk_sub = rk_base.set_index(["Operacion", "Tienda_Cod"]).loc[valid_pairs].reset_index()

    agg_main = rk_sub.groupby(["Operacion", "Tienda_Cod"], as_index=False).agg(
        Visitas_Limpias=("Tiempo de Estadia", "count"),
        Mediana_Bultos=("Total Bultos_num", "median"),
        Mediana_Est=("Tiempo de Estadia", "median"),
        P90_Est=("Tiempo de Estadia", lambda s: float(np.percentile(s.dropna(), 90))),
        Med_Seg_Val=("Mediana_Seg", "mean"),
        Base_Seg_Main=("Base_Seg_Str", lambda s: s.value_counts().index[0]),
        Tipo_Viaje_Ppal=("Tipo_Viaje", lambda s: s.value_counts().index[0]),
        Categoria_Ppal=("Categoria", lambda s: s.value_counts().index[0]),
    )
    agg_main = agg_main.merge(tot_cnt_df, on=["Operacion", "Tienda_Cod"], how="left")
    agg_main["Visitas_Totales"] = agg_main["Visitas_Totales"].fillna(agg_main["Visitas_Limpias"]).astype(int)

    prod_sub = rk_sub[rk_sub["Total Bultos_num"].fillna(0) >= 20].copy()
    if not prod_sub.empty:
        prod_sub["_m100"] = prod_sub["Tiempo de Estadia"] / (prod_sub["Total Bultos_num"] / 100.0)
        m100_agg = prod_sub.groupby(["Operacion", "Tienda_Cod"], as_index=False)["_m100"].median()
        agg_main = agg_main.merge(m100_agg, on=["Operacion", "Tienda_Cod"], how="left")
    else:
        agg_main["_m100"] = np.nan

    agg_main["CD"] = agg_main["Operacion"].astype(int).astype(str)
    agg_main["Tienda"] = agg_main["Tienda_Cod"].astype(str)
    agg_main["Tipo Viaje Ppal."] = agg_main["Tipo_Viaje_Ppal"]
    agg_main["Categoría"] = agg_main["Categoria_Ppal"]
    agg_main["Visitas Totales"] = agg_main["Visitas_Totales"]
    agg_main["Visitas Limpias"] = agg_main["Visitas_Limpias"].astype(int)
    agg_main["% Represent."] = np.where(
        agg_main["Visitas Totales"] > 0,
        (agg_main["Visitas Limpias"] / agg_main["Visitas Totales"] * 100).round(1),
        100.0,
    )
    agg_main["Mediana Bultos"] = agg_main["Mediana_Bultos"].round(0)
    agg_main["Mediana Est. (min)"] = agg_main["Mediana_Est"].round(0)
    agg_main["P90 Est. (min)"] = agg_main["P90_Est"].round(0)
    agg_main["Mediana del Segmento (min)"] = agg_main["Med_Seg_Val"].round(1)
    agg_main["Base Segmento (Tiendas / Paradas)"] = agg_main["Base_Seg_Main"]
    agg_main["Demora Extra (min)"] = (agg_main["Mediana_Est"] - agg_main["Med_Seg_Val"]).round(1)
    agg_main["Min / 100 Bultos"] = agg_main["_m100"].round(1)

    return agg_main[cols_out].sort_values("Demora Extra (min)", ascending=False).reset_index(drop=True)



@st.cache_data(show_spinner=False)
def _rank_internos_s4_cached(
    df_all_s3: pd.DataFrame,
    st_clean_s3: pd.DataFrame,
    tr_clean_all: pd.DataFrame,
    min_vis_int: int,
    filtro_cd_int: str,
    filtro_tv_int: str,
) -> pd.DataFrame:
    cols_out = [
        "CD",
        "Interno",
        "Dominio",
        "Tipo Camión",
        "Satelital",
        "Paradas Totales",
        "Paradas Limpias",
        "% Represent.",
        "Mediana Bultos",
        "Mediana Est. (min)",
        "P90 Est. (min)",
        "Mediana Tipo Camión (min)",
        "Base Tipo Camión (Internos / Paradas)",
        "Demora Extra (min)",
        "Vel. Entre Paradas (km/h)",
        "Min / 100 Bultos",
    ]
    if st_clean_s3.empty:
        return pd.DataFrame(columns=cols_out)

    # Identificador operativo de unidad: Interno de UNIGIS si es válido; si es S/D/0, usa Dominio
    def _unit_id(df_x: pd.DataFrame) -> pd.Series:
        ic = _clean_int_str(df_x["Interno"]).fillna(df_x["Interno"].fillna("").astype(str).str.strip())
        dom = df_x["Dominio"].fillna("S/D").astype(str).str.strip().str.upper()
        return np.where(ic.isin(["", "0", "S/D", "NAN", "None"]), "S/D (" + dom + ")", ic)

    st_c = st_clean_s3.copy()
    st_c["_unit"] = _unit_id(st_c)
    st_c["Tipo_Vehiculo"] = st_c["Tipo_Vehiculo"].fillna("Sin dato").astype(str)

    # Referencia por Tipo de Camión (a nivel global del período filtrado)
    veh_ref = (
        st_c.assign(_pair=st_c["Operacion"].astype(str) + "_" + st_c["_unit"].astype(str))
        .groupby("Tipo_Vehiculo", as_index=False)
        .agg(
            Mediana_Veh=("Tiempo de Estadia", "median"),
            Units_Veh=("_pair", "nunique"),
            Paradas_Veh=("Tiempo de Estadia", "count"),
        )
    )
    veh_ref["Base_Veh_Str"] = (
        veh_ref["Units_Veh"].map(lambda x: f"{int(x):,}".replace(",", "."))
        + " int. ("
        + veh_ref["Paradas_Veh"].map(lambda x: f"{int(x):,}".replace(",", "."))
        + " par.)"
    )

    # Paradas Totales por unidad
    tot_b = df_all_s3.copy()
    tot_b["_unit"] = _unit_id(tot_b)
    if filtro_cd_int != "Todos":
        tot_b = tot_b[tot_b["Operacion"].astype(int).astype(str) == str(filtro_cd_int)]
    if filtro_tv_int != "Todos":
        tot_b = tot_b[tot_b["Tipo_Vehiculo"].fillna("Sin dato").astype(str) == filtro_tv_int]
    tot_cnt = (
        tot_b.groupby(["Operacion", "_unit"], as_index=False)
        .size()
        .rename(columns={"size": "Paradas_Totales"})
    )

    rk_b = st_c.copy()
    if filtro_cd_int != "Todos":
        rk_b = rk_b[rk_b["Operacion"].astype(int).astype(str) == str(filtro_cd_int)]
    if filtro_tv_int != "Todos":
        rk_b = rk_b[rk_b["Tipo_Vehiculo"] == filtro_tv_int]
    if rk_b.empty:
        return pd.DataFrame(columns=cols_out)

    rk_b = rk_b.merge(veh_ref[["Tipo_Vehiculo", "Mediana_Veh", "Base_Veh_Str"]], on="Tipo_Vehiculo", how="left")

    g_cnt = rk_b.groupby(["Operacion", "_unit"]).size()
    valid_u = g_cnt[g_cnt >= min_vis_int].index
    if len(valid_u) == 0:
        return pd.DataFrame(columns=cols_out)

    rk_sub = rk_b.set_index(["Operacion", "_unit"]).loc[valid_u].reset_index()

    agg_u = rk_sub.groupby(["Operacion", "_unit"], as_index=False).agg(
        Dominio=("Dominio", lambda s: " / ".join(sorted(s.dropna().astype(str).unique()))),
        Tipo_Camion=("Tipo_Vehiculo", lambda s: s.value_counts().index[0]),
        Satelital=("Satelital", lambda s: s.value_counts().index[0]),
        Paradas_Limpias=("Tiempo de Estadia", "count"),
        Mediana_Bultos=("Total Bultos_num", "median"),
        Mediana_Est=("Tiempo de Estadia", "median"),
        P90_Est=("Tiempo de Estadia", lambda s: float(np.percentile(s.dropna(), 90))),
        Med_Veh_Val=("Mediana_Veh", "mean"),
        Base_Veh_Main=("Base_Veh_Str", lambda s: s.value_counts().index[0]),
    )
    agg_u = agg_u.merge(tot_cnt, on=["Operacion", "_unit"], how="left")
    agg_u["Paradas_Totales"] = agg_u["Paradas_Totales"].fillna(agg_u["Paradas_Limpias"]).astype(int)

    # Productividad Min / 100 Bultos
    prod_u = rk_sub[rk_sub["Total Bultos_num"].fillna(0) >= 20].copy()
    if not prod_u.empty:
        prod_u["_m100"] = prod_u["Tiempo de Estadia"] / (prod_u["Total Bultos_num"] / 100.0)
        m100_u = prod_u.groupby(["Operacion", "_unit"], as_index=False)["_m100"].median()
        agg_u = agg_u.merge(m100_u, on=["Operacion", "_unit"], how="left")
    else:
        agg_u["_m100"] = np.nan

    # Velocidad mediana entre paradas por unidad
    if not tr_clean_all.empty:
        tr_u = tr_clean_all[tr_clean_all["Tramo_Tipo"] == "Entre Paradas (Tienda -> Tienda)"].copy()
        if not tr_u.empty:
            tr_u["_unit"] = _unit_id(tr_u)
            vel_u = (
                tr_u.groupby(["Operacion", "_unit"], as_index=False)["Vel_Calculada"]
                .median()
                .rename(columns={"Vel_Calculada": "_vel_med"})
            )
            agg_u = agg_u.merge(vel_u, on=["Operacion", "_unit"], how="left")
        else:
            agg_u["_vel_med"] = np.nan
    else:
        agg_u["_vel_med"] = np.nan

    agg_u["CD"] = agg_u["Operacion"].astype(int).astype(str)
    agg_u["Interno"] = agg_u["_unit"].astype(str)
    agg_u["Tipo Camión"] = agg_u["Tipo_Camion"]
    agg_u["Paradas Totales"] = agg_u["Paradas_Totales"]
    agg_u["Paradas Limpias"] = agg_u["Paradas_Limpias"].astype(int)
    agg_u["% Represent."] = np.where(
        agg_u["Paradas Totales"] > 0,
        (agg_u["Paradas Limpias"] / agg_u["Paradas Totales"] * 100).round(1),
        100.0,
    )
    agg_u["Mediana Bultos"] = agg_u["Mediana_Bultos"].round(0)
    agg_u["Mediana Est. (min)"] = agg_u["Mediana_Est"].round(0)
    agg_u["P90 Est. (min)"] = agg_u["P90_Est"].round(0)
    agg_u["Mediana Tipo Camión (min)"] = agg_u["Med_Veh_Val"].round(1)
    agg_u["Base Tipo Camión (Internos / Paradas)"] = agg_u["Base_Veh_Main"]
    agg_u["Demora Extra (min)"] = (agg_u["Mediana_Est"] - agg_u["Med_Veh_Val"]).round(1)
    agg_u["Vel. Entre Paradas (km/h)"] = agg_u["_vel_med"].round(1)
    agg_u["Min / 100 Bultos"] = agg_u["_m100"].round(1)

    return agg_u[cols_out].sort_values("Demora Extra (min)", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _filter_all_datasets_cached(
    _data_dict: dict,
    d_start: pd.Timestamp,
    d_end: pd.Timestamp,
    selected_ops_t: tuple,
    selected_months_t: tuple,
    selected_weeks_t: tuple,
    selected_trans_t: tuple,
):
    master = _data_dict["master_trips"]
    dup_detail = _data_dict["dup_detail"]
    invalid_detail = _data_dict["invalid_detail"]
    stores_gps = _data_dict["stores_gps"]
    dq_stores = _data_dict["dq_stores"]
    dq_cd_orig = _data_dict["dq_cd_orig"]
    dq_trips = _data_dict["dq_trips"]

    ops_list = list(selected_ops_t)
    mask_base = (
        (master["Fecha_Base"] >= d_start)
        & (master["Fecha_Base"] <= d_end)
        & (master["Operacion"].isin(ops_list))
    )
    if selected_months_t:
        mask_base = mask_base & (master["Mes_Label"].isin(selected_months_t))
    if selected_weeks_t:
        mask_base = mask_base & (master["Semana_Label"].isin(selected_weeks_t))
    df_filt = master[mask_base].copy()
    if selected_trans_t:
        df_filt = df_filt[df_filt["Transportista"].isin(selected_trans_t)].copy()

    mask_dup = (
        (dup_detail["Fecha_Reparto"] >= d_start)
        & (dup_detail["Fecha_Reparto"] <= d_end)
        & (dup_detail["Operacion"].isin(ops_list))
    )
    mask_inv = (
        (invalid_detail["Fecha_Reparto"] >= d_start)
        & (invalid_detail["Fecha_Reparto"] <= d_end)
        & (invalid_detail["Operacion"].isin(ops_list))
    )
    if selected_months_t:
        mask_dup = mask_dup & (dup_detail["Fecha_Reparto"].dt.strftime("%Y-%m").isin(selected_months_t))
        mask_inv = mask_inv & (invalid_detail["Fecha_Reparto"].dt.strftime("%Y-%m").isin(selected_months_t))
    if selected_weeks_t:
        dup_sem = "Sem " + dup_detail["Fecha_Reparto"].dt.to_period("W").dt.start_time.dt.strftime("%d/%m")
        inv_sem = "Sem " + invalid_detail["Fecha_Reparto"].dt.to_period("W").dt.start_time.dt.strftime("%d/%m")
        mask_dup = mask_dup & (dup_sem.isin(selected_weeks_t))
        mask_inv = mask_inv & (inv_sem.isin(selected_weeks_t))
    dup_filt = dup_detail[mask_dup].copy()
    inv_filt = invalid_detail[mask_inv].copy()

    def _f_sub(df_in: pd.DataFrame) -> pd.DataFrame:
        m = (
            (df_in["Fecha_Reparto"] >= d_start)
            & (df_in["Fecha_Reparto"] <= d_end)
            & (df_in["Operacion"].isin(ops_list))
        )
        if selected_months_t:
            m = m & (df_in["Mes_Label"].isin(selected_months_t))
        if selected_weeks_t:
            m = m & (df_in["Semana_Label"].isin(selected_weeks_t))
        if selected_trans_t:
            m = m & (df_in["Transportista"].isin(selected_trans_t))
        return df_in[m].copy()

    return (
        df_filt,
        dup_filt,
        inv_filt,
        _f_sub(stores_gps),
        _f_sub(dq_stores),
        _f_sub(dq_cd_orig),
        _f_sub(dq_trips),
        _f_sub(_data_dict["trips_pre_s5"]),
    )


@st.cache_data(show_spinner=False)
def _clean_tramos_s4_cached(
    dq_stores_filt: pd.DataFrame,
    dq_trips_filt: pd.DataFrame,
    min_vel_v: float,
    max_vel_v: float,
    min_dist_v: float,
    max_ret_cerc_v: int,
    excl_solape_v: bool,
    solo_consecutivos_v: bool,
):
    tr_st = dq_stores_filt[dq_stores_filt["visita_efectiva"]].copy()
    tr_st["Tramo_Tipo"] = np.where(
        tr_st["Orden Visita Tienda"] == 1,
        "Salida Depósito (CD -> 1ª Tienda)",
        "Entre Paradas (Tienda -> Tienda)",
    )

    r_dist_v = tr_st["KMS Parada_num"].isna() | (tr_st["KMS Parada_num"] < float(min_dist_v))
    r_time_v = tr_st["Tiempo en Viaje"].isna() | (tr_st["Tiempo en Viaje"] <= 0)
    vel_calc_raw = np.where(
        tr_st["Tiempo en Viaje"].fillna(0) > 0,
        tr_st["KMS Parada_num"].fillna(0) / (tr_st["Tiempo en Viaje"].fillna(0) / 60.0),
        0.0,
    )
    r_vel_v = (vel_calc_raw < float(min_vel_v)) | (vel_calc_raw > float(max_vel_v))
    r_solape_v = (
        (
            tr_st["dt_ini_real"].notna()
            & tr_st["prev_dt_fin_real"].notna()
            & (tr_st["dt_ini_real"] < tr_st["prev_dt_fin_real"] - pd.Timedelta(minutes=1))
        )
        if excl_solape_v
        else pd.Series(False, index=tr_st.index)
    )

    if solo_consecutivos_v:
        all_st_ordered = dq_stores_filt.sort_values(["Id Viaje", "dt_ini_real"]).copy()
        all_st_ordered["_prev_vis_ok"] = all_st_ordered.groupby("Id Viaje")["visita_efectiva"].shift(1)
        prev_ok_aligned = all_st_ordered["_prev_vis_ok"].reindex(tr_st.index)
        r_salto_v = (tr_st["Orden Visita Tienda"] > 1) & (prev_ok_aligned == False)
    else:
        r_salto_v = pd.Series(False, index=tr_st.index)

    tr_clean_all = tr_st[(~r_dist_v) & (~r_time_v) & (~r_vel_v) & (~r_solape_v) & (~r_salto_v)].copy()
    tr_clean_all["Vel_Calculada"] = (
        tr_clean_all["KMS Parada_num"] / (tr_clean_all["Tiempo en Viaje"] / 60.0)
    ).round(1)

    max_ret_vec_v = np.where(dq_trips_filt["Categoria"] == "Lejana", max_ret_cerc_v * 2, max_ret_cerc_v)
    tr_ret = dq_trips_filt[
        (~dq_trips_filt["obs_err_fin"])
        & (dq_trips_filt["KMS_Retorno"].fillna(0) >= float(min_dist_v))
        & (dq_trips_filt["Tiempo_Retorno"].fillna(0) > 0)
        & (dq_trips_filt["Tiempo_Retorno"].fillna(0) <= max_ret_vec_v)
    ].copy()
    tr_ret["Vel_Retorno"] = (tr_ret["KMS_Retorno"] / (tr_ret["Tiempo_Retorno"] / 60.0)).round(1)
    tr_ret_clean_all = tr_ret[
        (tr_ret["Vel_Retorno"] >= float(min_vel_v)) & (tr_ret["Vel_Retorno"] <= float(max_vel_v))
    ].copy()

    return tr_st, tr_clean_all, tr_ret_clean_all


@st.cache_data(show_spinner=False)
def _eval_viajes_jornadas_s5_cached(
    trips_pre_filt: pd.DataFrame,
    cd_std_items: tuple,
    min_cd_j: int,
    max_cd_real_j: int,
    min_dur_v: float,
    max_dur_v: float,
    min_dur_j: float,
    max_dur_j: float,
):
    cd_std_map = dict(cd_std_items)
    med_std_fallback = float(np.mean(list(cd_std_map.values()))) if cd_std_map else 90.0
    trips_eval_s5 = trips_pre_filt.copy()
    trips_eval_s5["Estadia_CD_Std_min"] = (
        trips_eval_s5["Operacion"].map(cd_std_map).fillna(med_std_fallback)
    )

    cd_real_ok = (
        trips_eval_s5["CD_Origen_OK"]
        & trips_eval_s5["dt_ini_cd"].notna()
        & (trips_eval_s5["Estadia_CD_Origen"].fillna(-1) >= float(min_cd_j))
        & (trips_eval_s5["Estadia_CD_Origen"].fillna(9999) <= float(max_cd_real_j))
    )

    salida_cd_est = trips_eval_s5["dt_fin_cd"].fillna(
        trips_eval_s5["Ini_Primera_Tienda"] - pd.Timedelta(minutes=30)
    )
    dt_ini_cd_imputado = salida_cd_est - pd.to_timedelta(trips_eval_s5["Estadia_CD_Std_min"], unit="m")

    trips_eval_s5["Duracion_Viaje_Real_hs"] = (
        (trips_eval_s5["Fin_Ultima_Tienda"] - trips_eval_s5["dt_ini_cd"]).dt.total_seconds() / 3600.0
    ).round(2)

    trips_eval_s5["CD_Imputado"] = ~cd_real_ok
    trips_eval_s5["dt_ini_cd_Ajust"] = pd.to_datetime(
        np.where(cd_real_ok, trips_eval_s5["dt_ini_cd"], dt_ini_cd_imputado),
        errors="coerce",
    )
    trips_eval_s5["Estadia_CD_Ajust_min"] = np.where(
        cd_real_ok,
        trips_eval_s5["Estadia_CD_Origen"],
        trips_eval_s5["Estadia_CD_Std_min"],
    )
    trips_eval_s5["Duracion_Viaje_Ajust_hs"] = (
        (trips_eval_s5["Fin_Ultima_Tienda"] - trips_eval_s5["dt_ini_cd_Ajust"]).dt.total_seconds() / 3600.0
    ).round(2)

    max_st_trip_lim = np.where(trips_eval_s5["Categoria"] == "Lejana", 360.0, 240.0)
    cond_tiendas_y_kms_ok = (
        trips_eval_s5["Todas_Tiendas_OK"]
        & (trips_eval_s5["Min_Estadia_Tienda"].fillna(0) >= 5.0)
        & (trips_eval_s5["Max_Estadia_Tienda"].fillna(9999) <= max_st_trip_lim)
        & (~trips_eval_s5["Tiene_Solape"].fillna(True))
        & (trips_eval_s5["KMS_Total"].fillna(0) > 0)
        & (trips_eval_s5["KMS_Total"].fillna(0) <= 1500)
    )

    trips_eval_s5["CD_Real_OK_Flag"] = cd_real_ok
    trips_eval_s5["Tiendas_y_KMS_OK_Flag"] = cond_tiendas_y_kms_ok

    trips_eval_s5["Viaje_OK_Real"] = (
        cond_tiendas_y_kms_ok
        & cd_real_ok
        & (trips_eval_s5["Duracion_Viaje_Real_hs"].fillna(0) >= float(min_dur_v))
        & (trips_eval_s5["Duracion_Viaje_Real_hs"].fillna(0) <= float(max_dur_v))
    )
    trips_eval_s5["Viaje_OK_Ajust"] = (
        cond_tiendas_y_kms_ok
        & trips_eval_s5["dt_ini_cd_Ajust"].notna()
        & (trips_eval_s5["Duracion_Viaje_Ajust_hs"].fillna(0) >= float(min_dur_v))
        & (trips_eval_s5["Duracion_Viaje_Ajust_hs"].fillna(0) <= float(max_dur_v))
    )



    valid_fleet_all = trips_eval_s5[trips_eval_s5["Dominio_Valido"]].copy()

    def _build_j(df_fleet: pd.DataFrame, ok_col: str, ini_col: str, dur_v_col: str) -> pd.DataFrame:
        if df_fleet.empty:
            return pd.DataFrame()
        j_df = (
            df_fleet.groupby(["Operacion", "Fecha_Reparto", "Dominio_Clean"], as_index=False)
            .agg(
                Dominio=("Dominio", "first"),
                Interno=("Interno", "first"),
                Interno_Clean=("Interno_Clean", "first"),
                Tipo_Vehiculo=("Tipo_Vehiculo", "first"),
                Transportista=("Transportista", "first"),
                Mes_Label=("Mes_Label", "first"),
                Semana_Sort=("Semana_Sort", "first"),
                Semana_Label=("Semana_Label", "first"),
                Tiene_Lejana=("_is_lejana", "max"),
                Viajes_Plan_Dia=("Id Viaje", "count"),
                Viajes_OK_Dia=(ok_col, "sum"),
                Viajes_Imputados_Dia=("CD_Imputado", "sum"),
                Inicio_Jornada=(ini_col, "min"),
                Fin_Jornada=("Fin_Ultima_Tienda", "max"),
                Suma_Horas_Viajes=(dur_v_col, "sum"),
                KMS_Dia=("KMS_Total", "sum"),
                Tiendas_Dia=("Tiendas_Plan_Viaje", "sum"),
            )
        )
        j_df["Categoria_Jornada"] = np.where(j_df["Tiene_Lejana"] == 1, "Lejana", "Cercana")
        j_df["Viajes_OK_Dia"] = j_df["Viajes_OK_Dia"].astype(int)
        j_df["Viajes_Imputados_Dia"] = j_df["Viajes_Imputados_Dia"].astype(int)
        j_df["Todos_Viajes_OK"] = j_df["Viajes_Plan_Dia"] == j_df["Viajes_OK_Dia"]
        j_df["Duracion_Jornada_hs"] = (
            (j_df["Fin_Jornada"] - j_df["Inicio_Jornada"]).dt.total_seconds() / 3600.0
        ).round(2)
        j_df["Entre_Vueltas_hs"] = (
            (j_df["Duracion_Jornada_hs"] - j_df["Suma_Horas_Viajes"]).clip(lower=0.0).round(2)
        )
        j_df["Jornada_Limpia"] = (
            j_df["Todos_Viajes_OK"]
            & (j_df["Duracion_Jornada_hs"].fillna(0) >= float(min_dur_j))
            & (j_df["Duracion_Jornada_hs"].fillna(0) <= float(max_dur_j))
        )
        return j_df

    jornadas_real_all = _build_j(valid_fleet_all, "Viaje_OK_Real", "dt_ini_cd", "Duracion_Viaje_Real_hs")
    jornadas_ajust_all = _build_j(valid_fleet_all, "Viaje_OK_Ajust", "dt_ini_cd_Ajust", "Duracion_Viaje_Ajust_hs")

    return trips_eval_s5, jornadas_real_all, jornadas_ajust_all


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
master = data["master_trips"]

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

# 4. Filtro por Operacion (CD) — vacio = todos los CDs (permite elegir 1 puntual rapido)
all_ops = sorted(master["Operacion"].dropna().unique().tolist())
selected_ops_raw = st.sidebar.multiselect(
    "Operación (CD)",
    options=all_ops,
    default=[],
    placeholder="Todos los CDs",
)
selected_ops = selected_ops_raw if selected_ops_raw else all_ops

# 5. Filtro por Transportista
all_trans = sorted(master["Transportista"].dropna().unique().tolist())
selected_trans = st.sidebar.multiselect(
    "Transportista (opcional)",
    options=all_trans,
    default=[],
    placeholder="Todos los transportistas",
)

(
    df_filt,
    dup_filt,
    inv_filt,
    gps_filt,
    dq_stores_filt,
    dq_cd_filt,
    dq_trips_filt,
    trips_pre_filt,
) = _filter_all_datasets_cached(
    data,
    d_start,
    d_end,
    tuple(selected_ops),
    tuple(selected_months),
    tuple(selected_weeks),
    tuple(selected_trans),
)

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
        "🚛 4. Camiones",
        "📅 5. Viajes y Jornadas",
    ]
)

# =============================================================================
# SOLAPA 1: VISION MACRO - CONCILIACION WMS VS TMS
# =============================================================================
with tab1:
    st.subheader("Conciliación Macro: Despachos WMS (AMREPA) vs. Viajes TMS (UNIGIS)")

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

    @_ui_fragment
    def _render_t1_temporal_fragment():
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


    _render_t1_temporal_fragment()

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
    paradas_act_tarde_cnt = int(len(act_tarde_df))
    pct_paradas_act_tarde = (paradas_act_tarde_cnt / tot_paradas * 100) if tot_paradas > 0 else 0.0
    viajes_act_tarde_cnt = int(act_tarde_df["Id Viaje"].nunique())
    viajes_act_tarde_ok_cnt = int(
        act_tarde_df.groupby("Id Viaje")["Diagnostico_GPS"]
        .apply(lambda s: int((s == "Visitada (OK)").sum()) == len(s))
        .sum()
    )
    viajes_act_tarde_fail_cnt = viajes_act_tarde_cnt - viajes_act_tarde_ok_cnt

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
                "Activación Tarde (Paradas)",
                _fmt_int(paradas_act_tarde_cnt),
                f"{_fmt_pct(pct_paradas_act_tarde)} del total de paradas",
                "#475569",
                f"En <b>{_fmt_int(viajes_act_tarde_cnt)}</b> viajes (>5 km CD): <b>{_fmt_int(viajes_act_tarde_ok_cnt)}</b> OK | <b>{_fmt_int(viajes_act_tarde_fail_cnt)}</b> c/ pérdida",
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

    @_ui_fragment
    def _render_t2_temporal_fragment():
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


    _render_t2_temporal_fragment()

    @_ui_fragment
    def _render_t2_rankings_fragment():
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

        # 3D. Detalle de paradas sin GPS para consulta puntual y exportacion
        raw_fails_df = gps_filt[gps_filt["Diagnostico_GPS"] != "Visitada (OK)"].copy()
        raw_fails_df["Causa_Parada"] = np.select(
            [
                raw_fails_df["Diagnostico_GPS"] == "Sin coordenadas",
                raw_fails_df["Diagnostico_GPS"] == "Viaje Ciego (GPS/Unidad)",
                (raw_fails_df["Diagnostico_GPS"] == "Falla Geocerca (Parcial)") & (~raw_fails_df["obs_act_tarde"]),
                (raw_fails_df["Diagnostico_GPS"] == "Falla Geocerca (Parcial)") & (raw_fails_df["obs_act_tarde"]),
            ],
            [
                "Sin coordenadas",
                "Viaje Ciego (GPS/Unidad)",
                "Falla Geozona (Activ. OK)",
                "Por Activ. Tarde (>5 km)",
            ],
            default=raw_fails_df["Diagnostico_GPS"],
        )
        raw_fails_df = raw_fails_df.sort_values(["Fecha_Reparto", "Operacion", "Ref_Clean"], ascending=[False, True, True])
        raw_fails_df["Fecha Reparto"] = raw_fails_df["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")
        raw_fails_df["Dist. Activ. (km)"] = raw_fails_df["Dist_Activacion_km"].round(2)

        raw_fails_export = raw_fails_df[
            [
                "Operacion",
                "Fecha Reparto",
                "Hora Reparto",
                "Ref_Clean",
                "Id Viaje",
                "Transportista",
                "Dominio",
                "Tipo_Vehiculo",
                "Tienda_Cod",
                "Dist. Activ. (km)",
                "Diagnostico_GPS",
                "Causa_Parada",
            ]
        ].rename(
            columns={
                "Operacion": "CD",
                "Ref_Clean": "Ref Externa Viaje",
                "Tipo_Vehiculo": "Tipo Vehículo",
                "Tienda_Cod": "Tienda",
                "Diagnostico_GPS": "Categoría Gráfico",
                "Causa_Parada": "Causa Raíz",
            }
        )

        # 3E. Auditoria de Paradas en CD Origen (1a fila CD - Carga)
        # Si el camion tuvo GPS en tiendas (Tiendas_OK_Viaje > 0) pero no marco estadia en CD,
        # significa que ya habia cruzado el limite de la zona del CD al llegar la activacion.
        cd_audit_df = dq_trips_filt.copy()
        cd_audit_df["Causa_CD"] = np.select(
            [
                cd_audit_df["CD_Origen_OK"],
                cd_audit_df["Tiendas_OK_Viaje"] == 0,
                (cd_audit_df["Tiendas_OK_Viaje"] > 0) & cd_audit_df["obs_act_tarde"],
                (cd_audit_df["Tiendas_OK_Viaje"] > 0)
                & (~cd_audit_df["obs_act_tarde"])
                & (cd_audit_df["Dist_Activacion_km"].fillna(0) > 0.5),
            ],
            [
                "Con Estadía CD (OK)",
                "Viaje Ciego (0% GPS en todo el viaje)",
                "Activación Tarde (>5 km — etiq. UNIGIS)",
                "Activación Tarde (0,5 a 5 km del CD)",
            ],
            default="Activación Tarde (<=0,5 km — fuera de barrera)",
        )

        cd_sin_dato_df = cd_audit_df[~cd_audit_df["CD_Origen_OK"]].copy()
        cd_sin_dato_export = cd_sin_dato_df[
            [
                "Operacion",
                "Fecha_Reparto",
                "Hora_Reparto",
                "Ref_Clean",
                "Id Viaje",
                "Dominio",
                "Tipo_Vehiculo",
                "Dist_Activacion_km",
                "Tiendas_OK_Viaje",
                "Tiendas_Plan_Viaje",
                "Causa_CD",
            ]
        ].copy().sort_values(["Operacion", "Fecha_Reparto", "Ref_Clean"])
        cd_sin_dato_export["Fecha Reparto"] = cd_sin_dato_export["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")
        cd_sin_dato_export["Dist. Activación (km)"] = cd_sin_dato_export["Dist_Activacion_km"].round(2)
        cd_sin_dato_export = cd_sin_dato_export[
            [
                "Operacion",
                "Fecha Reparto",
                "Hora_Reparto",
                "Ref_Clean",
                "Id Viaje",
                "Dominio",
                "Tipo_Vehiculo",
                "Dist. Activación (km)",
                "Tiendas_OK_Viaje",
                "Tiendas_Plan_Viaje",
                "Causa_CD",
            ]
        ].rename(
            columns={
                "Operacion": "CD",
                "Hora_Reparto": "Hora Reparto",
                "Ref_Clean": "Ref Externa Viaje",
                "Tipo_Vehiculo": "Tipo Vehículo",
                "Tiendas_OK_Viaje": "Tiendas OK",
                "Tiendas_Plan_Viaje": "Tiendas Plan.",
                "Causa_CD": "Causa Sin Dato CD",
            }
        )

        gps_excel_bytes = build_gps_audit_excel(
            st_rank_all, dom_rank_all, sin_coord_tbl, raw_fails_export, cd_sin_dato_export
        )
        st.download_button(
            label="📥 Descargar Auditoría GPS, Geocercas y CD Origen en Excel (.xlsx)",
            data=gps_excel_bytes,
            file_name="Auditoria_GPS_y_Geocercas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        cant_geo_rec_pairs = int((st_rank_all["Falla Geozona (Activ. OK)"] >= 2).sum())
        cant_dom_ciegos = int((dom_rank_all["Viajes Ciegos"] > 0).sum())

        gtab1, gtab2, gtab3, gtab4 = st.tabs(
            [
                f"🗺️ Tiendas a Revisar Geozona ({cant_geo_rec_pairs} recurrentes)",
                f"📡 Unidades Sin Señal GPS ({cant_dom_ciegos})",
                f"🏭 Paradas en CD Origen ({len(cd_sin_dato_df):,} sin dato)",
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

            with st.expander("🔍 Ver detalle puntual de paradas sin visita (filtrable por Fecha exacta, ej. Sáb 21/11)", expanded=False):
                fechas_disp_gps = sorted(
                    raw_fails_df["Fecha_Reparto"].dropna().dt.floor("D").unique(),
                    reverse=True,
                )
                fechas_str_opts = ["Todas"] + [pd.Timestamp(f).strftime("%d/%m/%Y") for f in fechas_disp_gps]
                df_col1, df_col2 = st.columns([1.2, 1.5])
                with df_col1:
                    fecha_puntual_sel = st.selectbox(
                        "Filtrar por Fecha Reparto:",
                        options=fechas_str_opts,
                        index=0,
                        key="s2_fecha_puntual_sel",
                    )
                with df_col2:
                    causa_puntual_sel = st.selectbox(
                        "Filtrar por Categoría / Causa:",
                        options=[
                            "Todas",
                            "Falla Geocerca (Parcial) — Total gráfico",
                            "Falla Geozona (Activ. OK)",
                            "Por Activ. Tarde (>5 km)",
                            "Viaje Ciego (GPS/Unidad)",
                            "Sin coordenadas",
                        ],
                        index=1,
                        key="s2_causa_puntual_sel",
                    )

                det_view = raw_fails_export.copy()
                if fecha_puntual_sel != "Todas":
                    det_view = det_view[det_view["Fecha Reparto"] == fecha_puntual_sel]
                if causa_puntual_sel == "Falla Geocerca (Parcial) — Total gráfico":
                    det_view = det_view[det_view["Categoría Gráfico"] == "Falla Geocerca (Parcial)"]
                elif causa_puntual_sel != "Todas":
                    det_view = det_view[det_view["Causa Raíz"] == causa_puntual_sel]

                st.caption(f"Mostrando **{len(det_view):,}** paradas que coinciden con el filtro seleccionado.")
                st.dataframe(det_view, use_container_width=True, hide_index=True)

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
            st.caption(
                "Auditoría de captura GPS en la salida del depósito (1ª fila `CD - Carga`). "
                "Cuando la interfaz batch activa el viaje después de que el camión cruzó el límite de la geozona del CD (incluso a 100 m del centroide), "
                "UNIGIS no puede registrar la estadía de carga."
            )
            tot_cd_v = len(cd_audit_df)
            ok_cd_v = int(cd_audit_df["CD_Origen_OK"].sum())
            sin_cd_v = tot_cd_v - ok_cd_v
            pct_cd_ok_v = (ok_cd_v / tot_cd_v * 100) if tot_cd_v > 0 else 0.0

            c_cd_ciego = int((cd_audit_df["Causa_CD"] == "Viaje Ciego (0% GPS en todo el viaje)").sum())
            c_cd_act5 = int((cd_audit_df["Causa_CD"] == "Activación Tarde (>5 km — etiq. UNIGIS)").sum())
            c_cd_act05 = int((cd_audit_df["Causa_CD"] == "Activación Tarde (0,5 a 5 km del CD)").sum())
            c_cd_act_cerca = int((cd_audit_df["Causa_CD"] == "Activación Tarde (<=0,5 km — fuera de barrera)").sum())
            c_cd_act_tot = c_cd_act5 + c_cd_act05 + c_cd_act_cerca

            ck1, ck2, ck3 = st.columns([1.0, 1.65, 1.05])
            with ck1:
                st.markdown(
                    _kpi_card_html(
                        "Con Estadía en CD (OK)",
                        _fmt_int(ok_cd_v),
                        f"{_fmt_pct(pct_cd_ok_v)} de {_fmt_int(tot_cd_v)} viajes",
                        "#16a34a",
                        "Activado dentro de la geozona del CD",
                    ),
                    unsafe_allow_html=True,
                )
            with ck2:
                st.markdown(
                    _kpi_card_html(
                        "Sin Dato por Activación Tarde (Salió del CD)",
                        _fmt_int(c_cd_act_tot),
                        f"{_fmt_pct((c_cd_act_tot / tot_cd_v * 100) if tot_cd_v else 0)} (GPS OK en tiendas, pero activado fuera del CD):",
                        "#d97706",
                        f"• <b>{_fmt_int(c_cd_act_cerca)}</b> <=0,5 km | <b>{_fmt_int(c_cd_act05)}</b> 0,5-5 km | <b>{_fmt_int(c_cd_act5)}</b> >5 km",
                    ),
                    unsafe_allow_html=True,
                )
            with ck3:
                st.markdown(
                    _kpi_card_html(
                        "Viaje Ciego (0% GPS)",
                        _fmt_int(c_cd_ciego),
                        f"{_fmt_pct((c_cd_ciego / tot_cd_v * 100) if tot_cd_v else 0)} sin señal GPS",
                        "#dc2626",
                        "Sin registro en CD ni en tiendas",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("**Resumen por Operación (CD Origen)**")
            rows_cd_orig_op = []
            for op_c, grp_c in cd_audit_df.groupby("Operacion"):
                vt = len(grp_c)
                vok = int(grp_c["CD_Origen_OK"].sum())
                vsin = vt - vok
                v_act_cerca = int((grp_c["Causa_CD"] == "Activación Tarde (<=0,5 km — fuera de barrera)").sum())
                v_act_05 = int((grp_c["Causa_CD"] == "Activación Tarde (0,5 a 5 km del CD)").sum())
                v_act_5 = int((grp_c["Causa_CD"] == "Activación Tarde (>5 km — etiq. UNIGIS)").sum())
                rows_cd_orig_op.append(
                    {
                        "CD": str(int(op_c)),
                        "Viajes TMS": vt,
                        "Con Estadía CD": vok,
                        "Sin Dato CD": vsin,
                        "Act. Tarde (<=0,5 km)": v_act_cerca,
                        "Act. Tarde (0,5-5 km)": v_act_05,
                        "Act. Tarde (>5 km)": v_act_5,
                        "Viaje Ciego": int((grp_c["Causa_CD"] == "Viaje Ciego (0% GPS en todo el viaje)").sum()),
                        "% Captura CD": round((vok / vt * 100) if vt > 0 else 0.0, 1),
                    }
                )
            df_cd_orig_op = pd.DataFrame(rows_cd_orig_op)
            if not df_cd_orig_op.empty:
                tot_cd_op = pd.DataFrame(
                    [
                        {
                            "CD": "TOTAL",
                            "Viajes TMS": df_cd_orig_op["Viajes TMS"].sum(),
                            "Con Estadía CD": df_cd_orig_op["Con Estadía CD"].sum(),
                            "Sin Dato CD": df_cd_orig_op["Sin Dato CD"].sum(),
                            "Act. Tarde (<=0,5 km)": df_cd_orig_op["Act. Tarde (<=0,5 km)"].sum(),
                            "Act. Tarde (0,5-5 km)": df_cd_orig_op["Act. Tarde (0,5-5 km)"].sum(),
                            "Act. Tarde (>5 km)": df_cd_orig_op["Act. Tarde (>5 km)"].sum(),
                            "Viaje Ciego": df_cd_orig_op["Viaje Ciego"].sum(),
                            "% Captura CD": round(pct_cd_ok_v, 1),
                        }
                    ]
                )
                df_cd_orig_op_disp = pd.concat([df_cd_orig_op, tot_cd_op], ignore_index=True)
                st.dataframe(
                    df_cd_orig_op_disp,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "% Captura CD": st.column_config.ProgressColumn(
                            "% Captura CD",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                            width="medium",
                        ),
                    },
                )

            with st.expander("📋 Ver detalle de viajes sin estadía en CD Origen", expanded=False):
                causa_sel_cd = st.selectbox(
                    "Filtrar por causa:",
                    options=[
                        "Todas",
                        "Activación Tarde (<=0,5 km — fuera de barrera)",
                        "Activación Tarde (0,5 a 5 km del CD)",
                        "Activación Tarde (>5 km — etiq. UNIGIS)",
                        "Viaje Ciego (0% GPS en todo el viaje)",
                    ],
                    index=0,
                    key="s2_causa_cd_sel",
                )
                view_cd_det = (
                    cd_sin_dato_export
                    if causa_sel_cd == "Todas"
                    else cd_sin_dato_export[cd_sin_dato_export["Causa Sin Dato CD"] == causa_sel_cd]
                )
                st.dataframe(view_cd_det, use_container_width=True, hide_index=True)

        with gtab4:
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


    _render_t2_rankings_fragment()

# =============================================================================
# SOLAPA 3: ESTADIAS Y PRODUCTIVIDAD EN TIENDA (LIMPIEZA PROPIA + EVOLUCION + TOP)
# =============================================================================
with tab3:
    @_ui_fragment
    def _render_tab3_fragment():
        st.subheader("Estadías y Productividad de Descarga en Tienda")
        st.caption(
            "Muestra independiente sobre todas las paradas en tienda con visita GPS efectiva "
            "(no descarta paradas si faltó estadía en el CD o en otra parada del viaje). "
            "Filtrable por Período en el menú lateral y con apertura temporal o por Tienda puntual."
        )

        with st.expander("🎛️ Parámetros de Integridad Estadística de Estadías (Ajustables)", expanded=False):
            with st.form("form_params_s3", border=False):
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    min_est_directo = st.number_input(
                        "Piso mín. Directo (1 tienda):",
                        min_value=5,
                        max_value=60,
                        value=15,
                        step=5,
                        help="Un viaje exclusivo (mediana 670 bultos) no puede durar <15 min (default: 15 min).",
                        key="s3_min_dir",
                    )
                    min_est_comp = st.number_input(
                        "Piso mín. Compartido (2 tiendas):",
                        min_value=5,
                        max_value=45,
                        value=10,
                        step=1,
                        help="Piso mínimo para viajes de 2 tiendas (default: 10 min).",
                        key="s3_min_comp",
                    )
                with sc2:
                    min_est_rutas = st.number_input(
                        "Piso mín. Rutas (3+ tiendas):",
                        min_value=1,
                        max_value=30,
                        value=5,
                        step=1,
                        help="Piso mínimo para repartos capilares de 3+ tiendas (default: 5 min).",
                        key="s3_min_rutas",
                    )
                    min_min100b_phys = st.number_input(
                        "Ritmo físico mín. (min/100 bultos):",
                        min_value=0.0,
                        max_value=10.0,
                        value=3.0,
                        step=0.5,
                        help="En cargas >=50 bultos, <3,0 min/100b (>2.000 bultos/h) indica descarga fuera del polígono o paso de largo.",
                        key="s3_min_phys",
                    )
                with sc3:
                    max_est_cerc = st.number_input(
                        "Techo máx. Directo Cercana (min):",
                        min_value=120,
                        max_value=720,
                        value=240,
                        step=30,
                        help="Compartido aplica 75% de este techo (180m) y Rutas aplica 62,5% (150m).",
                        key="s3_max_cerc",
                    )
                    max_est_lej = st.number_input(
                        "Techo máx. Directo Lejana (min):",
                        min_value=180,
                        max_value=960,
                        value=360,
                        step=30,
                        help="Compartido Lejana aplica 300m y Rutas Lejana aplica 240m.",
                        key="s3_max_lej",
                    )
                with sc4:
                    st.write("")
                    excl_solape_s3 = st.checkbox(
                        "Excluir solapamiento horario",
                        value=True,
                        help="Descarta paradas cuya hora de entrada ocurre antes de salir de la parada anterior.",
                        key="s3_excl_solape",
                    )
                    bigotes_tukey_s3 = st.checkbox(
                        "Bigotes estadísticos Tukey (Q1/Q3 ± 1.5·IQR)",
                        value=True,
                        help="Evita que 1 caso extremo estire la línea del bigote hasta el techo; desmarcalo para ver Mín–Máx completo.",
                        key="s3_tukey",
                    )
                st.form_submit_button("✅ Aplicar parámetros de limpieza")

        # Construccion cacheada de la Base Limpia de Estadias en Tienda
        st_clean_s3, clean_stats_s3 = _clean_estadias_cached(
            dq_stores_filt,
            int(min_est_directo),
            int(min_est_comp),
            int(min_est_rutas),
            float(min_min100b_phys),
            int(max_est_cerc),
            int(max_est_lej),
            bool(excl_solape_s3),
        )
        st.session_state["_s3_clean_params"] = (
            int(min_est_directo),
            int(min_est_comp),
            int(min_est_rutas),
            float(min_min100b_phys),
            int(max_est_cerc),
            int(max_est_lej),
            bool(excl_solape_s3),
        )

        # KPIs de Limpieza + Estadía + Productividad + Fases de Puerta
        tot_vis_s3 = clean_stats_s3["tot"]
        ok_vis_s3 = len(st_clean_s3)
        excl_vis_s3 = tot_vis_s3 - ok_vis_s3
        pct_ret_s3 = (ok_vis_s3 / tot_vis_s3 * 100) if tot_vis_s3 > 0 else 0.0

        est_series = st_clean_s3["Tiempo de Estadia"].dropna()
        med_est_s3 = float(est_series.median()) if len(est_series) > 0 else 0.0
        p25_est_s3 = float(np.percentile(est_series, 25)) if len(est_series) > 0 else 0.0
        p75_est_s3 = float(np.percentile(est_series, 75)) if len(est_series) > 0 else 0.0
        p90_est_s3 = float(np.percentile(est_series, 90)) if len(est_series) > 0 else 0.0

        # Productividad por Bultos (paradas con >= 20 bultos para evitar distorsiones de 1-2 bultos)
        prod_s3 = st_clean_s3[st_clean_s3["Total Bultos_num"].fillna(0) >= 20]
        if not prod_s3.empty:
            med_min_100b = float((prod_s3["Tiempo de Estadia"] / (prod_s3["Total Bultos_num"] / 100.0)).median())
        else:
            med_min_100b = 0.0
        med_bultos_s3 = float(st_clean_s3["Total Bultos_num"].median()) if ok_vis_s3 > 0 else 0.0
        bultos_hora_s3 = (6000.0 / med_min_100b) if med_min_100b > 0 else 0.0

        # Sensor de puerta trasera (pre-conectado para cuando UNIGIS active el dato)
        puerta_activa_cnt = int((st_clean_s3["Minutos Apertura y Cierre_num"].fillna(0) > 0).sum()) if ok_vis_s3 > 0 else 0

        k3_1, k3_2, k3_3, k3_4 = st.columns([1.25, 1.20, 1.20, 1.15])
        with k3_1:
            st.markdown(
                _kpi_card_html(
                    "Base Limpia Estadías",
                    _fmt_int(ok_vis_s3),
                    f"{_fmt_pct(pct_ret_s3)} de {_fmt_int(tot_vis_s3)} visitas OK ({_fmt_int(excl_vis_s3)} excl.)",
                    "#16a34a" if pct_ret_s3 >= 90 else "#d97706",
                    f"Piso: <b>{clean_stats_s3['corta']}</b> | Físico bultos: <b>{clean_stats_s3['phys']}</b> | Techo: <b>{clean_stats_s3['larga']}</b> | Solape: <b>{clean_stats_s3['solape']}</b>",
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
        # BLOQUE 1: DIAGRAMAS DE CAJAS (POR SEGMENTO, TIEMPO O TIENDA PUNTUAL)
        # -------------------------------------------------------------------------
        @_ui_fragment
        def _render_s3_boxplot_fragment():
            st.markdown("### 1. Distribución y Evolución Temporal de Estadías (Diagrama de Cajas)")

            dim_map_s3 = {
                "Tipo de Viaje (Directo / Compartido / Rutas)": "Tipo_Viaje",
                "Tienda (Evolución Temporal de Tienda puntual)": "Tienda_Puntual",
                "Evolución por Mes": "Evol_Mes",
                "Evolución por Semana": "Evol_Semana",
                "Evolución por Fecha (Día)": "Evol_Dia",
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
                "Operación (CD)": "Operacion",
            }

            bx_c1, bx_c2 = st.columns([1.35, 1.15])
            with bx_c1:
                dim_label_s3 = st.selectbox(
                    "Agrupar eje principal por (segmento, período o tienda):",
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

            df_seg_summary = pd.DataFrame()
            if st_clean_s3.empty:
                st.info("No hay paradas limpias en el rango y filtros seleccionados.")
                return df_seg_summary

            df_plot_s3 = st_clean_s3.copy()
            df_tot_plot_s3 = dq_stores_filt.copy()

            chart_title_s3 = None
            if dim_col == "Tienda_Puntual":
                tiendas_unicas_s3 = sorted(
                    df_plot_s3["Tienda_Cod"].dropna().astype(str).unique().tolist(),
                    key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
                )
                t_c1, t_c2, t_c3 = st.columns([1.1, 0.9, 1.2])
                with t_c1:
                    t_cod_sel = st.selectbox(
                        "Tienda:",
                        options=tiendas_unicas_s3,
                        index=0,
                        key="s3_tienda_puntual_sel",
                    )
                cds_para_tienda = (
                    sorted(
                        df_plot_s3[df_plot_s3["Tienda_Cod"].astype(str) == str(t_cod_sel)]["Operacion"]
                        .dropna()
                        .astype(int)
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    if t_cod_sel
                    else []
                )
                with t_c2:
                    cd_t_puntual_sel = st.selectbox(
                        "CD:",
                        options=["Todos"] + cds_para_tienda,
                        index=0,
                        key="s3_cd_tienda_puntual_sel",
                    )
                with t_c3:
                    eje_tienda_tiempo = st.radio(
                        "Eje temporal:",
                        options=["Semana", "Mes", "Fecha (Día)"],
                        index=0,
                        horizontal=True,
                        key="s3_eje_tienda_t",
                    )

                if t_cod_sel:
                    df_plot_s3 = df_plot_s3[df_plot_s3["Tienda_Cod"].astype(str) == str(t_cod_sel)].copy()
                    df_tot_plot_s3 = df_tot_plot_s3[df_tot_plot_s3["Tienda_Cod"].astype(str) == str(t_cod_sel)].copy()
                    if cd_t_puntual_sel != "Todos":
                        df_plot_s3 = df_plot_s3[df_plot_s3["Operacion"].astype(int).astype(str) == str(cd_t_puntual_sel)].copy()
                        df_tot_plot_s3 = df_tot_plot_s3[df_tot_plot_s3["Operacion"].astype(int).astype(str) == str(cd_t_puntual_sel)].copy()
                        cd_titulo_str = f"CD {cd_t_puntual_sel}"
                    else:
                        cds_de_tienda = sorted(df_plot_s3["Operacion"].dropna().astype(int).astype(str).unique().tolist())
                        cd_titulo_str = f"CD {', '.join(cds_de_tienda)}" if cds_de_tienda else "Todos los CDs"
                    chart_title_s3 = f"{cd_titulo_str} y Tienda {t_cod_sel}"

                if eje_tienda_tiempo == "Mes":
                    active_dim = "Evol_Mes"
                elif eje_tienda_tiempo == "Fecha (Día)":
                    active_dim = "Evol_Dia"
                else:
                    active_dim = "Evol_Semana"
            else:
                active_dim = dim_col

            if active_dim == "Evol_Dia" and dim_col != "Tienda_Puntual":
                sems_s3_disp = (
                    df_plot_s3[["Semana_Sort", "Semana_Label"]]
                    .drop_duplicates()
                    .sort_values("Semana_Sort")["Semana_Label"]
                    .tolist()
                )
                def_sems_s3 = sems_s3_disp[-2:] if len(sems_s3_disp) >= 2 else sems_s3_disp
                sems_s3_sel = st.multiselect(
                    "Drill-down — Seleccioná semana(s) para abrir las cajas por día (por defecto últimas 2):",
                    options=sems_s3_disp,
                    default=def_sems_s3,
                    placeholder="Todas las semanas",
                    key="s3_sems_dia_sel",
                )
                if sems_s3_sel:
                    df_plot_s3 = df_plot_s3[df_plot_s3["Semana_Label"].isin(sems_s3_sel)].copy()
                    df_tot_plot_s3 = df_tot_plot_s3[df_tot_plot_s3["Semana_Label"].isin(sems_s3_sel)].copy()

            def _prep_dim_col(df_target: pd.DataFrame, col_mode: str) -> pd.DataFrame:
                d = df_target.copy()
                if col_mode == "Evol_Mes":
                    d["_p_sort"] = d["Fecha_Reparto"].dt.to_period("M").dt.start_time
                    d[col_mode] = d["_p_sort"].dt.strftime("%Y-%m")
                elif col_mode == "Evol_Semana":
                    d[col_mode] = d["Semana_Label"]
                elif col_mode == "Evol_Dia":
                    d["_p_sort"] = d["Fecha_Reparto"].dt.floor("D")
                    d[col_mode] = d["_p_sort"].dt.dayofweek.map(dias_es) + " " + d["_p_sort"].dt.strftime("%d/%m")
                elif col_mode == "Operacion":
                    d[col_mode] = "CD " + d[col_mode].astype(str)
                elif col_mode == "Orden_Parada_Grupo":
                    d[col_mode] = np.select(
                        [
                            d["Orden Visita Tienda"] == 1,
                            d["Orden Visita Tienda"] == 2,
                            d["Orden Visita Tienda"] == 3,
                            d["Orden Visita Tienda"] >= 4,
                        ],
                        ["1ª Parada", "2ª Parada", "3ª Parada", "4ª+ Parada"],
                        default="Sin orden",
                    )
                else:
                    d[col_mode] = d[col_mode].fillna("Sin dato").astype(str)
                return d

            df_plot_s3 = _prep_dim_col(df_plot_s3, active_dim)
            df_tot_plot_s3 = _prep_dim_col(df_tot_plot_s3, active_dim)
            if sub_col is not None:
                df_plot_s3 = _prep_dim_col(df_plot_s3, sub_col)
                df_tot_plot_s3 = _prep_dim_col(df_tot_plot_s3, sub_col)

            fixed_ord_s3 = None
            if active_dim == "Tipo_Viaje":
                fixed_ord_s3 = ("Directo (1 tienda)", "Compartido (2 tiendas)", "Rutas (3+ tiendas)")
            elif active_dim == "Orden_Parada_Grupo":
                fixed_ord_s3 = ("1ª Parada", "2ª Parada", "3ª Parada", "4ª+ Parada")

            nice_dim_name = {
                "Evol_Mes": "Mes",
                "Evol_Semana": "Semana",
                "Evol_Dia": "Fecha",
                "Tipo_Viaje": "Tipo de Viaje",
                "Tipo_Vehiculo": "Tipo de Vehículo",
                "Operacion": "Operación",
                "Categoria": "Categoría",
                "Orden_Parada_Grupo": "Orden Parada",
            }.get(active_dim, active_dim)

            fig_box, df_seg_summary = _build_grouped_boxplot_cached(
                df_plot_s3,
                df_tot_plot_s3,
                val_col="Tiempo de Estadia",
                dim_col=active_dim,
                sub_col=sub_col,
                unit_suffix="min",
                y_title="Tiempo de Estadía en Tienda (minutos)",
                use_tukey=bool(bigotes_tukey_s3),
                chart_title=chart_title_s3,
                fixed_dim_order=fixed_ord_s3,
                compute_summary_table=True,
                nice_dim_name=nice_dim_name,
                summary_mode="estadias",
            )
            st.plotly_chart(fig_box, use_container_width=True)

            st.dataframe(
                df_seg_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "% Represent.": st.column_config.NumberColumn("% Represent.", format="%.1f%%"),
                },
            )

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

            st.session_state["_s3_seg_summary"] = df_seg_summary

        _render_s3_boxplot_fragment()
        df_seg_summary = st.session_state.get("_s3_seg_summary", pd.DataFrame())

        st.divider()

        @_ui_fragment
        def _render_s3_ranking_fragment():
            # -------------------------------------------------------------------------
            # BLOQUE 2: TOP TIENDAS (MAYOR DEMORA vs. MENOR ESTADIA)
            # -------------------------------------------------------------------------
            st.markdown("### 2. Ranking de Tiendas: Mayor Demora vs. Menor Estadía (vs. Mediana del Segmento)")
            st.caption(
                "Compara la **Mediana de Estadía** de cada tienda contra la **Mediana del Segmento** (`Tipo de Viaje` + `Categoría`), "
                "mostrando el segmento exacto contra el que se compara y las **Visitas Totales** vs. **Visitas Limpias**."
            )

            cds_rk_opts = ["Todos"] + (
                sorted(st_clean_s3["Operacion"].dropna().astype(int).astype(str).unique().tolist())
                if not st_clean_s3.empty
                else []
            )

            rk_c0, rk_c1, rk_c2, rk_c3, rk_c4 = st.columns([0.85, 0.85, 0.95, 1.25, 1.0])
            with rk_c0:
                filtro_cd_rk = st.selectbox(
                    "Filtrar por CD:",
                    options=cds_rk_opts,
                    index=0,
                    key="s3_cd_rk",
                )
            with rk_c1:
                top_n_rk = st.selectbox(
                    "Cantidad Top:",
                    options=[10, 15, 20, 30, 50],
                    index=1,
                    key="s3_top_n_rk",
                )
            with rk_c2:
                min_vis_rk = st.number_input(
                    "Mín. visitas limpias:",
                    min_value=1,
                    max_value=100,
                    value=5,
                    step=1,
                    key="s3_min_vis_rk",
                )
            with rk_c3:
                filtro_tv_rk = st.selectbox(
                    "Filtrar por Tipo de Viaje:",
                    options=["Todos", "Directo (1 tienda)", "Compartido (2 tiendas)", "Rutas (3+ tiendas)"],
                    index=0,
                    key="s3_tv_rk",
                )
            with rk_c4:
                filtro_cat_rk = st.selectbox(
                    "Filtrar por Categoría:",
                    options=["Todas", "Cercana", "Lejana"],
                    index=0,
                    key="s3_cat_rk",
                )

            orden_top_rk = st.radio(
                "Ordenar ranking completo por:",
                options=[
                    "Demora Extra vs. Mediana del Segmento (min)",
                    "Estadía Mediana Absoluta (min)",
                    "Productividad (Min / 100 Bultos)",
                ],
                index=0,
                horizontal=True,
                key="s3_orden_top_rk",
            )

            df_rk_tiendas = _rank_tiendas_s3_cached(
                dq_stores_filt,
                st_clean_s3,
                int(min_vis_rk),
                filtro_cd_rk,
                filtro_tv_rk,
                filtro_cat_rk,
            )

            if not df_rk_tiendas.empty:
                if orden_top_rk == "Estadía Mediana Absoluta (min)":
                    df_rk_tiendas = df_rk_tiendas.sort_values("Mediana Est. (min)", ascending=False).reset_index(drop=True)
                    metric_col = "Mediana Est. (min)"
                    metric_lbl = "Mediana Estadía (min)"
                elif orden_top_rk == "Productividad (Min / 100 Bultos)":
                    df_rk_tiendas = df_rk_tiendas.sort_values("Min / 100 Bultos", ascending=False, na_position="last").reset_index(drop=True)
                    metric_col = "Min / 100 Bultos"
                    metric_lbl = "Min / 100 Bultos"
                else:
                    metric_col = "Demora Extra (min)"
                    metric_lbl = "Demora Extra vs. Segmento (min)"

                valid_metric_df = df_rk_tiendas.dropna(subset=[metric_col]).copy()
                valid_metric_df["Seg_Corto"] = (
                    valid_metric_df["Tipo Viaje Ppal."].str.split(" ").str[0]
                    + " · "
                    + valid_metric_df["Categoría"]
                )
                valid_metric_df["Tienda_Label"] = (
                    "T." + valid_metric_df["Tienda"] + " (CD " + valid_metric_df["CD"] + " | " + valid_metric_df["Seg_Corto"] + ")"
                )

                col_top_hi, col_top_lo = st.columns(2)

                with col_top_hi:
                    top_hi = valid_metric_df.head(int(top_n_rk)).iloc[::-1].copy()
                    colors_hi = [
                        "#dc2626" if v >= 30 else ("#f59e0b" if v > 0 else "#16a34a")
                        for v in top_hi["Demora Extra (min)"]
                    ]
                    fig_hi = go.Figure(
                        data=go.Bar(
                            x=top_hi[metric_col],
                            y=top_hi["Tienda_Label"],
                            orientation="h",
                            marker_color=colors_hi,
                            text=[
                                f"+{v:.0f}m" if (metric_col == "Demora Extra (min)" and v > 0) else f"{v:.1f}" if metric_col == "Min / 100 Bultos" else f"{v:.0f}m"
                                for v in top_hi[metric_col]
                            ],
                            textposition="outside",
                            customdata=np.dstack(
                                (
                                    top_hi["Mediana Est. (min)"],
                                    top_hi["Mediana del Segmento (min)"],
                                    top_hi["Tipo Viaje Ppal."] + " — " + top_hi["Categoría"],
                                    top_hi["Base Segmento (Tiendas / Paradas)"],
                                    top_hi["Visitas Limpias"],
                                    top_hi["Visitas Totales"],
                                    top_hi["Mediana Bultos"],
                                )
                            )[0],
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Segmento: <b>%{customdata[2]}</b><br>"
                                "Mediana Tienda: <b>%{customdata[0]:.0f} min</b> | Mediana Segmento: <b>%{customdata[1]:.0f} min</b><br>"
                                "Base Segmento: %{customdata[3]}<br>"
                                "Visitas: <b>%{customdata[4]:.0f} limpias</b> de %{customdata[5]:.0f} totales | Mediana: %{customdata[6]:,.0f} bultos"
                                "<extra></extra>"
                            ),
                        )
                    )
                    max_hi = float(top_hi[metric_col].max()) if not top_hi.empty else 100.0
                    fig_hi.update_layout(
                        title=f"🔴 Top {len(top_hi)} Mayor Demora — {metric_lbl}",
                        height=max(350, len(top_hi) * 26 + 90),
                        margin=dict(l=10, r=55, t=40, b=20),
                        xaxis_title=metric_lbl,
                        yaxis_title="",
                        xaxis=dict(range=[min(0.0, float(top_hi[metric_col].min())), max_hi * 1.24]),
                    )
                    st.plotly_chart(fig_hi, use_container_width=True)

                with col_top_lo:
                    top_lo = valid_metric_df.tail(int(top_n_rk)).copy()
                    top_lo = top_lo.sort_values(metric_col, ascending=False)
                    colors_lo = [
                        "#16a34a" if v <= 0 else ("#3b82f6" if v < 15 else "#f59e0b")
                        for v in top_lo["Demora Extra (min)"]
                    ]
                    fig_lo = go.Figure(
                        data=go.Bar(
                            x=top_lo[metric_col],
                            y=top_lo["Tienda_Label"],
                            orientation="h",
                            marker_color=colors_lo,
                            text=[
                                f"{v:+.0f}m" if metric_col == "Demora Extra (min)" else f"{v:.1f}" if metric_col == "Min / 100 Bultos" else f"{v:.0f}m"
                                for v in top_lo[metric_col]
                            ],
                            textposition="outside",
                            customdata=np.dstack(
                                (
                                    top_lo["Mediana Est. (min)"],
                                    top_lo["Mediana del Segmento (min)"],
                                    top_lo["Tipo Viaje Ppal."] + " — " + top_lo["Categoría"],
                                    top_lo["Base Segmento (Tiendas / Paradas)"],
                                    top_lo["Visitas Limpias"],
                                    top_lo["Visitas Totales"],
                                    top_lo["Mediana Bultos"],
                                )
                            )[0],
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Segmento: <b>%{customdata[2]}</b><br>"
                                "Mediana Tienda: <b>%{customdata[0]:.0f} min</b> | Mediana Segmento: <b>%{customdata[1]:.0f} min</b><br>"
                                "Base Segmento: %{customdata[3]}<br>"
                                "Visitas: <b>%{customdata[4]:.0f} limpias</b> de %{customdata[5]:.0f} totales | Mediana: %{customdata[6]:,.0f} bultos"
                                "<extra></extra>"
                            ),
                        )
                    )
                    min_lo = float(top_lo[metric_col].min()) if not top_lo.empty else -30.0
                    max_lo = float(top_lo[metric_col].max()) if not top_lo.empty else 30.0
                    fig_lo.update_layout(
                        title=f"🟢 Top {len(top_lo)} Menor Estadía — {metric_lbl}",
                        height=max(350, len(top_lo) * 26 + 90),
                        margin=dict(l=10, r=55, t=40, b=20),
                        xaxis_title=metric_lbl,
                        yaxis_title="",
                        xaxis=dict(range=[min_lo * 1.26 if min_lo < 0 else 0.0, max(10.0, max_lo * 1.24)]),
                    )
                    st.plotly_chart(fig_lo, use_container_width=True)

            st.session_state["_s3_rk_tiendas"] = df_rk_tiendas

        _render_s3_ranking_fragment()
        df_rk_tiendas = st.session_state.get("_s3_rk_tiendas", pd.DataFrame())

        with st.expander("📥 Descargar Excel de Estadías y Productividad (.xlsx)", expanded=False):
            clean_s3_export = st_clean_s3.copy()
            if not clean_s3_export.empty:
                clean_s3_export["Fecha Reparto"] = clean_s3_export["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")
            estadias_excel_bytes = build_estadias_excel(
                df_seg_summary,
                df_rk_tiendas,
                clean_s3_export,
            )
            st.download_button(
                label="📥 Descargar Análisis Completo de Estadías en Tienda (.xlsx)",
                data=estadias_excel_bytes,
                file_name="Analisis_Estadias_y_Productividad_Tienda.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.dataframe(
            df_rk_tiendas,
            use_container_width=True,
            hide_index=True,
            column_config={
                "% Represent.": st.column_config.NumberColumn(
                    "% Represent.",
                    help="Visitas Limpias / Visitas Totales",
                    format="%.1f%%",
                ),
                "Demora Extra (min)": st.column_config.NumberColumn(
                    "Demora Extra (min)",
                    help="Diferencia en minutos contra la Mediana del Segmento",
                    format="%+.1f min",
                ),
                "Min / 100 Bultos": st.column_config.NumberColumn(
                    "Min / 100 Bultos",
                    format="%.1f",
                ),
            },
        )


    _render_tab3_fragment()

# =============================================================================
# SOLAPA 4: CAMIONES (VELOCIDADES, ESTADIAS POR TIPO DE CAMION Y RANKING INTERNOS)
# =============================================================================
with tab4:
    @_ui_fragment
    def _render_tab4_fragment():
        st.subheader("Camiones: Velocidades, Estadías por Tipo de Vehículo y Ranking de Internos")
        st.caption(
            "Vista integral de la flota (cruzada con `Nomina.xlsx` por patente): "
            "analiza **velocidades por tramo**, **estadías en tienda por tipo de camión** y **desempeño individual por Interno**."
        )

        cat_v_sel = st.radio(
            "Filtrar Categoría de Viaje (actualiza tarjetas y matrices de velocidad):",
            options=["Todas", "Cercana", "Lejana"],
            index=0,
            horizontal=True,
            key="s4_cat_radio",
        )

        with st.expander("🎛️ Parámetros de Depuración de Tramos y Velocidades (Ajustables)", expanded=False):
            with st.form("form_params_s4", border=False):
                vc1, vc2, vc3, vc4 = st.columns(4)
                with vc1:
                    min_vel_v = st.number_input(
                        "Velocidad mínima válida (km/h):",
                        min_value=1.0,
                        max_value=20.0,
                        value=3.0,
                        step=1.0,
                        help="Descarta tramos congestionados o detenidos que distorsionan la velocidad de circulación.",
                        key="s4_min_vel",
                    )
                    max_vel_v = st.number_input(
                        "Velocidad máxima válida (km/h):",
                        min_value=60.0,
                        max_value=200.0,
                        value=120.0,
                        step=10.0,
                        help="Descarta saltos de coordenadas GPS irreales.",
                        key="s4_max_vel",
                    )
                with vc2:
                    min_dist_v = st.number_input(
                        "Distancia mínima del tramo (km):",
                        min_value=0.1,
                        max_value=10.0,
                        value=0.5,
                        step=0.1,
                        help="Evita calcular velocidades en maniobras de <500 metros donde 1 minuto cambia mucho el cociente.",
                        key="s4_min_dist",
                    )
                    max_ret_cerc_v = st.number_input(
                        "Tiempo retorno máx. Cercana (min):",
                        min_value=120,
                        max_value=720,
                        value=360,
                        step=60,
                        help="Tope máximo de Tiempo Retorno en viajes Cercana (default 360 min = 6h). En Lejana se aplica automáticamente el doble (720 min = 12h) para filtrar viajes mal cerrados.",
                        key="s4_max_ret",
                    )
                with vc3:
                    st.write("")
                    excl_solape_v = st.checkbox(
                        "Excluir solapamiento horario",
                        value=True,
                        help="Descarta tramos cuya llegada ocurre antes de salir de la parada anterior.",
                        key="s4_excl_solape",
                    )
                    solo_consecutivos_v = st.checkbox(
                        "Solo tramos consecutivos (sin saltos)",
                        value=True,
                        help="Excluye el tramo si la parada anterior quedó 'No visitada' para no mezclar desvíos.",
                        key="s4_solo_consec",
                    )
                with vc4:
                    st.write("")
                    tukey_v = st.checkbox(
                        "Bigotes estadísticos Tukey (Q1/Q3 ± 1.5·IQR)",
                        value=True,
                        key="s4_tukey",
                    )
                st.form_submit_button("✅ Aplicar parámetros de depuración")

        tr_st, tr_clean_all, tr_ret_clean_all = _clean_tramos_s4_cached(
            dq_stores_filt,
            dq_trips_filt,
            float(min_vel_v),
            float(max_vel_v),
            float(min_dist_v),
            int(max_ret_cerc_v),
            bool(excl_solape_v),
            bool(solo_consecutivos_v),
        )
        _p_s3 = st.session_state.get("_s3_clean_params", (15, 10, 5, 3.0, 240, 360, True))
        st_clean_s3, _ = _clean_estadias_cached(dq_stores_filt, *_p_s3)

        # Aplicar filtro de Categoria elegido (Todas / Cercana / Lejana)
        if cat_v_sel != "Todas":
            tr_clean = tr_clean_all[tr_clean_all["Categoria"] == cat_v_sel].copy()
            tr_ret_clean = tr_ret_clean_all[tr_ret_clean_all["Categoria"] == cat_v_sel].copy()
            tot_tr_raw = int((tr_st["Categoria"] == cat_v_sel).sum())
        else:
            tr_clean = tr_clean_all.copy()
            tr_ret_clean = tr_ret_clean_all.copy()
            tot_tr_raw = len(tr_st)

        # KPIs de Velocidades
        ok_tr_v = len(tr_clean)
        pct_ret_v = (ok_tr_v / tot_tr_raw * 100) if tot_tr_raw > 0 else 0.0

        sal_cd_s = tr_clean[tr_clean["Tramo_Tipo"] == "Salida Depósito (CD -> 1ª Tienda)"]["Vel_Calculada"]
        ent_par_s = tr_clean[tr_clean["Tramo_Tipo"] == "Entre Paradas (Tienda -> Tienda)"]["Vel_Calculada"]
        lleg_cd_s = tr_ret_clean["Vel_Retorno"]

        med_sal_cd = float(sal_cd_s.median()) if len(sal_cd_s) > 0 else 0.0
        med_ent_par = float(ent_par_s.median()) if len(ent_par_s) > 0 else 0.0
        med_lleg_cd = float(lleg_cd_s.median()) if len(lleg_cd_s) > 0 else 0.0
        p95_vel_flota = float(np.percentile(tr_clean["Vel_Calculada"], 95)) if ok_tr_v > 0 else 0.0
        p98_vel_flota = float(np.percentile(tr_clean["Vel_Calculada"], 98)) if ok_tr_v > 0 else 0.0

        kv1, kv2, kv3, kv4 = st.columns(4)
        with kv1:
            st.markdown(
                _kpi_card_html(
                    "Velocidad Entre Paradas",
                    f"{med_ent_par:.1f} km/h",
                    f"Mediana en {_fmt_int(len(ent_par_s))} tramos Tienda->Tienda",
                    "#16a34a",
                    f"Base limpia: <b>{_fmt_int(ok_tr_v)}</b> tramos ({_fmt_pct(pct_ret_v)})",
                ),
                unsafe_allow_html=True,
            )
        with kv2:
            st.markdown(
                _kpi_card_html(
                    "Velocidad Depósito Salida",
                    f"{med_sal_cd:.1f} km/h",
                    f"Mediana en {_fmt_int(len(sal_cd_s))} tramos CD->1ª Tienda",
                    "#1d4ed8",
                    "Camión cargado saliendo del CD",
                ),
                unsafe_allow_html=True,
            )
        with kv3:
            st.markdown(
                _kpi_card_html(
                    "Velocidad Depósito Llegada",
                    f"{med_lleg_cd:.1f} km/h",
                    f"Mediana en {_fmt_int(len(lleg_cd_s))} retornos limpios",
                    "#0f172a",
                    "Calculado: KMS Retorno / Tiempo Retorno",
                ),
                unsafe_allow_html=True,
            )
        with kv4:
            st.markdown(
                _kpi_card_html(
                    "Velocidad Máxima (P95 / P98)",
                    f"{p95_vel_flota:.0f} km/h",
                    f"Percentil 98: {p98_vel_flota:.0f} km/h",
                    "#d97706",
                    "Referencia operativa para alarmas GPS",
                ),
                unsafe_allow_html=True,
            )

        st.divider()

        # -------------------------------------------------------------------------
        # BLOQUE 1: VELOCIDADES (A: COMPARATIVO POR CATEGORIA | B: MATRIZ CD x CAMION)
        # -------------------------------------------------------------------------
        st.markdown("### 1. Resumen y Matrices de Velocidades")

        st.markdown("**A. Comparativo Global por Categoría de Viaje (Cercana vs. Lejana)**")
        cat_comp_rows = []
        for c_name in ["Cercana", "Lejana"]:
            sub_c = tr_clean_all[tr_clean_all["Categoria"] == c_name]
            sub_r = tr_ret_clean_all[tr_ret_clean_all["Categoria"] == c_name]
            tot_c_raw = int((tr_st["Categoria"] == c_name).sum())
            tot_r_raw = int((dq_trips_filt["Categoria"] == c_name).sum())
            if sub_c.empty and sub_r.empty:
                continue
            s_sal = sub_c[sub_c["Tramo_Tipo"] == "Salida Depósito (CD -> 1ª Tienda)"]["Vel_Calculada"]
            s_ent = sub_c[sub_c["Tramo_Tipo"] == "Entre Paradas (Tienda -> Tienda)"]["Vel_Calculada"]
            s_lleg = sub_r["Vel_Retorno"]
            pct_tr_c = (len(sub_c) / tot_c_raw * 100) if tot_c_raw > 0 else 0.0
            pct_ret_c = (len(sub_r) / tot_r_raw * 100) if tot_r_raw > 0 else 0.0
            cat_comp_rows.append(
                {
                    "Categoría": c_name,
                    "Vel. Entre Paradas (km/h)": round(float(s_ent.median()), 1) if len(s_ent) > 0 else None,
                    "Vel. Depósito Salida (km/h)": round(float(s_sal.median()), 1) if len(s_sal) > 0 else None,
                    "Vel. Depósito Llegada (km/h)": round(float(s_lleg.median()), 1) if len(s_lleg) > 0 else None,
                    "Vel. Máxima P95 (km/h)": round(float(np.percentile(sub_c["Vel_Calculada"], 95)), 0) if len(sub_c) > 0 else None,
                    "Tramos Limpios / Totales": f"{len(sub_c):,} / {tot_c_raw:,} ({pct_tr_c:.0f}%)".replace(",", "."),
                    "Retornos Limpios / Totales": f"{len(sub_r):,} / {tot_r_raw:,} ({pct_ret_c:.0f}%)".replace(",", "."),
                }
            )
        if cat_comp_rows:
            st.dataframe(pd.DataFrame(cat_comp_rows), use_container_width=True, hide_index=True)

        st.markdown(f"**B. Matriz por Operación (CD) y Tipo de Camión (Nómina) — Filtro Categoría: `{cat_v_sel}`**")
        st.caption(
            "Cada celda muestra: **Mediana** y entre paréntesis **`limpios / totales (% representatividad)`** "
            "para evaluar de inmediato si la muestra es representativa."
        )

        def _build_vel_matrix(
            df_clean_src: pd.DataFrame,
            df_tot_src: pd.DataFrame,
            val_col: str,
            agg_fn="median",
            p_val: int = 95,
        ) -> pd.DataFrame:
            if df_clean_src.empty:
                return pd.DataFrame(columns=["Operación"])
            d_tmp = df_clean_src.copy()
            d_tmp["CD_Str"] = "CD " + d_tmp["Operacion"].astype(str)
            if agg_fn == "median":
                piv_v = d_tmp.pivot_table(index="CD_Str", columns="Tipo_Vehiculo", values=val_col, aggfunc="median").round(1)
            else:
                piv_v = d_tmp.pivot_table(
                    index="CD_Str",
                    columns="Tipo_Vehiculo",
                    values=val_col,
                    aggfunc=lambda s: round(float(np.percentile(s.dropna(), p_val)), 1) if len(s.dropna()) > 0 else np.nan,
                )
            piv_n = d_tmp.pivot_table(index="CD_Str", columns="Tipo_Vehiculo", values=val_col, aggfunc="count").fillna(0).astype(int)

            # Totales antes de depuración de velocidad/distancia para calcular % representatividad
            d_tot = df_tot_src.copy()
            d_tot["CD_Str"] = "CD " + d_tot["Operacion"].astype(str)
            piv_tot = d_tot.pivot_table(index="CD_Str", columns="Tipo_Vehiculo", values="Operacion", aggfunc="count").fillna(0).astype(int)
            piv_tot = piv_tot.reindex(index=piv_v.index, columns=piv_v.columns, fill_value=0)

            out_m = pd.DataFrame(index=piv_v.index)
            for col_c in piv_v.columns:
                cells = []
                for r_i in piv_v.index:
                    val_num = piv_v.loc[r_i, col_c]
                    n_clean = int(piv_n.loc[r_i, col_c])
                    n_tot = max(n_clean, int(piv_tot.loc[r_i, col_c]))
                    if pd.notna(val_num) and n_clean > 0:
                        pct_rep = int(round(n_clean / n_tot * 100)) if n_tot > 0 else 100
                        s_clean = f"{n_clean:,}".replace(",", ".")
                        s_tot = f"{n_tot:,}".replace(",", ".")
                        cells.append(f"{val_num:.1f} km/h ({s_clean}/{s_tot} · {pct_rep}%)")
                    else:
                        cells.append("—")
                out_m[col_c] = cells
            return out_m.reset_index().rename(columns={"CD_Str": "Operación"})

        # Base total por tramo = todas las paradas/viajes planificados en UNIGIS (incluyendo los que no marcaron GPS o salieron tarde)
        all_stores_v = dq_stores_filt.copy()
        all_stores_v["Tramo_Tipo"] = np.where(
            all_stores_v["Orden Visita Tienda"] == 1,
            "Salida Depósito (CD -> 1ª Tienda)",
            "Entre Paradas (Tienda -> Tienda)",
        )
        all_stores_cat = all_stores_v[all_stores_v["Categoria"] == cat_v_sel] if cat_v_sel != "Todas" else all_stores_v
        tr_trips_cat = dq_trips_filt[dq_trips_filt["Categoria"] == cat_v_sel] if cat_v_sel != "Todas" else dq_trips_filt

        df_vp_src = tr_clean[tr_clean["Tramo_Tipo"] == "Entre Paradas (Tienda -> Tienda)"]
        df_vp_tot = all_stores_cat[all_stores_cat["Tramo_Tipo"] == "Entre Paradas (Tienda -> Tienda)"]

        df_vs_src = tr_clean[tr_clean["Tramo_Tipo"] == "Salida Depósito (CD -> 1ª Tienda)"]
        df_vs_tot = all_stores_cat[all_stores_cat["Tramo_Tipo"] == "Salida Depósito (CD -> 1ª Tienda)"]

        mat_vp = _build_vel_matrix(df_vp_src, df_vp_tot, "Vel_Calculada", "median")
        mat_vs = _build_vel_matrix(df_vs_src, df_vs_tot, "Vel_Calculada", "median")
        mat_vr = _build_vel_matrix(tr_ret_clean, tr_trips_cat, "Vel_Retorno", "median")
        mat_vmax = _build_vel_matrix(tr_clean, all_stores_cat, "Vel_Calculada", "percentile", p_val=95)

        vtab1, vtab2, vtab3, vtab4 = st.tabs(
            [
                "🛣️ Velocidad Entre Paradas (Tienda -> Tienda)",
                "🏭 Velocidad Depósito Salida (CD -> 1ª Tienda)",
                "🔄 Velocidad Depósito Llegada (Última -> CD)",
                "⚡ Velocidad Máxima (Percentil 95)",
            ]
        )

        with vtab1:
            st.caption("Mediana de velocidad real entre paradas consecutivas de entrega (limpios / totales del tramo).")
            st.dataframe(mat_vp, use_container_width=True, hide_index=True)

        with vtab2:
            st.caption("Mediana de velocidad real desde la salida del CD hasta la primera tienda (camión cargado).")
            st.dataframe(mat_vs, use_container_width=True, hide_index=True)

        with vtab3:
            st.caption(
                "Mediana de velocidad real desde la última tienda hasta el cierre en el CD, "
                "calculada a partir de `KMS Retorno / Tiempo Retorno` de cada viaje válido en UNIGIS."
            )
            st.dataframe(mat_vr, use_container_width=True, hide_index=True)

        with vtab4:
            st.caption("Percentil 95 de velocidad real observada por tipo de vehículo y operación.")
            st.dataframe(mat_vmax, use_container_width=True, hide_index=True)

        st.divider()

        # -------------------------------------------------------------------------
        # BLOQUE 2: DIAGRAMAS DE CAJAS DE VELOCIDADES (SEGMENTO O EVOLUCION TEMPORAL)
        # -------------------------------------------------------------------------
        @_ui_fragment
        def _render_s4_vel_boxplot_fragment():
            st.markdown("### 2. Distribución y Evolución Temporal de Velocidades (Diagrama de Cajas)")

            vdim_map = {
                "Tipo de Vehículo (Nómina)": "Tipo_Vehiculo",
                "Evolución por Mes": "Evol_Mes",
                "Evolución por Semana": "Evol_Semana",
                "Evolución por Fecha (Día)": "Evol_Dia",
                "Operación (CD)": "Operacion",
                "Categoría (Cercana vs. Lejana)": "Categoria",
            }
            vdim_lbl = st.selectbox(
                "Agrupar eje principal por (segmento o período):",
                options=list(vdim_map.keys()),
                index=0,
                key="s4_dim_box",
            )
            vdim_col = vdim_map[vdim_lbl]

            if not tr_clean.empty:
                df_vplot = tr_clean.copy()
                if vdim_col == "Evol_Dia":
                    sems_v4_disp = (
                        df_vplot[["Semana_Sort", "Semana_Label"]]
                        .drop_duplicates()
                        .sort_values("Semana_Sort")["Semana_Label"]
                        .tolist()
                    )
                    def_sems_v4 = sems_v4_disp[-2:] if len(sems_v4_disp) >= 2 else sems_v4_disp
                    sems_v4_sel = st.multiselect(
                        "Drill-down — Seleccioná semana(s) para abrir velocidades por día (por defecto últimas 2):",
                        options=sems_v4_disp,
                        default=def_sems_v4,
                        placeholder="Todas las semanas",
                        key="s4_sems_dia_sel",
                    )
                    if sems_v4_sel:
                        df_vplot = df_vplot[df_vplot["Semana_Label"].isin(sems_v4_sel)].copy()

                if vdim_col == "Evol_Mes":
                    df_vplot["_p_sort"] = df_vplot["Fecha_Reparto"].dt.to_period("M").dt.start_time
                    df_vplot[vdim_col] = df_vplot["_p_sort"].dt.strftime("%Y-%m")
                elif vdim_col == "Evol_Semana":
                    df_vplot[vdim_col] = df_vplot["Semana_Label"]
                elif vdim_col == "Evol_Dia":
                    df_vplot["_p_sort"] = df_vplot["Fecha_Reparto"].dt.floor("D")
                    df_vplot[vdim_col] = (
                        df_vplot["_p_sort"].dt.dayofweek.map(dias_es)
                        + " "
                        + df_vplot["_p_sort"].dt.strftime("%d/%m")
                    )
                elif vdim_col == "Operacion":
                    df_vplot[vdim_col] = "CD " + df_vplot[vdim_col].astype(str)
                else:
                    df_vplot[vdim_col] = df_vplot[vdim_col].fillna("Sin dato").astype(str)

                fig_vbox, _ = _build_grouped_boxplot_cached(
                    df_vplot,
                    None,
                    val_col="Vel_Calculada",
                    dim_col=vdim_col,
                    sub_col="Tramo_Tipo",
                    unit_suffix="km/h",
                    y_title="Velocidad Real (km/h)",
                    use_tukey=bool(tukey_v),
                    fixed_sub_order=("Salida Depósito (CD -> 1ª Tienda)", "Entre Paradas (Tienda -> Tienda)"),
                    compute_summary_table=False,
                )
                st.plotly_chart(fig_vbox, use_container_width=True)

        _render_s4_vel_boxplot_fragment()

        st.divider()

        # -------------------------------------------------------------------------
        # BLOQUE 3: ESTADIAS EN TIENDA POR OPERACION (CD) Y TIPO DE CAMION (NOMINA)
        # -------------------------------------------------------------------------
        st.markdown("### 3. Estadías en Tienda por Operación (CD) y Tipo de Camión (Nómina)")
        st.caption(
            "Mediana de tiempo de descarga en tienda y representatividad **`limpios / totales (%)`** según el porte del vehículo (`Nomina.xlsx`)."
        )

        mat_display_s4 = pd.DataFrame()
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
            piv_tot_veh = (
                dq_stores_filt.assign(CD_Str="CD " + dq_stores_filt["Operacion"].astype(str))
                .pivot_table(
                    index="CD_Str",
                    columns="Tipo_Vehiculo",
                    values="Operacion",
                    aggfunc="count",
                )
                .fillna(0)
                .astype(int)
                .reindex(index=piv_med_veh.index, columns=piv_med_veh.columns, fill_value=0)
            )

            mat_display_s4 = pd.DataFrame(index=piv_med_veh.index)
            for col_v in piv_med_veh.columns:
                cells_est = []
                for idx_r in piv_med_veh.index:
                    med_v = piv_med_veh.loc[idx_r, col_v]
                    nc = int(piv_cnt_veh.loc[idx_r, col_v])
                    nt = max(nc, int(piv_tot_veh.loc[idx_r, col_v]))
                    if pd.notna(med_v) and nc > 0:
                        pct_r = int(round(nc / nt * 100)) if nt > 0 else 100
                        sc = f"{nc:,}".replace(",", ".")
                        st_str = f"{nt:,}".replace(",", ".")
                        cells_est.append(f"{int(med_v)} min ({sc}/{st_str} · {pct_r}%)")
                    else:
                        cells_est.append("—")
                mat_display_s4[col_v] = cells_est
            mat_display_s4 = mat_display_s4.reset_index().rename(columns={"CD_Str": "Operación"})
            st.dataframe(mat_display_s4, use_container_width=True, hide_index=True)

        st.divider()

        @_ui_fragment
        def _render_s4_internos_fragment():
            # -------------------------------------------------------------------------
            # BLOQUE 4: RANKING DE INTERNOS / UNIDADES (MAYOR DEMORA vs. MAS AGILES)
            # -------------------------------------------------------------------------
            st.markdown("### 4. Ranking de Internos: Mayor Estadía vs. Más Ágiles (vs. Mediana del Tipo de Camión)")
            st.caption(
                "Compara la **Mediana de Estadía** de cada Interno contra la **Mediana de su Tipo de Camión** (`CHASIS`, `BALANCIN`, `SEMI`, `AVISPA`, etc.), "
                "mostrando también sus **Paradas Totales** vs. **Paradas Limpias** y su **Velocidad Entre Paradas**."
            )

            cds_int_opts = ["Todos"] + (
                sorted(st_clean_s3["Operacion"].dropna().astype(int).astype(str).unique().tolist())
                if not st_clean_s3.empty
                else []
            )
            tv_int_opts = ["Todos"] + (
                sorted(st_clean_s3["Tipo_Vehiculo"].dropna().astype(str).unique().tolist())
                if not st_clean_s3.empty
                else []
            )

            ic0, ic1, ic2, ic3 = st.columns([0.9, 0.9, 1.0, 1.3])
            with ic0:
                filtro_cd_int = st.selectbox(
                    "Filtrar por CD:",
                    options=cds_int_opts,
                    index=0,
                    key="s4_cd_int",
                )
            with ic1:
                top_n_int = st.selectbox(
                    "Cantidad Top:",
                    options=[10, 15, 20, 30, 50],
                    index=1,
                    key="s4_top_n_int",
                )
            with ic2:
                min_vis_int = st.number_input(
                    "Mín. paradas limpias:",
                    min_value=1,
                    max_value=200,
                    value=10,
                    step=5,
                    key="s4_min_vis_int",
                )
            with ic3:
                filtro_tv_int = st.selectbox(
                    "Filtrar por Tipo de Camión:",
                    options=tv_int_opts,
                    index=0,
                    key="s4_tv_int",
                )

            orden_top_int = st.radio(
                "Ordenar ranking de internos por:",
                options=[
                    "Demora Extra vs. Mediana Tipo Camión (min)",
                    "Estadía Mediana Absoluta (min)",
                    "Productividad (Min / 100 Bultos)",
                    "Velocidad Entre Paradas (km/h)",
                ],
                index=0,
                horizontal=True,
                key="s4_orden_top_int",
            )

            df_rk_internos = _rank_internos_s4_cached(
                dq_stores_filt,
                st_clean_s3,
                tr_clean_all,
                int(min_vis_int),
                filtro_cd_int,
                filtro_tv_int,
            )

            if not df_rk_internos.empty:
                if orden_top_int == "Estadía Mediana Absoluta (min)":
                    df_rk_internos = df_rk_internos.sort_values("Mediana Est. (min)", ascending=False).reset_index(drop=True)
                    m_col_i = "Mediana Est. (min)"
                    m_lbl_i = "Mediana Estadía (min)"
                elif orden_top_int == "Productividad (Min / 100 Bultos)":
                    df_rk_internos = df_rk_internos.sort_values("Min / 100 Bultos", ascending=False, na_position="last").reset_index(drop=True)
                    m_col_i = "Min / 100 Bultos"
                    m_lbl_i = "Min / 100 Bultos"
                elif orden_top_int == "Velocidad Entre Paradas (km/h)":
                    df_rk_internos = df_rk_internos.sort_values("Vel. Entre Paradas (km/h)", ascending=False, na_position="last").reset_index(drop=True)
                    m_col_i = "Vel. Entre Paradas (km/h)"
                    m_lbl_i = "Velocidad Entre Paradas (km/h)"
                else:
                    m_col_i = "Demora Extra (min)"
                    m_lbl_i = "Demora Extra vs. Tipo Camión (min)"

                valid_int_df = df_rk_internos.dropna(subset=[m_col_i]).copy()
                valid_int_df["Int_Label"] = (
                    "Int. "
                    + valid_int_df["Interno"]
                    + " ("
                    + valid_int_df["Dominio"]
                    + " | CD "
                    + valid_int_df["CD"]
                    + " · "
                    + valid_int_df["Tipo Camión"]
                    + ")"
                )

                col_int_hi, col_int_lo = st.columns(2)

                with col_int_hi:
                    top_int_hi = valid_int_df.head(int(top_n_int)).iloc[::-1].copy()
                    c_int_hi = [
                        "#1d4ed8"
                        if m_col_i == "Vel. Entre Paradas (km/h)"
                        else ("#dc2626" if v >= 20 else ("#f59e0b" if v > 0 else "#16a34a"))
                        for v in top_int_hi["Demora Extra (min)"]
                    ]
                    fig_int_hi = go.Figure(
                        data=go.Bar(
                            x=top_int_hi[m_col_i],
                            y=top_int_hi["Int_Label"],
                            orientation="h",
                            marker_color=c_int_hi,
                            text=[
                                f"+{v:.0f}m"
                                if (m_col_i == "Demora Extra (min)" and v > 0)
                                else f"{v:.1f}"
                                if m_col_i in {"Min / 100 Bultos", "Vel. Entre Paradas (km/h)"}
                                else f"{v:.0f}m"
                                for v in top_int_hi[m_col_i]
                            ],
                            textposition="outside",
                            customdata=np.dstack(
                                (
                                    top_int_hi["Mediana Est. (min)"],
                                    top_int_hi["Mediana Tipo Camión (min)"],
                                    top_int_hi["Tipo Camión"],
                                    top_int_hi["Base Tipo Camión (Internos / Paradas)"],
                                    top_int_hi["Paradas Limpias"],
                                    top_int_hi["Paradas Totales"],
                                    top_int_hi["Vel. Entre Paradas (km/h)"].fillna(0),
                                )
                            )[0],
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Tipo Camión: <b>%{customdata[2]}</b><br>"
                                "Mediana Interno: <b>%{customdata[0]:.0f} min</b> | Mediana Tipo Camión: <b>%{customdata[1]:.0f} min</b><br>"
                                "Referencia formada por: %{customdata[3]}<br>"
                                "Paradas: <b>%{customdata[4]:.0f} limpias</b> de %{customdata[5]:.0f} totales | Vel. Mediana: %{customdata[6]:.1f} km/h"
                                "<extra></extra>"
                            ),
                        )
                    )
                    max_i_hi = float(top_int_hi[m_col_i].max()) if not top_int_hi.empty else 100.0
                    title_hi_int = (
                        f"⚡ Top {len(top_int_hi)} Mayor Velocidad — {m_lbl_i}"
                        if m_col_i == "Vel. Entre Paradas (km/h)"
                        else f"🔴 Top {len(top_int_hi)} Mayor Estadía — {m_lbl_i}"
                    )
                    fig_int_hi.update_layout(
                        title=title_hi_int,
                        height=max(350, len(top_int_hi) * 26 + 90),
                        margin=dict(l=10, r=55, t=40, b=20),
                        xaxis_title=m_lbl_i,
                        yaxis_title="",
                        xaxis=dict(range=[min(0.0, float(top_int_hi[m_col_i].min())), max_i_hi * 1.24]),
                    )
                    st.plotly_chart(fig_int_hi, use_container_width=True)

                with col_int_lo:
                    top_int_lo = valid_int_df.tail(int(top_n_int)).copy()
                    top_int_lo = top_int_lo.sort_values(m_col_i, ascending=False)
                    c_int_lo = [
                        "#f59e0b"
                        if m_col_i == "Vel. Entre Paradas (km/h)"
                        else ("#16a34a" if v <= 0 else ("#3b82f6" if v < 15 else "#f59e0b"))
                        for v in top_int_lo["Demora Extra (min)"]
                    ]
                    fig_int_lo = go.Figure(
                        data=go.Bar(
                            x=top_int_lo[m_col_i],
                            y=top_int_lo["Int_Label"],
                            orientation="h",
                            marker_color=c_int_lo,
                            text=[
                                f"{v:+.0f}m"
                                if m_col_i == "Demora Extra (min)"
                                else f"{v:.1f}"
                                if m_col_i in {"Min / 100 Bultos", "Vel. Entre Paradas (km/h)"}
                                else f"{v:.0f}m"
                                for v in top_int_lo[m_col_i]
                            ],
                            textposition="outside",
                            customdata=np.dstack(
                                (
                                    top_int_lo["Mediana Est. (min)"],
                                    top_int_lo["Mediana Tipo Camión (min)"],
                                    top_int_lo["Tipo Camión"],
                                    top_int_lo["Base Tipo Camión (Internos / Paradas)"],
                                    top_int_lo["Paradas Limpias"],
                                    top_int_lo["Paradas Totales"],
                                    top_int_lo["Vel. Entre Paradas (km/h)"].fillna(0),
                                )
                            )[0],
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Tipo Camión: <b>%{customdata[2]}</b><br>"
                                "Mediana Interno: <b>%{customdata[0]:.0f} min</b> | Mediana Tipo Camión: <b>%{customdata[1]:.0f} min</b><br>"
                                "Referencia formada por: %{customdata[3]}<br>"
                                "Paradas: <b>%{customdata[4]:.0f} limpias</b> de %{customdata[5]:.0f} totales | Vel. Mediana: %{customdata[6]:.1f} km/h"
                                "<extra></extra>"
                            ),
                        )
                    )
                    min_i_lo = float(top_int_lo[m_col_i].min()) if not top_int_lo.empty else -30.0
                    max_i_lo = float(top_int_lo[m_col_i].max()) if not top_int_lo.empty else 30.0
                    title_lo_int = (
                        f"🐢 Top {len(top_int_lo)} Menor Velocidad — {m_lbl_i}"
                        if m_col_i == "Vel. Entre Paradas (km/h)"
                        else f"🟢 Top {len(top_int_lo)} Más Ágiles — {m_lbl_i}"
                    )
                    fig_int_lo.update_layout(
                        title=title_lo_int,
                        height=max(350, len(top_int_lo) * 26 + 90),
                        margin=dict(l=10, r=55, t=40, b=20),
                        xaxis_title=m_lbl_i,
                        yaxis_title="",
                        xaxis=dict(range=[min_i_lo * 1.26 if min_i_lo < 0 else 0.0, max(10.0, max_i_lo * 1.24)]),
                    )
                    st.plotly_chart(fig_int_lo, use_container_width=True)

            st.session_state["_s4_rk_internos"] = df_rk_internos

        _render_s4_internos_fragment()
        df_rk_internos = st.session_state.get("_s4_rk_internos", pd.DataFrame())

        with st.expander("📥 Descargar Excel Completo de Camiones (.xlsx)", expanded=False):
            tr_clean_exp = tr_clean.copy()
            if not tr_clean_exp.empty:
                tr_clean_exp["Fecha Reparto"] = tr_clean_exp["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")
            vel_excel_bytes = build_camiones_excel(
                mat_vp,
                mat_vs,
                mat_vr,
                mat_vmax,
                mat_display_s4,
                df_rk_internos,
                tr_clean_exp,
            )
            st.download_button(
                label="📥 Descargar Reporte de Camiones (Velocidades, Estadías y Ranking Internos) (.xlsx)",
                data=vel_excel_bytes,
                file_name="Analisis_Camiones_Velocidades_y_Estadias.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.dataframe(
            df_rk_internos,
            use_container_width=True,
            hide_index=True,
            column_config={
                "% Represent.": st.column_config.NumberColumn(
                    "% Represent.",
                    help="Paradas Limpias / Paradas Totales",
                    format="%.1f%%",
                ),
                "Demora Extra (min)": st.column_config.NumberColumn(
                    "Demora Extra (min)",
                    help="Diferencia en minutos contra la Mediana de su Tipo de Camión",
                    format="%+.1f min",
                ),
                "Vel. Entre Paradas (km/h)": st.column_config.NumberColumn(
                    "Vel. Entre Paradas (km/h)",
                    format="%.1f km/h",
                ),
                "Min / 100 Bultos": st.column_config.NumberColumn(
                    "Min / 100 Bultos",
                    format="%.1f",
                ),
            },
        )


    _render_tab4_fragment()

# =============================================================================
# SOLAPA 5: VIAJES Y JORNADAS (1 vs 2+ VUELTAS CON ESTADIA CD AJUSTABLE)
# =============================================================================
with tab5:
    @_ui_fragment
    def _render_tab5_fragment():
        st.subheader("Viajes y Jornadas Diarias (1 Vuelta vs. 2+ Vueltas)")
        st.caption(
            "Permite evaluar **dos escenarios**: **Solo Datos Reales** (únicamente viajes con entrada/salida real en CD y estadía <= 3 hs) "
            "o **Base Estimada con Estadía CD Ajustable** (imputando el tiempo estándar de carga por CD cuando falta el dato por interfaz "
            "o cuando el camión guardó dentro del predio > 3 hs). Todos los gráficos y tablas inferiores responden al interruptor."
        )

        # Panel de control principal: Interruptor Real vs Ajustado
        ctrl_s5_top1, ctrl_s5_top2 = st.columns([1.35, 1.65])
        with ctrl_s5_top1:
            usar_estadia_cd_ajust = st.toggle(
                "🔄 Aplicar Estadía CD Ajustable (cuando falta dato o es > 3 hs)",
                value=False,
                help=(
                    "APAGADO (Real): usa únicamente los viajes que registraron entrada/salida real en el CD dentro del rango válido. "
                    "ENCENDIDO (Estimado): si un viaje tiene 100% de tiendas OK pero le falta estadía en CD o supera 3 hs (guarda en predio), "
                    "imputa la estadía estándar configurada para ese CD y recalcula inicio de viaje y jornada."
                ),
                key="s5_usar_cd_ajust",
            )
        with ctrl_s5_top2:
            if usar_estadia_cd_ajust:
                st.info("**Modo activo: ESTIMADO TOTAL** — Se imputan las estadías estándar por CD en viajes sin dato CD o con estadía > 3 hs.")
            else:
                st.success("**Modo activo: SOLO DATOS REALES** — Únicamente viajes y jornadas con estadía real medida en CD.")

        with st.expander("⏱️ Tiempos Estándar de Estadía en CD por Operación y Parámetros de Limpieza", expanded=False):
            with st.form("form_params_s5", border=False):
                st.markdown("**1. Estadía Estándar en CD (en minutos) — Se aplica si no hay dato en CD o si supera las 3 hs (180 min):**")
                cd_col1, cd_col2, cd_col3, cd_col4, cd_col5, cd_col6 = st.columns(6)
                with cd_col1:
                    est_std_501 = st.number_input("CD 501 (1:15 = 75m):", min_value=15, max_value=240, value=75, step=5, key="s5_std_501")
                with cd_col2:
                    est_std_503 = st.number_input("CD 503 (1:22 = 82m):", min_value=15, max_value=240, value=82, step=1, key="s5_std_503")
                with cd_col3:
                    est_std_504 = st.number_input("CD 504 (1:41 = 101m):", min_value=15, max_value=240, value=101, step=1, key="s5_std_504")
                with cd_col4:
                    est_std_508 = st.number_input("CD 508 (1:50 = 110m):", min_value=15, max_value=240, value=110, step=5, key="s5_std_508")
                with cd_col5:
                    est_std_509 = st.number_input("CD 509 (1:05 = 65m):", min_value=15, max_value=240, value=65, step=5, key="s5_std_509")
                with cd_col6:
                    est_std_514 = st.number_input("CD 514 (1:51 = 111m):", min_value=15, max_value=240, value=111, step=1, key="s5_std_514")

                st.markdown("**2. Parámetros Físicos de Limpieza de Viajes y Jornadas:**")
                jc1, jc2, jc3, jc4 = st.columns(4)
                with jc1:
                    min_cd_j = st.number_input("Estadía mín. CD Real (min):", min_value=0, max_value=60, value=10, step=5, key="s5_min_cd")
                    max_cd_real_j = st.number_input(
                        "Tope máx. CD Real (min):",
                        min_value=60,
                        max_value=720,
                        value=180,
                        step=30,
                        help="Por defecto 180 min (3 hs). Por encima de este valor se considera guarda en predio.",
                        key="s5_max_cd",
                    )
                with jc2:
                    min_dur_v = st.number_input("Duración mín. Viaje (hs):", min_value=0.2, max_value=3.0, value=0.5, step=0.1, key="s5_min_dv")
                    max_dur_v = st.number_input("Duración máx. Viaje (hs):", min_value=8.0, max_value=24.0, value=16.0, step=1.0, key="s5_max_dv")
                with jc3:
                    min_dur_j = st.number_input("Duración mín. Jornada (hs):", min_value=0.5, max_value=4.0, value=1.0, step=0.5, key="s5_min_dj")
                    max_dur_j = st.number_input("Duración máx. Jornada (hs):", min_value=10.0, max_value=24.0, value=18.0, step=1.0, key="s5_max_dj")
                with jc4:
                    st.write("")
                    tukey_j = st.checkbox("Bigotes estadísticos Tukey (Q1/Q3 ± 1.5·IQR)", value=True, key="s5_tukey")
                st.form_submit_button("✅ Aplicar tiempos estándar y parámetros")

        cd_std_items = (
            (501, float(est_std_501)),
            (503, float(est_std_503)),
            (504, float(est_std_504)),
            (508, float(est_std_508)),
            (509, float(est_std_509)),
            (514, float(est_std_514)),
        )
        cd_std_map = dict(cd_std_items)
        med_std_fallback = float(np.mean(list(cd_std_map.values())))

        # Evaluacion cacheada ultra-rapida sobre la base pre-calculada
        trips_eval_s5, jornadas_real_all, jornadas_ajust_all = _eval_viajes_jornadas_s5_cached(
            trips_pre_filt,
            cd_std_items,
            int(min_cd_j),
            int(max_cd_real_j),
            float(min_dur_v),
            float(max_dur_v),
            float(min_dur_j),
            float(max_dur_j),
        )

        # Seleccionar columnas activas segun el interruptor del usuario
        trips_eval_s5 = trips_eval_s5.copy()
        if usar_estadia_cd_ajust:
            trips_eval_s5["Viaje_OK_Activo"] = trips_eval_s5["Viaje_OK_Ajust"]
            trips_eval_s5["dt_ini_cd_Activo"] = trips_eval_s5["dt_ini_cd_Ajust"]
            trips_eval_s5["Estadia_CD_Activa_min"] = trips_eval_s5["Estadia_CD_Ajust_min"]
            trips_eval_s5["Duracion_Viaje_hs"] = trips_eval_s5["Duracion_Viaje_Ajust_hs"]
            jornadas_all = jornadas_ajust_all
        else:
            trips_eval_s5["Viaje_OK_Activo"] = trips_eval_s5["Viaje_OK_Real"]
            trips_eval_s5["dt_ini_cd_Activo"] = trips_eval_s5["dt_ini_cd"]
            trips_eval_s5["Estadia_CD_Activa_min"] = trips_eval_s5["Estadia_CD_Origen"]
            trips_eval_s5["Duracion_Viaje_hs"] = trips_eval_s5["Duracion_Viaje_Real_hs"]
            jornadas_all = jornadas_real_all

        trips_eval_s5["Tiempo_CD_hs"] = (trips_eval_s5["Estadia_CD_Activa_min"].fillna(0) / 60.0).round(2)
        trips_eval_s5["Tiempo_Tiendas_hs"] = (trips_eval_s5["Suma_Estadia_Tiendas_min"].fillna(0) / 60.0).round(2)
        trips_eval_s5["Tiempo_Ruta_hs"] = (
            (trips_eval_s5["Duracion_Viaje_hs"] - trips_eval_s5["Tiempo_CD_hs"] - trips_eval_s5["Tiempo_Tiendas_hs"])
            .clip(lower=0.0)
            .round(2)
        )

        viajes_ok_s5 = trips_eval_s5[trips_eval_s5["Viaje_OK_Activo"]].copy()

        if not jornadas_all.empty:
            jornadas_clean_s5 = jornadas_all[jornadas_all["Jornada_Limpia"]].copy()
            jornadas_excl_s5 = jornadas_all[~jornadas_all["Jornada_Limpia"]].copy()
            jornadas_excl_s5["Motivo_Exclusion_Jornada"] = np.where(
                ~jornadas_excl_s5["Todos_Viajes_OK"],
                "Al menos 1 viaje del día excluido por falta de datos/anomalía",
                "Duración de jornada fuera de rango físico",
            )
            jornadas_clean_s5["Vueltas_Label"] = np.select(
                [jornadas_clean_s5["Viajes_Plan_Dia"] == 1, jornadas_clean_s5["Viajes_Plan_Dia"] == 2],
                ["1 Vuelta", "2 Vueltas"],
                default="3+ Vueltas",
            )
        else:
            jornadas_clean_s5 = pd.DataFrame()
            jornadas_excl_s5 = pd.DataFrame()

        # -------------------------------------------------------------------------
        # TARJETAS KPI SUPERIORES: REALES vs. CON ESTADIA CD AJUSTADA + TOTALES
        # -------------------------------------------------------------------------
        tot_v_s5 = int(len(trips_eval_s5))
        ok_v_real = int(trips_eval_s5["Viaje_OK_Real"].sum())
        ok_v_ajust = int(trips_eval_s5["Viaje_OK_Ajust"].sum())
        pct_v_real = (ok_v_real / tot_v_s5 * 100) if tot_v_s5 > 0 else 0.0
        pct_v_ajust = (ok_v_ajust / tot_v_s5 * 100) if tot_v_s5 > 0 else 0.0
        gain_v = ok_v_ajust - ok_v_real

        tot_j_s5 = int(len(jornadas_real_all))
        ok_j_real = int(jornadas_real_all["Jornada_Limpia"].sum()) if not jornadas_real_all.empty else 0
        ok_j_ajust = int(jornadas_ajust_all["Jornada_Limpia"].sum()) if not jornadas_ajust_all.empty else 0
        pct_j_real = (ok_j_real / tot_j_s5 * 100) if tot_j_s5 > 0 else 0.0
        pct_j_ajust = (ok_j_ajust / tot_j_s5 * 100) if tot_j_s5 > 0 else 0.0
        gain_j = ok_j_ajust - ok_j_real

        ok_v_act = len(viajes_ok_s5)
        med_dur_v_s5 = float(viajes_ok_s5["Duracion_Viaje_hs"].median()) if ok_v_act > 0 else 0.0

        ok_j_act = len(jornadas_clean_s5)
        j_1v = jornadas_clean_s5[jornadas_clean_s5["Vueltas_Label"] == "1 Vuelta"] if ok_j_act > 0 else pd.DataFrame()
        j_2v = jornadas_clean_s5[jornadas_clean_s5["Vueltas_Label"] == "2 Vueltas"] if ok_j_act > 0 else pd.DataFrame()
        j_2mas_cnt = int((jornadas_clean_s5["Viajes_Plan_Dia"] >= 2).sum()) if ok_j_act > 0 else 0
        pct_2mas = (j_2mas_cnt / ok_j_act * 100) if ok_j_act > 0 else 0.0

        med_j_tot = float(jornadas_clean_s5["Duracion_Jornada_hs"].median()) if ok_j_act > 0 else 0.0
        med_j_1v = float(j_1v["Duracion_Jornada_hs"].median()) if len(j_1v) > 0 else 0.0
        med_j_2v = float(j_2v["Duracion_Jornada_hs"].median()) if len(j_2v) > 0 else 0.0
        med_entre_v_min = float(j_2v["Entre_Vueltas_hs"].median() * 60.0) if len(j_2v) > 0 else 0.0

        kj1, kj2, kj3, kj4 = st.columns(4)
        with kj1:
            st.markdown(
                _kpi_card_html(
                    "Viajes OK (Real vs. Ajustado)",
                    f"{_fmt_int(ok_v_real)} ({_fmt_pct(pct_v_real)})",
                    f"Con CD Ajustable: {_fmt_int(ok_v_ajust)} ({_fmt_pct(pct_v_ajust)}) [+{_fmt_int(gain_v)}]",
                    "#16a34a",
                    f"Total viajes TMS evaluados: <b>{_fmt_int(tot_v_s5)}</b>",
                ),
                unsafe_allow_html=True,
            )
        with kj2:
            st.markdown(
                _kpi_card_html(
                    "Jornadas OK (Real vs. Ajustado)",
                    f"{_fmt_int(ok_j_real)} ({_fmt_pct(pct_j_real)})",
                    f"Con CD Ajustable: {_fmt_int(ok_j_ajust)} ({_fmt_pct(pct_j_ajust)}) [+{_fmt_int(gain_j)}]",
                    "#1d4ed8",
                    f"Total jornadas camión-día evaluadas: <b>{_fmt_int(tot_j_s5)}</b>",
                ),
                unsafe_allow_html=True,
            )
        with kj3:
            modo_lbl_kpi = "Estimado" if usar_estadia_cd_ajust else "Real"
            st.markdown(
                _kpi_card_html(
                    f"Jornadas de 2+ Vueltas ({modo_lbl_kpi})",
                    _fmt_int(j_2mas_cnt),
                    f"{_fmt_pct(pct_2mas)} de {_fmt_int(ok_j_act)} jornadas limpias",
                    "#d97706",
                    f"Re-carga entre vueltas: <b>{med_entre_v_min:.0f} min</b>",
                ),
                unsafe_allow_html=True,
            )
        with kj4:
            st.markdown(
                _kpi_card_html(
                    f"Mediana Horas Jornada ({modo_lbl_kpi})",
                    f"{med_j_tot:.1f} hs",
                    f"1 Vuelta: {med_j_1v:.1f} hs  |  2 Vueltas: {med_j_2v:.1f} hs",
                    "#0f172a",
                    f"Mediana duración viaje: <b>{med_dur_v_s5:.1f} hs</b>",
                ),
                unsafe_allow_html=True,
            )

        # Tabla resumen de representatividad y diagnóstico de exclusiones por CD
        with st.expander("📊 Ver Cobertura y Diagnóstico por CD: ¿Por qué se excluyen viajes? (CD vs. Estadías en Tienda)", expanded=False):
            st.caption(
                "Permite ver al instante si las exclusiones de un CD (como el **CD 504**) provienen de **falta de dato/guarda en CD** "
                "(recuperables con el interruptor) o de **estadías/GPS faltantes en las tiendas**."
            )
            cov_rows_s5 = []
            for op_s5 in sorted(trips_eval_s5["Operacion"].dropna().unique()):
                sub_t5 = trips_eval_s5[trips_eval_s5["Operacion"] == op_s5]
                tv_op = len(sub_t5)
                cd_ok_op = int(sub_t5["CD_Real_OK_Flag"].sum())
                st_ok_op = int(sub_t5["Tiendas_y_KMS_OK_Flag"].sum())
                vr_op = int(sub_t5["Viaje_OK_Real"].sum())
                va_op = int(sub_t5["Viaje_OK_Ajust"].sum())

                sub_jr = jornadas_real_all[jornadas_real_all["Operacion"] == op_s5] if not jornadas_real_all.empty else pd.DataFrame()
                sub_ja = jornadas_ajust_all[jornadas_ajust_all["Operacion"] == op_s5] if not jornadas_ajust_all.empty else pd.DataFrame()
                tj_op = len(sub_jr)
                jr_op = int(sub_jr["Jornada_Limpia"].sum()) if not sub_jr.empty else 0
                ja_op = int(sub_ja["Jornada_Limpia"].sum()) if not sub_ja.empty else 0

                cov_rows_s5.append(
                    {
                        "CD": str(int(op_s5)),
                        "Std. CD": f"{int(cd_std_map.get(int(op_s5), med_std_fallback))}m",
                        "Viajes Totales": tv_op,
                        "CD Real OK": f"{cd_ok_op:,} ({(cd_ok_op/tv_op*100 if tv_op else 0):.0f}%)".replace(",", "."),
                        "Tiendas/Estadías OK": f"{st_ok_op:,} ({(st_ok_op/tv_op*100 if tv_op else 0):.0f}%)".replace(",", "."),
                        "Viajes OK (Real)": f"{vr_op:,} ({(vr_op/tv_op*100 if tv_op else 0):.1f}%)".replace(",", "."),
                        "Viajes OK (Ajustado)": f"{va_op:,} ({(va_op/tv_op*100 if tv_op else 0):.1f}%)".replace(",", "."),
                        "Jornadas Totales": tj_op,
                        "Jornadas OK (Real)": f"{jr_op:,} ({(jr_op/tj_op*100 if tj_op else 0):.1f}%)".replace(",", "."),
                        "Jornadas OK (Ajustado)": f"{ja_op:,} ({(ja_op/tj_op*100 if tj_op else 0):.1f}%)".replace(",", "."),
                    }
                )
            if cov_rows_s5:
                st.dataframe(pd.DataFrame(cov_rows_s5), use_container_width=True, hide_index=True)

        st.divider()

        # -------------------------------------------------------------------------
        # BLOQUE 1: DURACION DE VIAJES OK (PREFIJADO: TIPO DE VIAJE x CATEGORIA)
        # -------------------------------------------------------------------------
        @_ui_fragment
        def _render_s5_viajes_fragment():
            st.markdown(
                f"### 1. Duración y Evolución Temporal de Viajes OK ({'Base Estimada con CD Ajustable' if usar_estadia_cd_ajust else 'Base Solo Datos Reales'})"
            )

            v_dim_map = {
                "Tipo de Viaje (Directo / Compartido / Rutas)": "Tipo_Viaje",
                "Evolución por Mes": "Evol_Mes",
                "Evolución por Semana": "Evol_Semana",
                "Evolución por Fecha (Día)": "Evol_Dia",
                "Tipo de Vehículo (Nómina)": "Tipo_Vehiculo",
                "Operación (CD)": "Operacion",
                "Categoría (Cercana vs. Lejana)": "Categoria",
            }
            v_sub_map = {
                "Sin sub-corte": None,
                "Categoría (Cercana vs. Lejana)": "Categoria",
                "Tipo de Viaje": "Tipo_Viaje",
                "Tipo de Vehículo (Nómina)": "Tipo_Vehiculo",
                "Operación (CD)": "Operacion",
            }

            vc_1, vc_2 = st.columns([1.3, 1.2])
            with vc_1:
                v_dim_lbl = st.selectbox("Agrupar viajes por (segmento o período):", list(v_dim_map.keys()), index=0, key="s5_v_dim")
            with vc_2:
                v_sub_lbl = st.selectbox("Sub-dividir cajas por color:", list(v_sub_map.keys()), index=1, key="s5_v_sub")

            v_dim_col = v_dim_map[v_dim_lbl]
            v_sub_col = v_sub_map[v_sub_lbl]
            if v_sub_col == v_dim_col:
                v_sub_col = None

            df_v_res = pd.DataFrame()
            if not viajes_ok_s5.empty:
                df_vp_s5 = viajes_ok_s5.copy()
                df_vt_s5 = trips_eval_s5.copy()

                if v_dim_col == "Evol_Dia":
                    sems_v5_disp = (
                        df_vp_s5[["Semana_Sort", "Semana_Label"]]
                        .drop_duplicates()
                        .sort_values("Semana_Sort")["Semana_Label"]
                        .tolist()
                    )
                    def_sems_v5 = sems_v5_disp[-2:] if len(sems_v5_disp) >= 2 else sems_v5_disp
                    sems_v5_sel = st.multiselect(
                        "Drill-down — Seleccioná semana(s) para abrir viajes por día (por defecto últimas 2):",
                        options=sems_v5_disp,
                        default=def_sems_v5,
                        placeholder="Todas las semanas",
                        key="s5_v_sems_dia",
                    )
                    if sems_v5_sel:
                        df_vp_s5 = df_vp_s5[df_vp_s5["Semana_Label"].isin(sems_v5_sel)].copy()
                        df_vt_s5 = df_vt_s5[df_vt_s5["Semana_Label"].isin(sems_v5_sel)].copy()

                def _prep_v_dim(d_in: pd.DataFrame, col_m: str) -> pd.DataFrame:
                    d = d_in.copy()
                    if col_m == "Evol_Mes":
                        d["_p_sort"] = d["Fecha_Reparto"].dt.to_period("M").dt.start_time
                        d[col_m] = d["_p_sort"].dt.strftime("%Y-%m")
                    elif col_m == "Evol_Semana":
                        d[col_m] = d["Semana_Label"]
                    elif col_m == "Evol_Dia":
                        d["_p_sort"] = d["Fecha_Reparto"].dt.floor("D")
                        d[col_m] = d["_p_sort"].dt.dayofweek.map(dias_es) + " " + d["_p_sort"].dt.strftime("%d/%m")
                    elif col_m == "Operacion":
                        d[col_m] = "CD " + d[col_m].astype(str)
                    else:
                        d[col_m] = d[col_m].fillna("Sin dato").astype(str)
                    return d

                df_vp_s5 = _prep_v_dim(df_vp_s5, v_dim_col)
                df_vt_s5 = _prep_v_dim(df_vt_s5, v_dim_col)
                if v_sub_col is not None:
                    df_vp_s5 = _prep_v_dim(df_vp_s5, v_sub_col)
                    df_vt_s5 = _prep_v_dim(df_vt_s5, v_sub_col)

                fixed_ord_v5 = ("Directo (1 tienda)", "Compartido (2 tiendas)", "Rutas (3+ tiendas)") if v_dim_col == "Tipo_Viaje" else None
                nice_v_dim = {
                    "Evol_Mes": "Mes",
                    "Evol_Semana": "Semana",
                    "Evol_Dia": "Fecha",
                    "Tipo_Viaje": "Tipo de Viaje",
                    "Tipo_Vehiculo": "Tipo de Vehículo",
                    "Operacion": "Operación",
                    "Categoria": "Categoría",
                }.get(v_dim_col, v_dim_col)

                fig_vbox_s5, df_v_res = _build_grouped_boxplot_cached(
                    df_vp_s5,
                    df_vt_s5,
                    val_col="Duracion_Viaje_hs",
                    dim_col=v_dim_col,
                    sub_col=v_sub_col,
                    unit_suffix="hs",
                    y_title="Duración del Viaje (horas)",
                    use_tukey=bool(tukey_j),
                    y_range=(0, float(max_dur_v) + 1.0),
                    y_dtick=2,
                    fixed_dim_order=fixed_ord_v5,
                    compute_summary_table=True,
                    nice_dim_name=nice_v_dim,
                    summary_mode="viajes",
                )
                st.plotly_chart(fig_vbox_s5, use_container_width=True)

                st.dataframe(
                    df_v_res,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "% Represent.": st.column_config.NumberColumn("% Represent.", format="%.1f%%"),
                    },
                )
            st.session_state["_s5_v_res"] = df_v_res

        _render_s5_viajes_fragment()
        df_v_res = st.session_state.get("_s5_v_res", pd.DataFrame())

        st.divider()

        # -------------------------------------------------------------------------
        # BLOQUE 2: ANALISIS DE JORNADAS DIARIAS (PREFIJADO: OPERACION CD x CANTIDAD DE VUELTAS)
        # -------------------------------------------------------------------------
        @_ui_fragment
        def _render_s5_jornadas_fragment():
            st.markdown(
                f"### 2. Análisis de Jornadas Diarias ({'Base Estimada con CD Ajustable' if usar_estadia_cd_ajust else 'Base Solo Datos Reales'})"
            )
            st.caption(
                "Cada jornada agrupa todos los viajes de un camión en un mismo día (**1 Vuelta** o **2+ Vueltas**), "
                "incluyendo únicamente días donde **el 100% de sus viajes** quedaron limpios en el modo activo."
            )

            df_j_res = pd.DataFrame()
            if not jornadas_clean_s5.empty:
                cat_j_sel = st.radio(
                    "Filtrar Categoría de Jornada (Cercana vs. Lejana):",
                    options=["Todas", "Cercana", "Lejana"],
                    index=0,
                    horizontal=True,
                    key="s5_j_cat_radio",
                )

                df_jp_base = jornadas_clean_s5.copy()
                df_jt_base = jornadas_all.copy()
                if cat_j_sel != "Todas":
                    df_jp_base = df_jp_base[df_jp_base["Categoria_Jornada"] == cat_j_sel].copy()
                    df_jt_base = df_jt_base[df_jt_base["Categoria_Jornada"] == cat_j_sel].copy()

                jb_c1, jb_c2 = st.columns([1.3, 1.2])
                with jb_c1:
                    j_box_dim_map = {
                        "Operación (CD)": "Operacion",
                        "Cantidad de Vueltas (1 vs. 2+ Vueltas)": "Vueltas_Label",
                        "Categoría (Cercana vs. Lejana)": "Categoria_Jornada",
                        "Tipo de Vehículo (Nómina)": "Tipo_Vehiculo",
                        "Evolución por Mes": "Evol_Mes",
                        "Evolución por Semana": "Evol_Semana",
                        "Evolución por Fecha (Día)": "Evol_Dia",
                    }
                    j_box_dim_lbl = st.selectbox(
                        "Agrupar diagrama de cajas de jornadas por:",
                        options=list(j_box_dim_map.keys()),
                        index=0,
                        key="s5_j_box_dim",
                    )
                with jb_c2:
                    j_box_sub_map = {
                        "Sin sub-corte": None,
                        "Cantidad de Vueltas (1 vs. 2+ Vueltas)": "Vueltas_Label",
                        "Categoría (Cercana vs. Lejana)": "Categoria_Jornada",
                        "Operación (CD)": "Operacion",
                    }
                    j_box_sub_lbl = st.selectbox(
                        "Sub-dividir cajas de jornadas por color:",
                        options=list(j_box_sub_map.keys()),
                        index=1,
                        key="s5_j_box_sub",
                    )

                j_dim_col = j_box_dim_map[j_box_dim_lbl]
                j_sub_col = j_box_sub_map[j_box_sub_lbl]
                if j_sub_col == j_dim_col:
                    j_sub_col = None

                df_jplot = df_jp_base.copy()
                if j_dim_col == "Evol_Dia" and not df_jplot.empty:
                    sems_j5_disp = (
                        df_jplot[["Semana_Sort", "Semana_Label"]]
                        .drop_duplicates()
                        .sort_values("Semana_Sort")["Semana_Label"]
                        .tolist()
                    )
                    def_sems_j5 = sems_j5_disp[-2:] if len(sems_j5_disp) >= 2 else sems_j5_disp
                    sems_j5_sel = st.multiselect(
                        "Drill-down — Seleccioná semana(s) para abrir jornadas por día (por defecto últimas 2):",
                        options=sems_j5_disp,
                        default=def_sems_j5,
                        placeholder="Todas las semanas",
                        key="s5_j_sems_dia",
                    )
                    if sems_j5_sel:
                        df_jplot = df_jplot[df_jplot["Semana_Label"].isin(sems_j5_sel)].copy()

                def _prep_j_col(d_in: pd.DataFrame, col_m: str) -> pd.DataFrame:
                    d = d_in.copy()
                    if col_m == "Evol_Mes":
                        d["_p_sort"] = d["Fecha_Reparto"].dt.to_period("M").dt.start_time
                        d[col_m] = d["_p_sort"].dt.strftime("%Y-%m")
                    elif col_m == "Evol_Semana":
                        d[col_m] = d["Semana_Label"]
                    elif col_m == "Evol_Dia":
                        d["_p_sort"] = d["Fecha_Reparto"].dt.floor("D")
                        d[col_m] = d["_p_sort"].dt.dayofweek.map(dias_es) + " " + d["_p_sort"].dt.strftime("%d/%m")
                    elif col_m == "Operacion":
                        d[col_m] = "CD " + d[col_m].astype(str)
                    else:
                        d[col_m] = d[col_m].fillna("Sin dato").astype(str)
                    return d

                if not df_jplot.empty:
                    df_jplot = _prep_j_col(df_jplot, j_dim_col)
                    if j_sub_col is not None:
                        df_jplot = _prep_j_col(df_jplot, j_sub_col)

                    fixed_ord_j5 = ("1 Vuelta", "2 Vueltas", "3+ Vueltas") if j_dim_col == "Vueltas_Label" else None
                    fixed_sub_j5 = ("1 Vuelta", "2 Vueltas", "3+ Vueltas") if j_sub_col == "Vueltas_Label" else None

                    fig_jbox, _ = _build_grouped_boxplot_cached(
                        df_jplot,
                        None,
                        val_col="Duracion_Jornada_hs",
                        dim_col=j_dim_col,
                        sub_col=j_sub_col,
                        unit_suffix="hs",
                        y_title="Duración de Jornada (horas)",
                        use_tukey=bool(tukey_j),
                        y_range=(0, float(max_dur_j) + 1.0),
                        y_dtick=2,
                        fixed_dim_order=fixed_ord_j5,
                        fixed_sub_order=fixed_sub_j5,
                        compute_summary_table=False,
                    )
                    st.plotly_chart(fig_jbox, use_container_width=True)

                j_cut = st.selectbox(
                    "Abrir tabla resumen de jornadas por:",
                    options=["Operación (CD)", "Tipo de Vehículo (Nómina)", "Categoría (Cercana vs. Lejana)"],
                    index=0,
                    key="s5_j_cut",
                )
                j_cut_col = (
                    "Operacion"
                    if j_cut == "Operación (CD)"
                    else ("Tipo_Vehiculo" if j_cut == "Tipo de Vehículo (Nómina)" else "Categoria_Jornada")
                )

                df_jp = df_jp_base.copy()
                df_jt = df_jt_base.copy()
                if j_cut_col == "Operacion":
                    df_jp[j_cut_col] = "CD " + df_jp[j_cut_col].astype(str)
                    df_jt[j_cut_col] = "CD " + df_jt[j_cut_col].astype(str)
                else:
                    df_jp[j_cut_col] = df_jp[j_cut_col].fillna("Sin dato").astype(str)
                    df_jt[j_cut_col] = df_jt[j_cut_col].fillna("Sin dato").astype(str)

                tot_j_map = df_jt.groupby(j_cut_col).size().to_dict()

                j_rows = []
                for k_val, grp in df_jp.groupby(j_cut_col):
                    j_tot_k = int(tot_j_map.get(k_val, len(grp)))
                    j_ok_k = int(len(grp))
                    pct_rep_j = round((j_ok_k / j_tot_k * 100), 1) if j_tot_k > 0 else 100.0
                    g1 = grp[grp["Vueltas_Label"] == "1 Vuelta"]
                    g2 = grp[grp["Vueltas_Label"] == "2 Vueltas"]
                    g3 = grp[grp["Vueltas_Label"] == "3+ Vueltas"]
                    j_rows.append(
                        {
                            j_cut: k_val,
                            "Jornadas Totales": j_tot_k,
                            "Jornadas Limpias": j_ok_k,
                            "% Represent.": pct_rep_j,
                            "1 Vuelta": int(len(g1)),
                            "2 Vueltas": int(len(g2)),
                            "3+ Vueltas": int(len(g3)),
                            "Med. 1 Vuelta (hs)": round(float(g1["Duracion_Jornada_hs"].median()), 1) if len(g1) > 0 else None,
                            "Med. 2 Vueltas (hs)": round(float(g2["Duracion_Jornada_hs"].median()), 1) if len(g2) > 0 else None,
                            "P90 Jornada (hs)": round(float(np.percentile(grp["Duracion_Jornada_hs"], 90)), 1),
                            "Mediana KMS/Día": round(float(grp["KMS_Dia"].median()), 1),
                        }
                    )
                df_j_res = pd.DataFrame(j_rows).sort_values("Jornadas Limpias", ascending=False)
                st.dataframe(
                    df_j_res,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "% Represent.": st.column_config.NumberColumn("% Represent.", format="%.1f%%"),
                    },
                )
            st.session_state["_s5_j_res"] = df_j_res

        _render_s5_jornadas_fragment()
        df_j_res = st.session_state.get("_s5_j_res", pd.DataFrame())

        st.divider()

        @_ui_fragment
        def _render_s5_interno_fragment():
            # -------------------------------------------------------------------------
            # BLOQUE 3: RANKING DE INTERNOS EN JORNADAS Y EVOLUCION PUNTUAL POR CAMION
            # -------------------------------------------------------------------------
            st.markdown(
                f"### 3. Ranking de Internos en Jornadas y Evolución Puntual ({'Base Estimada con CD Ajustable' if usar_estadia_cd_ajust else 'Base Solo Datos Reales'})"
            )
            st.caption(
                "Identifica los **Internos con jornadas más extensas vs. más cortas** (comparados contra la mediana de su **Tipo de Camión**), "
                "y permite inspeccionar la **evolución día por día** de cualquier unidad puntual."
            )

            if jornadas_clean_s5.empty:
                st.info("No hay jornadas limpias para construir el ranking de internos en el modo activo.")
                return

            cds_j_int_opts = ["Todos"] + sorted(
                jornadas_clean_s5["Operacion"].dropna().astype(int).astype(str).unique().tolist()
            )
            tv_j_int_opts = ["Todos"] + sorted(
                jornadas_clean_s5["Tipo_Vehiculo"].fillna("Sin dato").astype(str).unique().tolist()
            )

            rkj1, rkj2, rkj3, rkj4 = st.columns([0.9, 0.9, 1.0, 1.3])
            with rkj1:
                cd_rk_j = st.selectbox("Filtrar por CD:", options=cds_j_int_opts, index=0, key="s5_rkj_cd")
            with rkj2:
                top_n_j = st.selectbox("Cantidad Top:", options=[10, 15, 20, 30], index=1, key="s5_rkj_topn")
            with rkj3:
                min_j_int = st.number_input("Mín. jornadas limpias:", min_value=1, max_value=100, value=5, step=1, key="s5_rkj_min")
            with rkj4:
                tv_rk_j = st.selectbox("Filtrar por Tipo de Camión:", options=tv_j_int_opts, index=0, key="s5_rkj_tv")

            # Referencia global por Tipo de Camión en el modo activo
            veh_j_ref = (
                jornadas_clean_s5.assign(_tv=jornadas_clean_s5["Tipo_Vehiculo"].fillna("Sin dato").astype(str))
                .groupby("_tv", as_index=False)
                .agg(Mediana_Veh_hs=("Duracion_Jornada_hs", "median"))
            )

            j_tot_u = jornadas_all.copy()
            j_ok_u = jornadas_clean_s5.copy()
            j_ok_u["_tv"] = j_ok_u["Tipo_Vehiculo"].fillna("Sin dato").astype(str)
            if cd_rk_j != "Todos":
                j_tot_u = j_tot_u[j_tot_u["Operacion"].astype(int).astype(str) == str(cd_rk_j)]
                j_ok_u = j_ok_u[j_ok_u["Operacion"].astype(int).astype(str) == str(cd_rk_j)]
            if tv_rk_j != "Todos":
                j_tot_u = j_tot_u[j_tot_u["Tipo_Vehiculo"].fillna("Sin dato").astype(str) == tv_rk_j]
                j_ok_u = j_ok_u[j_ok_u["_tv"] == tv_rk_j]

            tot_j_cnt = (
                j_tot_u.groupby(["Operacion", "Interno_Clean"], as_index=False)
                .size()
                .rename(columns={"size": "Jornadas_Totales"})
            )

            agg_ji = (
                j_ok_u.groupby(["Operacion", "Interno_Clean"], as_index=False)
                .agg(
                    Dominio=("Dominio", lambda s: " / ".join(sorted(s.dropna().astype(str).unique()))),
                    Tipo_Camion=("_tv", lambda s: s.value_counts().index[0]),
                    Jornadas_Limpias=("Duracion_Jornada_hs", "count"),
                    J_1V=("Vueltas_Label", lambda s: int((s == "1 Vuelta").sum())),
                    J_2V=("Vueltas_Label", lambda s: int((s == "2 Vueltas").sum())),
                    Med_Jornada_hs=("Duracion_Jornada_hs", "median"),
                    P90_Jornada_hs=("Duracion_Jornada_hs", lambda s: float(np.percentile(s.dropna(), 90))),
                    Med_KMS_Dia=("KMS_Dia", "median"),
                )
                .merge(tot_j_cnt, on=["Operacion", "Interno_Clean"], how="left")
                .merge(veh_j_ref, left_on="Tipo_Camion", right_on="_tv", how="left")
            )
            agg_ji = agg_ji[agg_ji["Jornadas_Limpias"] >= int(min_j_int)].copy()

            if not agg_ji.empty:
                agg_ji["CD"] = agg_ji["Operacion"].astype(int).astype(str)
                agg_ji["Diff_vs_Camion_hs"] = (agg_ji["Med_Jornada_hs"] - agg_ji["Mediana_Veh_hs"]).round(1)
                agg_ji["Int_Label"] = (
                    "Int. "
                    + agg_ji["Interno_Clean"]
                    + " ("
                    + agg_ji["Dominio"]
                    + " | CD "
                    + agg_ji["CD"]
                    + " · "
                    + agg_ji["Tipo_Camion"]
                    + ")"
                )
                agg_ji = agg_ji.sort_values("Med_Jornada_hs", ascending=False).reset_index(drop=True)

                c_jhi, c_jlo = st.columns(2)
                with c_jhi:
                    top_j_hi = agg_ji.head(int(top_n_j)).iloc[::-1].copy()
                    col_jhi = [
                        "#dc2626" if v >= 10.0 else ("#f59e0b" if v >= 8.5 else "#1d4ed8")
                        for v in top_j_hi["Med_Jornada_hs"]
                    ]
                    fig_jhi = go.Figure(
                        data=go.Bar(
                            x=top_j_hi["Med_Jornada_hs"],
                            y=top_j_hi["Int_Label"],
                            orientation="h",
                            marker_color=col_jhi,
                            text=[f"{v:.1f} hs" for v in top_j_hi["Med_Jornada_hs"]],
                            textposition="outside",
                            customdata=np.dstack(
                                (
                                    top_j_hi["Med_Jornada_hs"],
                                    top_j_hi["Mediana_Veh_hs"],
                                    top_j_hi["Diff_vs_Camion_hs"],
                                    top_j_hi["Jornadas_Limpias"],
                                    top_j_hi["Jornadas_Totales"],
                                    top_j_hi["J_1V"],
                                    top_j_hi["J_2V"],
                                    top_j_hi["Med_KMS_Dia"],
                                )
                            )[0],
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Mediana Jornada: <b>%{customdata[0]:.1f} hs</b> (Ref. Tipo Camión: %{customdata[1]:.1f} hs | %{customdata[2]:+.1f} hs)<br>"
                                "Jornadas: <b>%{customdata[3]:.0f} limpias</b> de %{customdata[4]:.0f} totales (1V: %{customdata[5]:.0f} | 2V: %{customdata[6]:.0f})<br>"
                                "Mediana KMS/Día: <b>%{customdata[7]:.0f} km</b><extra></extra>"
                            ),
                        )
                    )
                    max_jh = float(top_j_hi["Med_Jornada_hs"].max()) if not top_j_hi.empty else 12.0
                    fig_jhi.update_layout(
                        title=f"🔴 Top {len(top_j_hi)} Mayor Duración de Jornada (hs)",
                        height=max(340, len(top_j_hi) * 26 + 90),
                        margin=dict(l=10, r=55, t=40, b=20),
                        xaxis_title="Mediana Duración Jornada (hs)",
                        yaxis_title="",
                        xaxis=dict(range=[0, max_jh * 1.24]),
                    )
                    st.plotly_chart(fig_jhi, use_container_width=True)

                with c_jlo:
                    top_j_lo = agg_ji.tail(int(top_n_j)).copy().sort_values("Med_Jornada_hs", ascending=False)
                    col_jlo = [
                        "#16a34a" if v <= 0 else ("#3b82f6" if v <= 1.0 else "#f59e0b")
                        for v in top_j_lo["Diff_vs_Camion_hs"]
                    ]
                    fig_jlo = go.Figure(
                        data=go.Bar(
                            x=top_j_lo["Med_Jornada_hs"],
                            y=top_j_lo["Int_Label"],
                            orientation="h",
                            marker_color=col_jlo,
                            text=[f"{v:.1f} hs" for v in top_j_lo["Med_Jornada_hs"]],
                            textposition="outside",
                            customdata=np.dstack(
                                (
                                    top_j_lo["Med_Jornada_hs"],
                                    top_j_lo["Mediana_Veh_hs"],
                                    top_j_lo["Diff_vs_Camion_hs"],
                                    top_j_lo["Jornadas_Limpias"],
                                    top_j_lo["Jornadas_Totales"],
                                    top_j_lo["J_1V"],
                                    top_j_lo["J_2V"],
                                    top_j_lo["Med_KMS_Dia"],
                                )
                            )[0],
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Mediana Jornada: <b>%{customdata[0]:.1f} hs</b> (Ref. Tipo Camión: %{customdata[1]:.1f} hs | %{customdata[2]:+.1f} hs)<br>"
                                "Jornadas: <b>%{customdata[3]:.0f} limpias</b> de %{customdata[4]:.0f} totales (1V: %{customdata[5]:.0f} | 2V: %{customdata[6]:.0f})<br>"
                                "Mediana KMS/Día: <b>%{customdata[7]:.0f} km</b><extra></extra>"
                            ),
                        )
                    )
                    max_jl = float(top_j_lo["Med_Jornada_hs"].max()) if not top_j_lo.empty else 12.0
                    fig_jlo.update_layout(
                        title=f"🟢 Top {len(top_j_lo)} Jornadas Más Cortas (hs)",
                        height=max(340, len(top_j_lo) * 26 + 90),
                        margin=dict(l=10, r=55, t=40, b=20),
                        xaxis_title="Mediana Duración Jornada (hs)",
                        yaxis_title="",
                        xaxis=dict(range=[0, max_jl * 1.24]),
                    )
                    st.plotly_chart(fig_jlo, use_container_width=True)

            # -----------------------------------------------------------------
            # EVOLUCION PUNTUAL DE UN INTERNO (SERIE DIARIA CON PUNTOS + REFERENCIA)
            # -----------------------------------------------------------------
            with st.expander("📈 Ver Evolución Día por Día de un Interno Puntual (Serie Temporal)", expanded=True):
                st.caption(
                    "Como cada camión realiza **1 jornada por día**, una **línea de evolución diaria con puntos** "
                    "(diferenciando 1 Vuelta vs. 2+ Vueltas) permite ver su comportamiento mucho mejor que un diagrama de cajas."
                )
                f_s5_c1, f_s5_c2, f_s5_c3 = st.columns([1.2, 0.9, 1.3])
                internos_opts_s5 = sorted(
                    j_ok_u["Interno_Clean"].dropna().unique().tolist(),
                    key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
                ) if not j_ok_u.empty else sorted(
                    jornadas_clean_s5["Interno_Clean"].dropna().unique().tolist(),
                    key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
                )
                with f_s5_c1:
                    interno_sel_s5 = st.selectbox(
                        "Interno:",
                        options=internos_opts_s5,
                        index=0 if internos_opts_s5 else None,
                        placeholder="Seleccioná un interno",
                        key="s5_interno_sel",
                    )

                cds_int_s5 = (
                    sorted(
                        jornadas_clean_s5[jornadas_clean_s5["Interno_Clean"] == interno_sel_s5]["Operacion"]
                        .dropna()
                        .astype(int)
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    if interno_sel_s5
                    else []
                )
                with f_s5_c2:
                    cd_sel_s5 = st.selectbox(
                        "CD:",
                        options=["Todos"] + cds_int_s5,
                        index=0,
                        key="s5_cd_sel",
                    )
                with f_s5_c3:
                    metrica_int_s5 = st.radio(
                        "Métrica:",
                        options=["Jornadas Diarias (hs)", "Viajes OK (hs)"],
                        index=0,
                        horizontal=True,
                        key="s5_int_metrica",
                    )

                if interno_sel_s5:
                    sub_t_int = trips_eval_s5[trips_eval_s5["Interno_Clean"] == interno_sel_s5].copy()
                    sub_j_int = jornadas_clean_s5[jornadas_clean_s5["Interno_Clean"] == interno_sel_s5].copy()
                    if cd_sel_s5 != "Todos":
                        sub_t_int = sub_t_int[sub_t_int["Operacion"].astype(int).astype(str) == str(cd_sel_s5)].copy()
                        sub_j_int = sub_j_int[sub_j_int["Operacion"].astype(int).astype(str) == str(cd_sel_s5)].copy()

                    dom_info = " / ".join(sorted(sub_t_int["Dominio"].dropna().astype(str).unique()))
                    tv_info = " / ".join(sorted(sub_t_int["Tipo_Vehiculo"].dropna().astype(str).unique()))
                    cd_titulo_int = f"CD {cd_sel_s5}" if cd_sel_s5 != "Todos" else f"CD {', '.join(cds_int_s5)}"
                    v_ok_int = sub_t_int[sub_t_int["Viaje_OK_Activo"]].copy()

                    st.caption(
                        f"**{cd_titulo_int} · Interno {interno_sel_s5}** ({dom_info} · {tv_info}) — "
                        f"Jornadas limpias: **{len(sub_j_int)}** | Viajes OK: **{len(v_ok_int)} / {len(sub_t_int)}**"
                    )

                    if metrica_int_s5 == "Jornadas Diarias (hs)" and not sub_j_int.empty:
                        sub_j_int = sub_j_int.sort_values("Fecha_Reparto").copy()
                        sub_j_int["Fecha_Lbl"] = (
                            sub_j_int["Fecha_Reparto"].dt.dayofweek.map(dias_es)
                            + " "
                            + sub_j_int["Fecha_Reparto"].dt.strftime("%d/%m")
                        )
                        med_u = float(sub_j_int["Duracion_Jornada_hs"].median())
                        fig_line_i = go.Figure()
                        fig_line_i.add_trace(
                            go.Scatter(
                                x=sub_j_int["Fecha_Lbl"],
                                y=sub_j_int["Duracion_Jornada_hs"],
                                mode="lines",
                                line=dict(color="#94a3b8", width=1.8),
                                name="Tendencia",
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                        for vl_i, col_vl in [("1 Vuelta", "#1d4ed8"), ("2 Vueltas", "#d97706"), ("3+ Vueltas", "#dc2626")]:
                            sub_v_i = sub_j_int[sub_j_int["Vueltas_Label"] == vl_i]
                            if sub_v_i.empty:
                                continue
                            fig_line_i.add_trace(
                                go.Scatter(
                                    x=sub_v_i["Fecha_Lbl"],
                                    y=sub_v_i["Duracion_Jornada_hs"],
                                    mode="markers",
                                    name=vl_i,
                                    marker=dict(size=10, color=col_vl, line=dict(width=1.5, color="#ffffff")),
                                    customdata=np.dstack(
                                        (
                                            sub_v_i["Duracion_Jornada_hs"],
                                            sub_v_i["Vueltas_Label"],
                                            sub_v_i["Categoria_Jornada"],
                                            sub_v_i["KMS_Dia"],
                                            sub_v_i["Tiendas_Dia"],
                                        )
                                    )[0],
                                    hovertemplate=(
                                        "<b>%{x}</b><br>"
                                        "Duración Jornada: <b>%{customdata[0]:.1f} hs</b> (%{customdata[1]})<br>"
                                        "Categoría: %{customdata[2]} | KMS Día: %{customdata[3]:.0f} km | Tiendas: %{customdata[4]:.0f}"
                                        "<extra></extra>"
                                    ),
                                )
                            )
                        fig_line_i.add_hline(
                            y=med_u,
                            line_dash="dot",
                            line_color="#0f172a",
                            annotation_text=f"Mediana Interno: {med_u:.1f} hs",
                            annotation_position="top right",
                        )
                        fig_line_i.update_layout(
                            title=f"{cd_titulo_int} · Interno {interno_sel_s5} — Evolución Diaria de Jornadas (hs)",
                            height=370,
                            margin=dict(l=20, r=20, t=42, b=20),
                            yaxis=dict(title="Duración de Jornada (horas)", range=[0, float(max_dur_j) + 1.0], dtick=2),
                            xaxis=dict(tickangle=-35 if len(sub_j_int) > 12 else 0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            hovermode="closest",
                        )
                        st.plotly_chart(fig_line_i, use_container_width=True)
                    elif metrica_int_s5 == "Viajes OK (hs)" and not v_ok_int.empty:
                        v_ok_int = v_ok_int.sort_values(["Fecha_Reparto", "Hora_Reparto"]).copy()
                        v_ok_int["Fecha_Lbl"] = (
                            v_ok_int["Fecha_Reparto"].dt.dayofweek.map(dias_es)
                            + " "
                            + v_ok_int["Fecha_Reparto"].dt.strftime("%d/%m")
                            + " ("
                            + v_ok_int["Ref_Clean"].astype(str)
                            + ")"
                        )
                        med_uv = float(v_ok_int["Duracion_Viaje_hs"].median())
                        fig_line_v = go.Figure()
                        for cat_i, col_ci in [("Cercana", "#1d4ed8"), ("Lejana", "#d97706")]:
                            sv_c = v_ok_int[v_ok_int["Categoria"] == cat_i]
                            if sv_c.empty:
                                continue
                            fig_line_v.add_trace(
                                go.Bar(
                                    x=sv_c["Fecha_Lbl"],
                                    y=sv_c["Duracion_Viaje_hs"],
                                    name=cat_i,
                                    marker_color=col_ci,
                                    customdata=np.dstack(
                                        (
                                            sv_c["Duracion_Viaje_hs"],
                                            sv_c["Tipo_Viaje"],
                                            sv_c["KMS_Total"],
                                            sv_c["Tiempo_CD_hs"],
                                            sv_c["Tiempo_Ruta_hs"],
                                            sv_c["Tiempo_Tiendas_hs"],
                                        )
                                    )[0],
                                    hovertemplate=(
                                        "<b>%{x}</b><br>"
                                        "Duración Viaje: <b>%{customdata[0]:.1f} hs</b> (%{customdata[1]})<br>"
                                        "CD: %{customdata[3]:.1f}h | Ruta: %{customdata[4]:.1f}h | Tiendas: %{customdata[5]:.1f}h | KMS: %{customdata[2]:.0f}"
                                        "<extra></extra>"
                                    ),
                                )
                            )
                        fig_line_v.add_hline(
                            y=med_uv,
                            line_dash="dot",
                            line_color="#0f172a",
                            annotation_text=f"Mediana Viajes: {med_uv:.1f} hs",
                            annotation_position="top right",
                        )
                        fig_line_v.update_layout(
                            title=f"{cd_titulo_int} · Interno {interno_sel_s5} — Detalle Cronológico de Viajes OK (hs)",
                            height=370,
                            margin=dict(l=20, r=20, t=42, b=20),
                            yaxis=dict(title="Duración del Viaje (horas)", range=[0, float(max_dur_v) + 1.0], dtick=2),
                            xaxis=dict(tickangle=-35 if len(v_ok_int) > 10 else 0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            hovermode="closest",
                        )
                        st.plotly_chart(fig_line_v, use_container_width=True)
                    else:
                        st.info(f"El Interno {interno_sel_s5} no tiene registros limpios en el modo activo para esta métrica.")


        _render_s5_interno_fragment()

        # Descarga Excel de Viajes y Jornadas
        v_clean_exp = viajes_ok_s5.copy()
        v_clean_exp["Fecha Reparto"] = v_clean_exp["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")
        j_clean_exp = jornadas_clean_s5.copy()
        j_clean_exp["Fecha Reparto"] = j_clean_exp["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")
        j_excl_exp = jornadas_excl_s5.copy()
        j_excl_exp["Fecha Reparto"] = j_excl_exp["Fecha_Reparto"].dt.strftime("%d/%m/%Y").fillna("")

        vj_excel_bytes = build_viajes_jornadas_excel(df_v_res, df_j_res, v_clean_exp, j_clean_exp, j_excl_exp)
        st.download_button(
            label="📥 Descargar Análisis de Viajes y Jornadas (.xlsx)",
            data=vj_excel_bytes,
            file_name="Analisis_Viajes_y_Jornadas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    _render_tab5_fragment()
