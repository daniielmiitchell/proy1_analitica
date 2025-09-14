import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# ========== Cargar datos ==========
df = pd.read_csv("incident_event_log.csv")

# Fechas a datetime
for col in ["opened_at", "resolved_at", "closed_at", "sys_created_at", "sys_updated_at"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Calcular TTR en horas
if {"opened_at", "resolved_at"}.issubset(df.columns):
    df["ttr_h"] = (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600

# Convertir made_sla a booleano
if "made_sla" in df.columns:
    df["made_sla"] = df["made_sla"].astype(str).str.lower().isin(["1", "true", "yes", "y"])

# ========== App ==========
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Tablero de Incidentes"

# Valores iniciales para filtros
date_min = df["opened_at"].min().date()
date_max = df["opened_at"].max().date()

# Layout
app.layout = dbc.Container([
    html.H2("📊 Tablero de Incidentes – Jefe de Mesa de Ayuda", className="my-3"),

    # --- Filtros ---
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id="f_cat",
            options=[{"label": c, "value": c} for c in sorted(df["category"].dropna().unique())],
            placeholder="Categoría"
        ), md=3),
        dbc.Col(dcc.Dropdown(
            id="f_grp",
            options=[{"label": g, "value": g} for g in sorted(df["assignment_group"].dropna().unique())],
            placeholder="Grupo de soporte"
        ), md=3),
        dbc.Col(dcc.Dropdown(
            id="f_prio",
            options=[{"label": p, "value": p} for p in sorted(df["priority"].dropna().unique())],
            placeholder="Prioridad"
        ), md=2),
        dbc.Col(dcc.DatePickerRange(
            id="f_dates",
            start_date=date_min,
            end_date=date_max
        ), md=4),
    ], className="g-2 mb-3"),

    # --- KPIs ---
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("% SLA cumplido"), html.H2(id="kpi_sla")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("TTR medio (h)"), html.H2(id="kpi_ttr")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("# Incidentes"), html.H2(id="kpi_cnt")]))),
    ], className="g-3 mb-4"),

    # --- Gráficas ---
    dbc.Row([
        dbc.Col(dcc.Graph(id="g_barras_cat"), md=6),
        dbc.Col(dcc.Graph(id="g_box_ttr"), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="g_heatmap"), md=6),
        dbc.Col(dcc.Graph(id="g_serie"), md=6),
    ]),
])

# ========== Callbacks ==========
@app.callback(
    [Output("kpi_sla","children"),
     Output("kpi_ttr","children"),
     Output("kpi_cnt","children"),
     Output("g_barras_cat","figure"),
     Output("g_box_ttr","figure"),
     Output("g_heatmap","figure"),
     Output("g_serie","figure")],
    [Input("f_cat","value"),
     Input("f_grp","value"),
     Input("f_prio","value"),
     Input("f_dates","start_date"),
     Input("f_dates","end_date")]
)
def update_dashboard(cat, grp, prio, dstart, dend):
    data = df.copy()

    # Aplicar filtros
    if dstart and dend:
        data = data[(data["opened_at"] >= dstart) & (data["opened_at"] <= dend)]
    if cat: data = data[data["category"] == cat]
    if grp: data = data[data["assignment_group"] == grp]
    if prio: data = data[data["priority"] == prio]

    # KPIs
    total = len(data)
    sla = f"{data['made_sla'].mean()*100:.1f}%" if total else "0%"
    ttr = f"{data['ttr_h'].mean():.1f}" if "ttr_h" in data and total else "-"
    cnt = str(total)

    # Gráfica barras
    fig_barras = px.bar(
        data.groupby("category").size().reset_index(name="count"),
        x="category", y="count", title="Incidentes por Categoría"
    )

    # Boxplot TTR
    fig_box = px.box(
        data.dropna(subset=["ttr_h","priority"]),
        x="priority", y="ttr_h", title="Distribución TTR por Prioridad"
    )

    # Heatmap día-hora
    tmp = data.dropna(subset=["opened_at"]).copy()
    tmp["day"] = tmp["opened_at"].dt.day_name()
    tmp["hour"] = tmp["opened_at"].dt.hour
    piv = tmp.pivot_table(index="day", columns="hour", values="number", aggfunc="count").fillna(0)
    fig_heat = px.imshow(piv, aspect="auto", title="Incidentes por Día y Hora")

    # Serie temporal mensual
    ts = (data.assign(month=lambda x: x["opened_at"].dt.to_period("M").dt.to_timestamp())
               .groupby("month").size().reset_index(name="count"))
    fig_serie = px.line(ts, x="month", y="count", title="Tendencia Mensual de Aperturas")

    return sla, ttr, cnt, fig_barras, fig_box, fig_heat, fig_serie

# ========== Run ==========
if __name__ == "__main__":
    app.run(debug=True)
