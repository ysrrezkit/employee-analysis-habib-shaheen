import os
import logging
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from huggingface_hub import InferenceClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load and prepare HR Dataset
# If reading directly from a local file, ensure 'employees.csv' matches your file name
try:
    df = pd.read_csv("employees.csv")
except FileNotFoundError:
    # Fallback to loading a subset string if file doesn't exist locally yet
    import io
    logger.warning("employees.csv not found, using placeholder layout logic.")
    df = pd.DataFrame(columns=["Employee ID", "Name", "Role", "Department", "Monthly Salary (EGP)", "Source", "Referral Bonus", "Referred By Name"])

# Clean numerical types
df["Monthly Salary (EGP)"] = pd.to_numeric(df["Monthly Salary (EGP)"], errors='coerce').fillna(0)
df["Referral Bonus"] = pd.to_numeric(df["Referral Bonus"], errors='coerce').fillna(0)

HF_TOKEN = os.getenv("HF_TOKEN")
client = None
MODEL = None

if HF_TOKEN:
    try:
        client = InferenceClient(token=HF_TOKEN)
        logger.info("Hugging Face client initialized successfully.")
        MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" 
    except Exception as e:
        logger.error(f"HF init error: {e}")

def generate_insights(dataframe):
    if dataframe.empty or client is None or MODEL is None:
        return "⚡ AI Insights are currently warming up. Adjust filters to update operational analysis."

    try:
        dataframe = dataframe.fillna(0)
        summary_text = f"""
Total Monthly Payroll: {dataframe['Monthly Salary (EGP)'].sum():,.0f} EGP
Total Headcount: {dataframe['Employee ID'].nunique()}
Unique Specialized Roles: {dataframe['Role'].nunique()}
Total Referral Bonus Paid: {dataframe['Referral Bonus'].sum():,.0f} EGP
Top Department Budget: {dataframe.groupby('Department')['Monthly Salary (EGP)'].sum().idxmax()}
Primary Recruitment Source: {dataframe['Source'].value_counts().idxmax()}
"""

        messages = [
            {"role": "system", "content": "You are a senior HR director and talent acquisition strategist. Provide 5 short, actionable, and professional insights based on the organizational data summary provided. your answers is breif short no more than 3 lines and no bolding only insights answer."},
            {"role": "user", "content": summary_text}
        ]

        response = client.chat_completion(
            model=MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI system response: {str(e)[:200]}"

def calculate_kpis(df_):
    if df_.empty:
        return 0, 0, 0, 0
    return (
        df_["Monthly Salary (EGP)"].sum(),
        df_["Employee ID"].nunique(),
        df_["Role"].nunique(),
        df_["Monthly Salary (EGP)"].mean()
    )

DARK_STYLE = {
    "background-color": "#0b0f19",
    "color": "#010c17",
    "font-family": "'Inter', 'Segoe UI', sans-serif",
    "min-height": "100vh",
    "padding": "24px"
}

CARD_STYLE = {
    "background": "linear-gradient(145deg, #111827, #1f2937)",
    "border": "1px solid #2d3748",
    "border-radius": "12px",
    "box-shadow": "0 4px 20px 0 rgba(0, 0, 0, 0.3)",
    "padding": "15px",
    "transition": "transform 0.2s"
}

PLOTLY_DARK_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#94a3b8", "family": "'Inter', sans-serif"},
    "xaxis": {"gridcolor": "#1e293b", "zerolinecolor": "#1e293b"},
    "yaxis": {"gridcolor": "#1e293b", "zerolinecolor": "#1e293b"},
    "margin": {"t": 40, "b": 40, "l": 40, "r": 40}
}

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server
app.title = "HR Talent & Payroll Analytics Dashboard"

app.layout = html.Div(style=DARK_STYLE, children=[
    dbc.Container([
        
        html.Div([
            html.H1("HR Talent & Payroll Analytics Engine", 
                    style={"letter-spacing": "2px", "font-weight": "800", "background": "linear-gradient(to right, #38bdf8, #818cf8)", "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent"}),
            html.P("Real-time Human Capital Intelligence Dashboard", style={"color": "#64748b", "font-size": "14px", "margin-top": "-5px"})
        ], className="text-center my-4"),

        # Filter Section
        dbc.Row([
            dbc.Col([
                html.Div(className="dash-dropdown-grid-container dash-dropdown-trigger", children=[
                    html.Label("🏢 Corporate Department", style={"color": "#38bdf8", "font-weight": "600", "margin-bottom": "6px"}),
                    dcc.Dropdown(
                        id="department-filter",
                        options=[{"label": d, "value": d} for d in sorted(df["Department"].unique())] if not df.empty else [],
                        value=list(df["Department"].unique()) if not df.empty else [],
                        multi=True,
                        className="dash-bootstrap"
                    )
                ])
            ], md=4, className="mb-3"),

            dbc.Col([
                html.Div(className="dash-dropdown-grid-container dash-dropdown-trigger", children=[
                    html.Label("🛠️ Operational Role", style={"color": "#38bdf8", "font-weight": "600", "margin-bottom": "6px"}),
                    dcc.Dropdown(
                        id="role-filter",
                        options=[{"label": r, "value": r} for r in sorted(df["Role"].unique())] if not df.empty else [],
                        value=list(df["Role"].unique()) if not df.empty else [],
                        multi=True,
                        className="dash-bootstrap"
                    )
                ])
            ], md=4, className="mb-3"),

            dbc.Col([
                html.Div(className="dash-dropdown-grid-container dash-dropdown-trigger", children=[
                    html.Label("🎯 Sourcing Channel", style={"color": "#38bdf8", "font-weight": "600", "margin-bottom": "6px"}),
                    dcc.Dropdown(
                        id="source-filter",
                        options=[{"label": s, "value": s} for s in sorted(df["Source"].unique())] if not df.empty else [],
                        value=list(df["Source"].unique()) if not df.empty else [],
                        multi=True,
                        className="dash-bootstrap"
                    )
                ])
            ], md=4, className="mb-3"),
        ], className="mb-4"),

        # KPI Metrics Cards Section
        dbc.Row([
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                html.H6("MONTHLY PAYROLL BUDGET", style={"color": "#64748b", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="total-payroll", style={"color": "#34d399", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                html.H6("TOTAL ACTIVE HEADCOUNT", style={"color": "#64748b", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="total-headcount", style={"color": "#38bdf8", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                html.H6("UNIQUE SPECIALIZED ROLES", style={"color": "#64748b", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="unique-roles", style={"color": "#a78bfa", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=CARD_STYLE, children=[
                html.H6("AVERAGE MONTHLY SALARY", style={"color": "#64748b", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="avg-salary", style={"color": "#fb923c", "font-weight": "700"})
            ]), md=3, className="mb-3"),
        ], className="mb-4"),

        # AI Insights Section
        html.Div(style={**CARD_STYLE, "background": "linear-gradient(145deg, #1e1b4b, #111827)", "border-color": "#4338ca"}, children=[
            html.H5("✨ Neural AI Workforce Insights", style={"color": "#818cf8", "font-weight": "700", "margin-bottom": "12px"}),
            html.Div(id="ai-insights", style={
                "color": "#cbd5e1", 
                "whiteSpace": "pre-line", 
                "font-size": "14px", 
                "line-height": "1.7",
                "font-family": "inherit"
            })
        ], className="mb-4"),

        # Row 1 Charts
        dbc.Row([
            dbc.Col(html.Div(style=CARD_STYLE, children=[dcc.Graph(id="salary-trend", config={"displayModeBar": False})]), md=6, className="mb-4"),
            dbc.Col(html.Div(style=CARD_STYLE, children=[dcc.Graph(id="category-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
        ]),

        # Row 2 Charts
        dbc.Row([
            dbc.Col(html.Div(style=CARD_STYLE, children=[dcc.Graph(id="city-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
            dbc.Col(html.Div(style=CARD_STYLE, children=[dcc.Graph(id="rep-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
        ]),

        # Data Ledger Section
        html.H4("Active Corporate Personnel Ledger", className="mt-2 mb-3", style={"color": "#94a3b8", "font-weight": "600"}),
        html.Div(style={"border-radius": "12px", "overflow": "hidden", "border": "1px solid #2d3748"}, children=[
            dash_table.DataTable(
                id="sales-table",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left", 
                    "backgroundColor": "#111827", 
                    "color": "#cbd5e1",
                    "border": "1px solid #1f2937",
                    "padding": "12px 15px",
                    "font-family": "'Inter', sans-serif"
                },
                style_header={
                    "backgroundColor": "#1f2937",
                    "color": "#38bdf8",
                    "fontWeight": "bold",
                    "border": "1px solid #2d3748"
                },
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#1f2937', #type: ignore
                }]
            )
        ], className="mb-5")

    ], fluid=True)
])

@app.callback(
    [
        Output("total-payroll", "children"),
        Output("total-headcount", "children"),
        Output("unique-roles", "children"),
        Output("avg-salary", "children"),
        Output("salary-trend", "figure"),
        Output("category-chart", "figure"),
        Output("city-chart", "figure"),
        Output("rep-chart", "figure"),
        Output("sales-table", "data"),
        Output("sales-table", "columns"),
        Output("ai-insights", "children"),
    ],
    [
        Input("department-filter", "value"),
        Input("role-filter", "value"),
        Input("source-filter", "value"),
    ]
)
def update_dashboard(departments, roles, sources):
    departments = departments or list(df["Department"].unique())
    roles = roles or list(df["Role"].unique())
    sources = sources or list(df["Source"].unique())

    filtered = df[
        df["Department"].isin(departments) &
        df["Role"].isin(roles) &
        df["Source"].isin(sources)
    ]

    payroll, headcount, unique_roles, avg_sal = calculate_kpis(filtered)

    if not filtered.empty:
        # Chart 1: Average Salary Scaling per Role Group
        trend_df = filtered.groupby("Role")["Monthly Salary (EGP)"].mean().reset_index().sort_values(by="Monthly Salary (EGP)", ascending=False)
        trend = px.bar(trend_df, x="Role", y="Monthly Salary (EGP)", title="Role Compensation Scaling (Avg EGP)")
        trend.update_traces(marker_color="#38bdf8", marker_line_color="#0369a1", marker_line_width=1.5)
        
        # Chart 2: Payroll Expenditures Allocated Across Corporate Departments
        cat_df = filtered.groupby("Department")["Monthly Salary (EGP)"].sum().reset_index()
        category = px.bar(cat_df, x="Department", y="Monthly Salary (EGP)", title="Payroll Budget Allocation Across Verticals")
        category.update_traces(marker_color="#818cf8", marker_line_color="#4338ca", marker_line_width=1.5)
        
        # Chart 3: Recruitment Channels Share
        city_df = filtered.groupby("Source")["Employee ID"].count().reset_index().rename(columns={"Employee ID": "Count"})
        city = px.pie(city_df, names="Source", values="Count", title="Recruitment Sourcing Share Channels", hole=0.4)
        city.update_traces(textinfo='percent+label', marker=dict(colors=["#38bdf8", "#818cf8", "#a78bfa", "#f43f5e"]))
        
        # Chart 4: Top Employee Referral Networks (Who referred who)
        #