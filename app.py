from dash import html, Dash, dcc, callback, Input, Output, Patch, dash_table
from whitenoise import WhiteNoise
import dash_bootstrap_components as dbc
from dash_ag_grid import AgGrid
import pandas as pd
import os
import pickle
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from constants import OUTPUT_PATH

app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
           meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
           suppress_callback_exceptions=True)
server = app.server
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/')

with open(os.path.join(OUTPUT_PATH, 'ports.pkl'), 'rb') as file:
    ports = pickle.load(file)

benchmark_stats = pd.read_csv(os.path.join(OUTPUT_PATH, "benchmark_stats.csv"), index_col=0)
strs_stats = pd.read_csv(os.path.join(OUTPUT_PATH, "strs_res_stats.csv"), index_col=0)
strs_rtn = pd.read_csv(os.path.join(OUTPUT_PATH, "rtn.csv"), index_col=0) * 100
strs_rtn_com = pd.read_csv(os.path.join(OUTPUT_PATH, "rtn_com.csv"), index_col=0) * 100
strs_alpha = pd.read_csv(os.path.join(OUTPUT_PATH, "alpha.csv"), index_col=0) * 100
strs_alpha_com = pd.read_csv(os.path.join(OUTPUT_PATH, "alpha_com.csv"), index_col=0) * 100
multivariate_input = pd.read_csv(os.path.join(OUTPUT_PATH, "multivariate_input.csv"), index_col=0)
multivariate_input.index.name = "Factor"

rename_dict = {"rtn_avg": "Return AVG (%)", "rtn_std": "Return STD (%)",
               "rtn_downside_std": "Downside return STD (%)", "rtn_tot": "Return Total (%)",
               "rtn_risk_adj": "Return Risk Adj (%)", "sharpe_ratio": "Sharpe Ratio",
               "sortino_ratio": "Sortino Ratio", "info_ratio": "Information Ratio", "alpha_avg": "Alpha AVG (%)",
               "alpha_tot": "Alpha Total (%)", "treynor_ratio": "Treynor ratio", "beta": "Beta",
               "factors_used": "Factors used"}

benchmark_stats = benchmark_stats.rename(index=rename_dict, columns={"benchmark_stats": "Value"})
benchmark_stats.iloc[:-2] = 100 * benchmark_stats.iloc[:-2]
benchmark_stats = benchmark_stats.round(4)
benchmark_stats.index.name = "Statistics"
benchmark_stats_table = dbc.Table.from_dataframe(benchmark_stats.reset_index(), bordered=True, hover=True,
                                                 responsive=True, striped=True)

strs_stats = strs_stats.rename(columns=rename_dict)
strs_stats.index.name = "Strategies/Factors"
strs_stats.columns.name = "Statistics"
strs_stats.iloc[:, 1:8] = 100 * strs_stats.iloc[:, 1:8]
strs_stats = strs_stats.round(4)
strs_stats_table = AgGrid(rowData=strs_stats.iloc[:, :-1].reset_index().to_dict('records'),
                          columnDefs=[{'field': col} for col in strs_stats.reset_index().columns[1:-1]] +
                                     [{'field': strs_stats.reset_index().columns[0], 'pinned': 'left',
                                       'minWidth': 250, "checkboxSelection": True}],
                          defaultColDef={"resizable": True, "sortable": True, "filter": True,
                                         "wrapText": True, 'autoHeight': True,
                                         "wrapHeaderText": True, "autoHeaderHeight": True}, id='strs_stats_table',
                          style={'height': '600px'}, dashGridOptions={'rowSelection': 'multiple'},
                          columnSize='responsiveSizeToFit', className="ag-theme-alpine custom-compact")
multi_cols = strs_rtn.columns[-3:].values.tolist()

stats_summary = make_subplots(rows=4, cols=3, subplot_titles=strs_stats.columns[:-1], shared_xaxes=True)
for i in range(len(strs_stats.columns[:-1])):
    stats_summary.add_trace(go.Bar(x=strs_stats.index, y=strs_stats.iloc[:, i], name=strs_stats.columns[i]),
                            row=int(i / 3) + 1, col=(i % 3) + 1)

stats_summary.update_layout(height=1000, showlegend=False, hovermode="x unified")


app.layout = dbc.Container(fluid=True, children=[dbc.Container(className="fluid", children=[

    dcc.Location(id='url', refresh=False),
    html.Center(html.H1("Presentation of results", className="display-3 my-4")),
    html.Center(html.H5("Benchmark statistics")),
    html.Div(className="my-3", children=benchmark_stats_table),

    html.Center(html.H3("Univariate strategy analysis", className="my-4")),
    html.Center(html.H5("Logarithmic returns")),
    dcc.Graph(figure=px.line(strs_rtn.iloc[:, :-3], x=strs_rtn.index, y=strs_rtn.columns[:-3],
                             labels={"value": "Return (%)", "variable": "Factor", 'date': 'Date'}, render_mode='webg1'
                             ).update_xaxes(rangeslider_visible=True).update_layout(hovermode="x unified")),

    html.Center(html.H5("Compound returns")),
    dcc.Graph(figure=px.line(strs_rtn_com.iloc[:, :-3], x=strs_rtn_com.index, y=strs_rtn_com.columns[:-3],
                             labels={"value": "Return (%)", "variable": "Factor", 'date': 'Date'}, render_mode='webg1'
                             ).update_xaxes(rangeslider_visible=True).update_layout(hovermode="x unified")),

    html.Center(html.H5("Information Ratio", className="my-4")),
    dcc.Graph(figure=px.bar(strs_stats.iloc[:-3], x=strs_stats.iloc[:-3].index, y="Information Ratio",
                            color="Return Total (%)", color_continuous_scale=px.colors.sequential.thermal,
                            hover_data=["Return Total (%)"])),

    html.Center(html.H3("Multivariate strategy analysis", className="my-4")),
    html.Div(className="my-3", children=dbc.Table.from_dataframe(multivariate_input.reset_index(), bordered=True,
                                                                 hover=True, responsive=True, striped=True)),
    html.Center(html.H5("Logarithmic returns")),
    dcc.Graph(figure=px.line(strs_rtn[['benchmark'] + multi_cols], x=strs_rtn.index, y=['benchmark'] + multi_cols,
                             labels={"value": "Return (%)", "variable": "Strategy", 'date': 'Date'},
                             ).update_xaxes(rangeslider_visible=True).update_layout(hovermode="x unified")),

    html.Center(html.H5("Compound returns")),
    dcc.Graph(figure=px.line(strs_rtn_com[['benchmark'] + multi_cols], x=strs_rtn_com.index,
                             y=['benchmark'] + multi_cols,
                             labels={"value": "Return (%)", "variable": "Strategy", 'date': 'Date'},
                             ).update_xaxes(rangeslider_visible=True).update_layout(hovermode="x unified")),

    html.Center(html.H5("Information ratio", className="my-4")),
    dcc.Graph(figure=px.bar(strs_stats.loc[multi_cols], x=strs_stats.loc[multi_cols].index, y="Information Ratio",
                            color="Return Total (%)", color_continuous_scale=px.colors.sequential.thermal,
                            hover_data=["Return Total (%)"])),

    html.Center(html.H3("Analysis of positions in each strategy", className="my-4")),
    dbc.Select([i for i in strs_stats.index.values if "no-fees" not in i], value=strs_stats.index.values[0],
               id="strategy_selector", className="date-group-items justify-content-center mt-4"),
    dcc.Graph(id="strategy_positions"),
    html.Center(html.H3("Total statistics of strategies", className="my-5")),
    html.Center(html.H5("Alpha at each period", className="my-3")),
    dcc.Graph(figure=px.line(strs_alpha, x=strs_alpha.index, y=strs_alpha.columns, render_mode='webg1',
                             labels={"value": "Alpha (%)", "variable": "Factor/Strategy", 'date': 'Date'},
                             ).update_xaxes(rangeslider_visible=True).update_layout(hovermode="x unified")),

    html.Center(html.H5("Compound Alpha", className="my-3")),
    dcc.Graph(figure=px.line(strs_alpha_com, x=strs_alpha_com.index, y=strs_alpha_com.columns, render_mode='webg1',
                             labels={"value": "Alpha (%)", "variable": "Factor/Strategy", 'date': 'Date'},
                             ).update_xaxes(rangeslider_visible=True).update_layout(hovermode="x unified")),
]), html.Div(className="mx-5", children=[
    html.Center(html.H5("Summary table of statistics", className="my-3")),
    html.Div(className="my-3", children=strs_stats_table),
    html.Center(html.H5("Summary graph of statistics", className="my-3")),
    dcc.Graph(figure=stats_summary),

])])

color_scale = [[0.00, "rgb(255,0,0)"], [0.25, "rgb(255,0,0)"],  # red
               [0.25, "rgb(0,0,0)"], [0.50, "rgb(0,0,0)"],  # black
               [0.50, "rgb(255,255,255)"], [0.75, "rgb(255,255,255)"],  # white
               [0.75, "rgb(0,128,0)"], [1.00, "rgb(0,128,0)"]]  # green


@app.callback(Output("strategy_positions", "figure"), Input("strategy_selector", "value"), )
def update_strategy_positions(strategy):
    port_fig = px.imshow(ports[strategy].transpose(), color_continuous_scale=color_scale
                         ).update_yaxes(showticklabels=False).update_layout(coloraxis_colorbar=dict(
        title="Operation", tickvals=[0.125, 0.375, 0.625, 0.875], ticktext=["Removed", "Not Present",
                                                                            "Present", "Added"]))
    return port_fig


@callback(Output("strs_stats_table", "dashGridOptions"),
          Input("strs_stats_table", "selectedRows"), prevent_initial_call=True)
def row_pinning_top(selected_rows):
    grid_option_patch = Patch()
    grid_option_patch["pinnedTopRowData"] = selected_rows
    return grid_option_patch

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
# %%
# %%
