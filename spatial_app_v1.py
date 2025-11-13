"""
Dash + SpatialData app
- Press "Run Pipeline" to fetch example data, run the pipeline, and render UMAP.
- Drop-down menu to switch between datasets
- Use the sliders to choose the number of highly variable genes, principal components, and nearest neighbors.
"""

# ---- Imports ----
import os, base64, tempfile
import numpy as np
import pandas as pd


# ---- Squidpy ----
import spatialdata as sd
from spatialdata_io import xenium

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import squidpy as sq

# ---- SpatialData ----
import fsspec
import spatialdata_plot
import scanpy as sc


import anndata as ad


from dash import Dash, html, dcc, Input, Output, State, no_update
import plotly.graph_objects as go

# Global variable for example dataset values
EXAMPLE_DATASETS = ["xenium"]

# ---- Prepare example data once at startup (downloaded & cached by pooch) ----
def load_example_sdata(dataset) -> sd.SpatialData:
    # FOR NOW, JUST USING A LOCAL XENIUM DATASET FOR PROOF-OF-CONCEPT
    if dataset == "xenium":
        xenium_path = "./Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs"
        zarr_path = "./Xenium.zarr"
        
        sdata = xenium(xenium_path)
        
        # convert to zarr if doesn't already exist
        sdata.write(zarr_path, overwrite=True)
        
        sdata = sd.read_zarr(zarr_path)

    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
        
    # saving a "counts" layer with the raw gene expression data
    adata = sdata.tables["table"]
    adata.layers["counts"] = adata.X.copy()
    adata.raw = adata
    sdata.tables["table"] = adata
    
    # One-time snapshot of original obs
    adata.uns.setdefault("raw_snapshots", {})
    adata.uns["raw_snapshots"]["obs_v0"] = adata.obs.copy()
    
    # Preserve original Visium coordinates
    if "spatial" in adata.obsm and "spatial_raw" not in adata.obsm:
        adata.obsm["spatial_raw"] = adata.obsm["spatial"].copy()
    
    sdata.tables["table"] = adata
    
    return sdata


def run_pipeline(sdata: sd.SpatialData, dataset: str, n_hvg: int, n_pcas: int, n_neighbs: int) -> sd.SpatialData:
    """Run the Scanpy pipeline, parameterized by n_hvg, n_pcas, and n_neighbs."""
    adata = sdata.tables["table"]
    # Work on a fresh copy so multiple runs are consistent
    adata = adata.copy()

    # Reset to raw counts, make sure values are finite, then normalize + log1p
    adata.X = adata.layers["counts"].copy()
    
    # QC
    sc.pp.calculate_qc_metrics(adata, percent_top=(10, 20, 50, 150), inplace=True)
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)
    
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    # HVGs (batch-aware across 'sample')
    # Only use batch_key if multiple samples exist
    batch_key = "sample" if ("sample" in adata.obs) else None # and adata.obs["sample"].nunique() > 1
    sc.pp.highly_variable_genes(adata, n_top_genes=int(n_hvg), batch_key=batch_key)

    # IMPORTANT: subset to HVGs so the slider actually affects PCA/UMAP
    adata = adata[:, adata.var["highly_variable"]].copy()

    # PCA → Neighbors → UMAP
    
    if n_pcas >= n_hvg:
        n_pcas = n_hvg - 1 # makes sure num PCAs does not exceed num HVGs
        
    # Adding variable to select number of PCAs
    sc.tl.pca(adata, n_comps=int(n_pcas))

    # Adding variable to select number for KNN
    sc.pp.neighbors(adata, n_neighbors=int(n_neighbs))
    
    
    sc.tl.umap(adata) # Has a default random state of zero
    
    # Leiden clustering if not already included in the data
    if "leiden" not in adata.obs:
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False, key_added="new_leiden")
        
    sdata.tables["table"] = adata

    return sdata


def umap_figure_from_adata(sdata: sd.SpatialData, dataset) -> go.Figure: ### DATASET
    """Convert adata.obsm['X_umap'] + adata.obs['leiden'] into a Plotly scatter."""
    adata = sdata.tables["table"]
    
    if "X_umap" not in adata.obsm_keys():
        raise ValueError("UMAP coordinates not found.")

    cluster_message = "Clusters are pre-calculated (changing the slider parameters will not impact clusters)"
    
    # Checking for different potential column names that would contain celltype/cluster info for UMAP
    if "cell_type" in adata.obs:
        cell_type_index = "cell_type"
    elif "clusters" in adata.obs:
        cell_type_index = "clusters"
    elif "leiden" in adata.obs:
        cell_type_index = "leiden"
    elif "louvain" in adata.obs:
        cell_type_index = "louvain"
    else:
        cell_type_index = "new_leiden"
        cluster_message = "Clusters calculated using slider-specified parameters"
    

    coords = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
    coords["cluster"] = adata.obs[cell_type_index].astype(str).values


    fig = go.Figure()
    for cluster in sorted(coords["cluster"].unique()):
        sub = coords[coords["cluster"] == cluster]
        fig.add_trace(
            go.Scattergl(
                x=sub["UMAP1"],
                y=sub["UMAP2"],
                mode="markers",
                name=str(cluster),
                marker={"size": 4, "opacity": 0.8},
                hovertemplate=f"cluster: {cluster}<extra></extra>"
            )
        )

    fig.update_layout(
        title=f"UMAP colored by cell type",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        template="plotly_white",
        legend_title_text="cell type label",
        margin=dict(l=20, r=20, t=40, b=20),
        dragmode="pan",
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig, cluster_message
    
def spatial_scatter_figure_from_adata(sdata: sd.SpatialData, dataset) -> go.Figure:
    adata = sdata.tables["table"]
    # Checking for different potential column names that would contain celltype/cluster info for UMAP
    if "cell_type" in adata.obs:
        cell_type_index = "cell_type"
    elif "clusters" in adata.obs:
        cell_type_index = "clusters"
    elif "leiden" in adata.obs:
        cell_type_index = "leiden"
    elif "louvain" in adata.obs:
        cell_type_index = "louvain"
    else:
        cell_type_index = "new_leiden"
    
    
    fig = sq.pl.spatial_scatter(
        adata,
        library_id="spatial",
        shape=None,
        color="cell_type_index",
        wspace=0.4,
    )
    
    fig.update_layout(
        title=f"Spatial coordinates colored by cell type",
        xaxis_title="spatial1",
        yaxis_title="spatial2",
        template="plotly_white",
        legend_title_text="cell type label",
        margin=dict(l=20, r=20, t=40, b=20),
        dragmode="pan",
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    
    return fig


# ---- Dash app ----
app = Dash(__name__)
server = app.server  # for production servers (e.g., gunicorn)

app.title = "Interactive Squidpy"

app.layout = html.Div(
    style={"maxWidth": "900px", "margin": "0 auto", "fontFamily": "system-ui, sans-serif", "marginBottom": "24px"},
    children=[
        html.H2("Interactive spatial proteogenomics SpatialData pipeline"),
        html.P("Toggle HVG, PC, and KNN parameters and produce the resulting UMAP using the scanpy analysis pipeline."
        ),
        html.P("Please select and example Xenium dataset."
        ),
        html.Div([
                html.Label("Dataset", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    clearable=False,
                    options=[
                        {"label": "Xenium Human Brain", "value": "xenium"}
                    ],
                    value="xenium",       # default
                ),
            ],
            style={"marginBottom": "24px"},
        ),
        html.Div(
            style={"marginBottom": "24px"},
            children=html.P(
                "New example datasets coming soon...",
                style={"fontStyle": "italic"}
                )
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr auto", "gap": "16px", "alignItems": "center", "marginBottom": "24px"},
            children=[
                html.Div(
                    children=[
                        html.Label("Highly variable genes"),
                        dcc.Slider(
                            id="hvg-slider",
                            min=5,
                            max=15000,
                            step=100,
                            value=2000,  # default as requested
                            marks={10: "10", 500: "500", 1000: "1000", 2000: "2000", 3000: "3000", 4000: "4000", 5000: "5000", 10000: "10000", 15000: "15000"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr auto", "gap": "16px", "alignItems": "center", "marginBottom": "24px"},
            children=[
                html.Div(
                    children=[
                        html.Label("Principal components"),
                        dcc.Slider(
                            id="pca-slider",
                            min=2,
                            max=100,
                            step=1,
                            value=50,  # default as requested
                            marks={2: "2", 10: "10", 20: "20", 50: "50", 100: "100"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr auto", "gap": "16px", "alignItems": "center", "marginBottom": "24px"},
            children=[
                html.Div(
                    children=[
                        html.Label("Number of nearest neighbors (for UMAP)"),
                        dcc.Slider(
                            id="neighbor-slider",
                            min=2,
                            max=100,
                            step=1,
                            value=15,  # default as requested
                            marks={2: "2", 10: "10", 15: "15", 25: "25", 30: "30", 50: "50", 100: "100"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            html.Button("Run Pipeline", id="run-btn", n_clicks=0, style={"height": "44px", "minWidth": "200px"}),
            style={"display": "flex", "justifyContent": "center", "margin": "8px 0 24px"}
            ),
        html.Div(id="status", style={"margin": "10px 0 10px", "fontSize": "0.95rem", "color": "#444", "whiteSpace": "pre-wrap"}),
        html.Div(id="cluster-message", style={"margin": "16px 0 10px", "fontSize": "0.95rem", "color": "#444", "whiteSpace": "pre-wrap"}),
        dcc.Loading(
            id="loading",
            type="default",
            children=dcc.Graph(
                id="umap-graph",
                figure=go.Figure(layout={"template": "plotly_white"}),
                style={"height": "640px", "marginTop": "10px"},
                config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
            ),
        ),
        dcc.Loading(
            id="loading",
            type="default",
            children=dcc.Graph(
                id="scatter-graph",
                figure=go.Figure(layout={"template": "plotly_white"}),
                style={"height": "640px", "marginTop": "10px"},
                config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
            ),
        ),
        html.Div([
            html.P([
                "Xenium human brain dataset from ",
                html.A("10x Genomics", href="https://www.10xgenomics.com/datasets/xenium-human-brain-preview-data-1-standard", target="_blank", rel="noopener noreferrer"),
                "."
            ]),
            html.P("Tip: rerun with different HVG, PC, and KNN settings to see stability."),
            html.P("Typical ranges:"),
            html.P("HVG: 1,000–5,000 depending on tissue & depth  |  PCs: 10-100, increasing with dataset size  |  KNN: 5-15 for small datasets, but can go up to 60s for very large datasets"),
            html.P("Warning: If set number of HVGs > PCs, PCs will be automatically set to HVGs - 1")
            ], style={"marginTop": "8px", "color": "#666", "fontSize": "0.9rem"},
        ),
    ],
)

@app.callback(
    Output("umap-graph", "figure"),
    Output("scatter-graph", "figure"),
    Output("status", "children"),
    Output("cluster-message", "children"),
    Input("run-btn", "n_clicks"),
    State("dataset-dropdown", "value"),
    State("hvg-slider", "value"),
    State("pca-slider", "value"),
    State("neighbor-slider", "value"),
    prevent_initial_call=True,
)
def on_run(n_clicks: int, dataset: str, n_hvg: int, n_pcas: int, n_neighbs: int):
    # Run the pipeline
    try:
        status_lines = []
        
        if dataset in EXAMPLE_DATASETS:
            status_lines.append(f"Loading example dataset: {dataset} …")
            sdata = load_example_sdata(dataset)
        else:
            return go.Figure(), "Please choose a data source."
        
        # Run the analysis
        status_lines.append("Running pipeline (HVG → PCA → neighbors → UMAP → spatial scatter)")
        sdata = run_pipeline(sdata=sdata, dataset=dataset, n_hvg=int(n_hvg), n_pcas=int(n_pcas), n_neighbs=int(n_neighbs))
        fig1, cluster_message = umap_figure_from_adata(sdata, dataset)
        fig2 = spatial_scatter_figure_from_adata(sdata, dataset)
        status_lines.append(f"Done. Dataset: {dataset}  |  Cells: {adata.n_obs:,}  |  Genes (HVGs): {adata.n_vars:,}  |  PCs: {min(n_pcas, adata.n_vars)}  |  kNN: {n_neighbs}")
        return fig1, fig2, "\n\n".join(status_lines), cluster_message
    
    except Exception as e:
        # Report the error to the UI
        err = f"Error: {type(e).__name__}: {e}"
        return no_update, err


if __name__ == "__main__":
    # Bind to localhost by default; set debug=True if you like
    app.run(host="127.0.0.1", port=8050, debug=False)
