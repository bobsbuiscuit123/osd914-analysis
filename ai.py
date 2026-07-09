import os
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")

import html
import time
import traceback
import webbrowser
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

print("DEBUG: Matplotlib backend forced to Agg for browser dashboard rendering.", flush=True)

HIGH_CONFIDENCE_LIVER_MIRNAS = [
    "mmu-miR-122-5p",
    "mmu-miR-17-5p",
    "mmu-miR-18a-5p",
    "mmu-miR-19a-5p",
    "mmu-miR-20a-5p",
    "mmu-miR-92a-1-5p",
]


def debug(message):
    print(f"DEBUG: {time.strftime('%H:%M:%S')} - {message}", flush=True)


def log1p_zscore(df):
    debug("Applying log1p + z-score transform...")
    transformed = np.log1p(df)
    std = transformed.std(axis=0, ddof=0).replace(0, 1)
    normalized = (transformed - transformed.mean(axis=0)) / std
    debug("Finished log1p + z-score transform.")
    return normalized


# ==========================================
# 1. RESEARCH-GRADE MLP ARCHITECTURE
# ==========================================
class LiverBrainCrossTalkMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, dropout_rate=0.3):
        super(LiverBrainCrossTalkMLP, self).__init__()
        
        # Shallow network architecture with Dropout to prevent overfitting on small n
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)


# ==========================================
# 2. CLEAN DATA LOADING
# ==========================================
def load_clean_research_data(
    liver_features_path="liver_features_clean.csv",
    brain_targets_path="brain_targets_clean.csv"
):
    debug("Loading data...")
    debug(f"Reading liver features from {liver_features_path}...")
    X_df = pd.read_csv(liver_features_path, index_col=0)
    debug(f"Loaded liver features with shape {X_df.shape}.")
    debug(f"Reading brain targets from {brain_targets_path}...")
    y_df = pd.read_csv(brain_targets_path, index_col=0)
    debug(f"Loaded brain targets with shape {y_df.shape}.")

    debug("Aligning sample IDs between liver and brain matrices...")
    matching_subjects = X_df.index.intersection(y_df.index)
    if matching_subjects.empty:
        raise ValueError("No matching sample IDs found between clean liver and brain matrices.")

    X_df = X_df.loc[matching_subjects].sort_index()
    y_df = y_df.loc[matching_subjects].sort_index()
    debug(f"Aligned matrices to {len(matching_subjects)} shared samples.")

    debug("Selecting liver miRNA input features...")
    selected_liver_mirnas = [
        mirna for mirna in HIGH_CONFIDENCE_LIVER_MIRNAS
        if mirna in X_df.columns
    ]
    if not selected_liver_mirnas:
        debug("No high-confidence liver miRNAs found; falling back to top 10 variance filter.")
        selected_liver_mirnas = (
            X_df.var(axis=0)
            .sort_values(ascending=False)
            .head(10)
            .index
            .tolist()
        )
    else:
        debug(f"Found {len(selected_liver_mirnas)} high-confidence liver miRNAs.")

    debug("Selecting top 5 highest-variance brain target miRNAs...")
    selected_brain_targets = (
        y_df.var(axis=0)
        .sort_values(ascending=False)
        .head(5)
        .index
        .tolist()
    )

    X_df = X_df[selected_liver_mirnas]
    y_df = y_df[selected_brain_targets]

    print(f"Selected liver miRNA features: {', '.join(selected_liver_mirnas)}")
    print(f"Selected brain target miRNAs: {', '.join(selected_brain_targets)}")
    debug("Feature and target selection complete.")

    X_df = log1p_zscore(X_df)
    y_df = log1p_zscore(y_df)
    print("Applied log1p + z-score normalization to selected feature and target matrices.")

    debug("Converting pandas matrices to float32 tensors...")
    X = X_df.to_numpy(dtype=np.float32)
    y = y_df.to_numpy(dtype=np.float32)
    debug(f"Tensor conversion complete: X={X.shape}, y={y.shape}.")

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        selected_liver_mirnas,
        selected_brain_targets,
    )


# ==========================================
# 3. LEAVE-ONE-OUT CROSS-VALIDATION TRAINING LOOP
# ==========================================
def train_and_evaluate():
    debug("Starting train_and_evaluate()...")
    X, y, feature_names, target_names = load_clean_research_data()
    num_samples, input_dim = X.shape
    output_dim = y.shape[1]
    debug(f"Data ready for training: samples={num_samples}, input_dim={input_dim}, output_dim={output_dim}.")
    
    print(f"Initialized Pipeline with {num_samples} samples.")
    print(f"Input features (Liver miRNAs): {input_dim} | Output targets (Brain Pathways): {output_dim}\n")
    
    # Leave-One-Out Cross-Validation is standard for small-n spaceflight omics
    oo_losses = []
    
    for i in range(num_samples):
        debug(f"Starting LOOCV fold {i+1}/{num_samples}...")
        # Split into training (8 samples) and validation (1 sample)
        X_train = torch.cat([X[:i], X[i+1:]], dim=0)
        y_train = torch.cat([y[:i], y[i+1:]], dim=0)
        X_val, y_val = X[i:i+1], y[i:i+1]
        
        # Instantiate fresh model instance
        debug(f"Initializing model for fold {i+1}...")
        model = LiverBrainCrossTalkMLP(input_dim=input_dim, output_dim=output_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-2) # Stronger L2 regularization included
        
        # Training Loop
        debug(f"Training model for fold {i+1}...")
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
        debug(f"Model trained successfully for fold {i+1}.")
            
        # Validation
        debug(f"Validating fold {i+1}...")
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
            oo_losses.append(val_loss.item())
            
        print(f"LOOCV Fold {i+1}/{num_samples} - Validation MSE: {val_loss.item():.4f}")
        debug(f"Finished LOOCV fold {i+1}/{num_samples} with MSE={val_loss.item():.4f}.")
        
    average_mse = np.mean(oo_losses)
    print(f"\nAverage System Cross-Talk Validation MSE: {average_mse:.4f}\n")
    debug(f"Model trained successfully. Average LOOCV MSE={average_mse:.4f}.")
    return model, X, {
        "num_samples": num_samples,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "fold_losses": oo_losses,
        "average_mse": average_mse,
        "feature_names": feature_names,
        "target_names": target_names,
    }


# ==========================================
# 4. SHAP EXPLAINABILITY INTEGRATION
# ==========================================
def compute_shap_explainability(model, X_tensor):
    debug("Starting compute_shap_explainability()...")
    print("--- Initiating SHAP Explainability Layer ---")
    debug("Setting model to eval mode for SHAP...")
    model.eval()
    
    # Convert back to numpy for SHAP framework compatibility
    debug("Converting SHAP background tensor to numpy...")
    X_background = X_tensor.numpy()
    debug(f"SHAP background shape: {X_background.shape}.")
    
    # Wrapper function for PyTorch to map numpy arrays through the torch model
    def model_predict_numpy(data_numpy):
        data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
        with torch.no_grad():
            return model(data_tensor).numpy()
            
    # Initialize SHAP Explainer using KernelSHAP (ideal for model-agnostic omics ranking)
    debug("Initializing SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(model_predict_numpy, X_background)
    debug("SHAP KernelExplainer initialized.")
    debug("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_background)
    debug("SHAP calculations finished.")
    
    print("SHAP values successfully computed.")
    if isinstance(shap_values, list):
        num_targets = len(shap_values)
        debug(f"SHAP returned list with {num_targets} target arrays.")
    elif getattr(shap_values, "ndim", 0) == 3:
        num_targets = shap_values.shape[-1]
        debug(f"SHAP returned ndarray with shape {shap_values.shape}.")
    else:
        num_targets = 1
        debug(f"SHAP returned single-output values with type {type(shap_values).__name__}.")

    print(f"Generated feature importance matrices for all {num_targets} target pathways.")
    print("Ready to plot top driving liver hepatic anchors.")
    debug("compute_shap_explainability() complete.")
    return shap_values


# ==========================================
# 5. LOCAL DESKTOP UI
# ==========================================
def calculate_global_shap_importance(shap_values):
    debug("Calculating global SHAP feature importance...")
    if isinstance(shap_values, list):
        values = np.stack(shap_values, axis=-1)
    else:
        values = np.asarray(shap_values)

    if values.ndim == 3:
        importance = np.abs(values).mean(axis=(0, 2))
        debug(f"Global SHAP importance calculated from 3D values: {importance.shape}.")
        return importance
    if values.ndim == 2:
        importance = np.abs(values).mean(axis=0)
        debug(f"Global SHAP importance calculated from 2D values: {importance.shape}.")
        return importance

    values = np.squeeze(values)
    if values.ndim == 2:
        importance = np.abs(values).mean(axis=0)
        debug(f"Global SHAP importance calculated after squeeze: {importance.shape}.")
        return importance

    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


def launch_dashboard():
    debug("launch_dashboard() called.")
    ui_bg = "#f0f0f0"
    panel_bg = "#ffffff"
    text_fg = "#17212b"
    muted_fg = "#4d5b69"
    accent = "#2f6f73"
    results_queue = queue.Queue()
    debug("Thread-safe results queue initialized.")

    debug("Initializing Tkinter master root...")
    root = tk.Tk()
    debug("Tkinter master root initialized.")
    root.title("Liver-Brain Cross-Talk Dashboard")
    root.geometry("1120x720")
    root.minsize(920, 620)
    root.configure(bg=ui_bg)
    debug("Configured Tkinter root window style and background.")

    debug("Configuring Tkinter grid layout...")
    root.grid_columnconfigure(0, minsize=360)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)
    debug("Tkinter grid layout configured.")

    debug("Creating dashboard panel...")
    dashboard = tk.Frame(
        root,
        bg=panel_bg,
        padx=18,
        pady=18,
        highlightbackground="#d8dee7",
        highlightthickness=1,
    )
    dashboard.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)
    debug("Dashboard panel created.")

    debug("Creating plot panel...")
    plot_panel = tk.Frame(
        root,
        bg=panel_bg,
        padx=18,
        pady=18,
        highlightbackground="#d8dee7",
        highlightthickness=1,
    )
    plot_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
    plot_panel.grid_columnconfigure(0, weight=1)
    plot_panel.grid_rowconfigure(0, weight=1)
    debug("Plot panel created.")

    def add_label(parent, text, font_size=12, bold=False, color=text_fg, pady=2, wraplength=310):
        weight = "bold" if bold else "normal"
        label = tk.Label(
            parent,
            text=text,
            bg=panel_bg,
            fg=color,
            font=("Helvetica", font_size, weight),
            anchor="w",
            justify="left",
            wraplength=wraplength,
        )
        label.pack(anchor="w", fill="x", pady=pady)
        return label

    add_label(dashboard, "Pipeline Running", font_size=16, bold=True, pady=(0, 2))
    tk.Frame(dashboard, bg="#d8dee7", height=1).pack(fill="x", pady=(8, 12))
    status_label = add_label(
        dashboard,
        "Preparing neural network training and SHAP calculations...",
        font_size=12,
        color=muted_fg,
        wraplength=310,
    )
    add_label(
        plot_panel,
        "The dashboard will populate when the background analysis finishes.",
        font_size=15,
        bold=True,
        color=text_fg,
        pady=(0, 8),
        wraplength=620,
    )
    add_label(
        plot_panel,
        "Tkinter is active now; model training and SHAP are running outside the UI event loop.",
        font_size=12,
        color=muted_fg,
        wraplength=620,
    )
    debug("Initial loading UI painted.")

    active_figures = []
    image_refs = []
    pipeline_finished = {"done": False}
    poll_state = {"count": 0}

    def force_window_paint():
        debug("Forcing macOS/Tk first-paint cycle...")
        try:
            root.update_idletasks()
            root.update()
            root.lift()
            root.focus_force()
            root.after(50, root.update_idletasks)
            debug("Forced macOS/Tk first-paint cycle completed.")
        except Exception:
            debug("ERROR: Forced first-paint cycle failed. Printing traceback...")
            traceback.print_exc()

    def clear_frame(frame):
        debug(f"Clearing Tkinter frame {frame}.")
        for child in frame.winfo_children():
            child.destroy()

    def display_png_plot(parent, png_path):
        debug(f"Displaying PNG plot from {png_path}...")
        image = tk.PhotoImage(file=png_path)
        image_refs.append(image)
        image_label = tk.Label(parent, image=image, bg=panel_bg, bd=0, highlightthickness=0)
        image_label.grid(row=0, column=0, sticky="nsew")
        debug("PNG plot display completed.")
        return image_label

    def run_pipeline():
        debug("Background run_pipeline thread started.")
        try:
            debug("Background thread: starting training pipeline...")
            trained_model, final_data_tensor, training_metrics = train_and_evaluate()
            debug("Background thread: training pipeline completed.")
            debug("Background thread: starting SHAP pipeline...")
            final_shap_values = compute_shap_explainability(trained_model, final_data_tensor)
            debug("Background thread: SHAP pipeline completed.")
            results_queue.put(("success", training_metrics, final_shap_values))
            debug("Background thread: success result posted to queue.")
        except Exception as error:
            debug("ERROR: Background pipeline failed. Printing traceback...")
            traceback.print_exc()
            results_queue.put(("error", error))
            debug("Background thread: error result posted to queue.")

    def start_pipeline():
        debug("Starting background pipeline thread with threading.Thread(target=run_pipeline, daemon=True).start()...")
        worker = threading.Thread(target=run_pipeline, daemon=True)
        worker.start()
        debug(f"Background pipeline thread started: name={worker.name}, alive={worker.is_alive()}.")

    def animate_status(step=0):
        if pipeline_finished["done"]:
            debug("Status animation stopped because pipeline finished.")
            return

        phases = [
            "Training LOOCV folds",
            "Optimizing PyTorch MLP",
            "Calculating SHAP values",
            "Keeping Tkinter responsive",
        ]
        dots = "." * ((step % 3) + 1)
        status_label.configure(text=f"{phases[step % len(phases)]}{dots}")
        root.update_idletasks()
        root.after(500, lambda: animate_status(step + 1))

    def render_results(metrics, shap_values):
        debug("Rendering completed analysis results in Tkinter main thread...")
        pipeline_finished["done"] = True
        clear_frame(dashboard)
        clear_frame(plot_panel)

        add_label(dashboard, "Technical Metadata", font_size=16, bold=True, pady=(0, 2))
        tk.Frame(dashboard, bg="#d8dee7", height=1).pack(fill="x", pady=(8, 12))

        metadata_lines = [
            f"Input feature count: {metrics['input_dim']}",
            f"Output target count: {metrics['output_dim']}",
            f"Samples: {metrics['num_samples']}",
            f"Average Validation MSE: {metrics['average_mse']:.4f}",
        ]
        for line in metadata_lines:
            add_label(dashboard, line, font_size=12, color=muted_fg)

        add_label(dashboard, "LOOCV Fold MSE", font_size=14, bold=True, pady=(18, 6))
        for fold_index, fold_mse in enumerate(metrics["fold_losses"], start=1):
            add_label(dashboard, f"Fold {fold_index}: {fold_mse:.4f}", font_size=11, color=muted_fg, pady=1)

        add_label(dashboard, "Liver miRNA Features", font_size=14, bold=True, pady=(18, 6))
        for feature_name in metrics["feature_names"]:
            add_label(dashboard, feature_name, font_size=10, color=muted_fg, pady=1)

        try:
            debug("Preparing SHAP importance data for plot...")
            importance = calculate_global_shap_importance(shap_values)
            feature_names = np.asarray(metrics["feature_names"])
            order = np.argsort(importance)
            debug(f"Prepared SHAP importance plot data with {len(importance)} features.")

            debug("Creating Matplotlib figure and axes...")
            fig, ax = plt.subplots(figsize=(8, 5), dpi=110, facecolor=panel_bg)
            active_figures.append(fig)
            ax.set_facecolor(panel_bg)
            ax.barh(feature_names[order], importance[order], color=accent)
            ax.set_title("Global SHAP Feature Importance")
            ax.set_xlabel("Mean absolute SHAP value")
            ax.set_ylabel("Liver miRNA")
            ax.tick_params(colors=text_fg)
            ax.title.set_color(text_fg)
            ax.xaxis.label.set_color(text_fg)
            ax.yaxis.label.set_color(text_fg)
            ax.grid(axis="x", linestyle="--", color="#a9b6c4", alpha=0.45)
            for spine in ax.spines.values():
                spine.set_color("#c8d1dc")
            fig.tight_layout()
            debug("Matplotlib figure created successfully.")

            fallback_png_path = os.path.abspath("shap_feature_importance.png")
            debug(f"Saving Matplotlib fallback PNG to {fallback_png_path}...")
            fig.savefig(fallback_png_path, dpi=140, bbox_inches="tight", facecolor=panel_bg)
            debug("Matplotlib fallback PNG saved.")

            debug("Embedding matplotlib canvas...")
            canvas = FigureCanvasTkAgg(fig, master=plot_panel)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.configure(bg=panel_bg, highlightthickness=0)
            canvas_widget.grid(row=0, column=0, sticky="nsew")
            root.update_idletasks()
            debug("Calling canvas.draw()...")
            canvas.draw()
            debug("canvas.draw() completed.")
            root.after(100, canvas.draw_idle)
            debug("Scheduled canvas.draw_idle() with root.after().")
            debug("Switching visible plot layer to PNG fallback to avoid silent macOS TkAgg black-canvas rendering.")
            canvas_widget.grid_remove()
            display_png_plot(plot_panel, fallback_png_path)
            force_window_paint()
        except Exception as error:
            debug("ERROR: Matplotlib embedding failed. Printing full traceback...")
            traceback.print_exc()
            clear_frame(plot_panel)
            try:
                debug("Attempting PNG fallback display after Matplotlib embedding failure...")
                fallback_png_path = os.path.abspath("shap_feature_importance.png")
                display_png_plot(plot_panel, fallback_png_path)
            except Exception:
                debug("ERROR: PNG fallback display failed. Printing full traceback...")
                traceback.print_exc()
                add_label(
                    plot_panel,
                    "Matplotlib plot rendering failed. Check the terminal DEBUG traceback.",
                    font_size=14,
                    bold=True,
                    color="#9b1c1c",
                    wraplength=620,
                )
                add_label(
                    plot_panel,
                    str(error),
                    font_size=11,
                    color="#9b1c1c",
                    wraplength=620,
                )
            root.update_idletasks()
        debug("render_results() complete.")

    def render_error(error):
        debug("Rendering pipeline error in Tkinter main thread...")
        pipeline_finished["done"] = True
        clear_frame(dashboard)
        clear_frame(plot_panel)
        add_label(dashboard, "Pipeline Error", font_size=16, bold=True, color="#9b1c1c")
        add_label(dashboard, str(error), font_size=11, color="#9b1c1c", wraplength=310)
        add_label(plot_panel, "The analysis did not complete.", font_size=15, bold=True, color=text_fg, wraplength=620)
        root.update_idletasks()
        debug("render_error() complete.")

    def check_completion():
        poll_state["count"] += 1
        try:
            message = results_queue.get_nowait()
        except queue.Empty:
            if poll_state["count"] == 1 or poll_state["count"] % 10 == 0:
                debug(f"check_completion(): no result yet after {poll_state['count']} polls.")
            root.after(100, check_completion)
            return

        debug(f"check_completion(): received queue message type={message[0]}.")
        if message[0] == "success":
            _, metrics, shap_values = message
            render_results(metrics, shap_values)
        else:
            _, error = message
            render_error(error)

    def on_close():
        debug("Window close requested. Cleaning up Matplotlib and Tkinter...")
        for fig in active_figures:
            plt.close(fig)
        plt.close("all")
        root.quit()
        root.destroy()
        debug("Tkinter shutdown complete.")

    root.protocol("WM_DELETE_WINDOW", on_close)
    debug("Calling root.update_idletasks() before mainloop...")
    root.update_idletasks()
    debug("Initial root.update_idletasks() complete.")
    root.after(25, force_window_paint)
    root.after(100, start_pipeline)
    root.after(100, check_completion)
    root.after(200, animate_status)
    debug("Scheduled start_pipeline(), check_completion(), and animate_status() with root.after().")
    debug("Entering Tkinter mainloop now.")
    root.mainloop()
    debug("Tkinter mainloop exited.")


def save_shap_importance_plot(metrics, shap_values, output_path="shap_feature_importance.png"):
    debug("Generating browser-safe SHAP PNG plot...")
    importance = calculate_global_shap_importance(shap_values)
    feature_names = np.asarray(metrics["feature_names"])
    order = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.barh(feature_names[order], importance[order], color="#2f6f73")
    ax.set_title("Global SHAP Feature Importance")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Liver miRNA")
    ax.grid(axis="x", linestyle="--", color="#a9b6c4", alpha=0.45)
    for spine in ax.spines.values():
        spine.set_color("#c8d1dc")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    debug(f"Saved SHAP importance plot to {os.path.abspath(output_path)}.")
    return output_path


def write_browser_dashboard(metrics, plot_path, output_path="dashboard.html"):
    debug("Writing browser dashboard HTML...")
    escaped_features = [html.escape(str(feature)) for feature in metrics["feature_names"]]
    escaped_targets = [html.escape(str(target)) for target in metrics["target_names"]]
    fold_rows = "\n".join(
        f"<tr><td>Fold {index}</td><td>{loss:.4f}</td></tr>"
        for index, loss in enumerate(metrics["fold_losses"], start=1)
    )
    feature_items = "\n".join(f"<li>{feature}</li>" for feature in escaped_features)
    target_items = "\n".join(f"<li>{target}</li>" for target in escaped_targets)
    plot_src = html.escape(Path(plot_path).name)

    dashboard_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Liver-Brain Cross-Talk Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #eef2f5;
      --panel: #ffffff;
      --text: #17212b;
      --muted: #4d5b69;
      --line: #d8dee7;
      --accent: #2f6f73;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(320px, 380px) minmax(0, 1fr);
      gap: 16px;
      min-height: 100vh;
      padding: 18px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    h1, h2 {{ margin: 0; }}
    h1 {{ font-size: 22px; }}
    h2 {{ font-size: 16px; margin-top: 20px; }}
    .meta {{
      display: grid;
      gap: 8px;
      margin-top: 14px;
      color: var(--muted);
    }}
    .value {{
      color: var(--text);
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 14px;
    }}
    td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 0;
      color: var(--muted);
    }}
    td:last-child {{
      color: var(--text);
      font-variant-numeric: tabular-nums;
      text-align: right;
    }}
    ul {{
      margin: 8px 0 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .plot {{
      display: grid;
      align-content: start;
      min-height: calc(100vh - 36px);
    }}
    .plot img {{
      width: 100%;
      max-width: 980px;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      margin-top: 16px;
    }}
    .note {{
      color: var(--muted);
      margin-top: 8px;
      line-height: 1.45;
    }}
    @media (max-width: 820px) {{
      main {{ grid-template-columns: 1fr; }}
      .plot {{ min-height: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Liver-Brain Cross-Talk Dashboard</h1>
      <div class="meta">
        <div>Input feature count: <span class="value">{metrics["input_dim"]}</span></div>
        <div>Output target count: <span class="value">{metrics["output_dim"]}</span></div>
        <div>Samples: <span class="value">{metrics["num_samples"]}</span></div>
        <div>Average Validation MSE: <span class="value">{metrics["average_mse"]:.4f}</span></div>
      </div>

      <h2>LOOCV Fold MSE</h2>
      <table>
        <tbody>
          {fold_rows}
        </tbody>
      </table>

      <h2>Liver miRNA Features</h2>
      <ul>{feature_items}</ul>

      <h2>Brain Target miRNAs</h2>
      <ul>{target_items}</ul>
    </section>
    <section class="plot">
      <h1>Global SHAP Feature Importance</h1>
      <p class="note">Mean absolute SHAP values for selected liver miRNAs, averaged across samples and output targets.</p>
      <img src="{plot_src}" alt="Global SHAP feature importance bar chart">
    </section>
  </main>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(dashboard_html)

    debug(f"Wrote browser dashboard to {os.path.abspath(output_path)}.")
    return output_path


def launch_browser_dashboard():
    debug("Starting browser dashboard pipeline...")
    trained_model, final_data_tensor, training_metrics = train_and_evaluate()
    final_shap_values = compute_shap_explainability(trained_model, final_data_tensor)
    plot_path = save_shap_importance_plot(training_metrics, final_shap_values)
    dashboard_path = write_browser_dashboard(training_metrics, plot_path)
    dashboard_url = Path(dashboard_path).resolve().as_uri()
    debug(f"Opening dashboard in browser: {dashboard_url}")
    webbrowser.open(dashboard_url)
    print(f"\nDashboard written to: {os.path.abspath(dashboard_path)}")
    print(f"SHAP plot written to: {os.path.abspath(plot_path)}")


# ==========================================
# 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    debug("Script entry point reached. Launching browser dashboard...")
    launch_browser_dashboard()
    debug("Script execution finished.")
