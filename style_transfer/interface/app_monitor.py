from dataclasses import dataclass
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel



@dataclass
class AppModelMonitor:
    model: TransformerStyleTransferModel

    def RUN(self):
        c1, c2= st.columns((2, 1))
        with c1 :
            st.header("Plot of training loss")
            fig = self._build_log_line_plot()
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.header("BLEU metric")
            st.metric(label="BLEU metric", value=f"{self._build_google_bleu_metric_render()}", delta="12 %")
    
    def _build_log_line_plot(self):
        train_by_loss_dataframe = TransformerStyleTransferModel.retrieve_model_logs()
        x_axis = train_by_loss_dataframe["Step"]
        y_axis = train_by_loss_dataframe["Training Loss"]
        legend = "train/loss"
        fig = go.Figure()
        # Create and style traces
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, name = legend,
                                line=dict(color='orange', width=4)))
        # Edit the layout
        fig.update_layout(title="Model's training performance over steps",
                        xaxis_title='Training steps',
                        yaxis_title='Training loss')
        return fig
    def _build_google_bleu_metric_render(self):
        google_bleu_score_df_test = TransformerStyleTransferModel.retrieve_model_logs(method="Bleu")
        return google_bleu_score_df_test
