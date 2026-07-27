"""
Evaluation metrics calculation engine and ReportLab PDF report generator.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from src.config import EVALUATION_REPORT_PDF_PATH
from src.logger import logger


class Evaluator:
    """Evaluates ML model metrics and builds PDF reports."""

    def __init__(self):
        pass

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, num_features: int) -> Dict[str, float]:
        """
        Calculates MAE, MSE, RMSE, R2, Adjusted R2, and MAPE metrics.
        
        Args:
            y_true (np.ndarray): Ground truth target values.
            y_pred (np.ndarray): Model prediction values.
            num_features (int): Number of predictor features p for Adjusted R2.
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics.
        """
        n = len(y_true)
        p = num_features

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Adjusted R2
        adj_r2 = 1 - ((1 - r2) * (n - 1) / max(1, (n - p - 1)))
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100

        return {
            "MAE": round(float(mae), 2),
            "MSE": round(float(mse), 2),
            "RMSE": round(float(rmse), 2),
            "R2_Score": round(float(r2), 4),
            "Adj_R2": round(float(adj_r2), 4),
            "MAPE": round(float(mape), 2)
        }

    def evaluate_cross_validation(self, pipeline: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """Performs K-Fold cross validation and returns mean/std RMSE."""
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
        rmse_scores = -scores
        return {
            "CV_RMSE_Mean": round(float(rmse_scores.mean()), 2),
            "CV_RMSE_Std": round(float(rmse_scores.std()), 2)
        }

    def generate_pdf_report(
        self,
        best_model_name: str,
        leaderboard_df: pd.DataFrame,
        best_metrics: Dict[str, float],
        pdf_path=EVALUATION_REPORT_PDF_PATH
    ) -> str:
        """
        Generates an executive ReportLab PDF evaluation report.
        
        Args:
            best_model_name (str): Best model title.
            leaderboard_df (pd.DataFrame): Leaderboard table.
            best_metrics (Dict[str, float]): Metrics of winning model.
            pdf_path: File output path.
            
        Returns:
            str: Path to generated PDF file.
        """
        try:
            pdf_path = str(pdf_path)
            doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                "TitleStyle",
                parent=styles["Heading1"],
                fontName="Helvetica-Bold",
                fontSize=22,
                textColor=colors.HexColor("#4f46e5"),
                spaceAfter=12
            )

            subtitle_style = ParagraphStyle(
                "SubtitleStyle",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=11,
                textColor=colors.HexColor("#475569"),
                spaceAfter=20
            )

            heading2 = ParagraphStyle(
                "Heading2",
                parent=styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=14,
                textColor=colors.HexColor("#0f172a"),
                spaceBefore=12,
                spaceAfter=8
            )

            story = []
            story.append(Paragraph("King County House Price Prediction - AI Evaluation Report", title_style))
            story.append(Paragraph("Automated Machine Learning Benchmark and Explainable AI Assessment", subtitle_style))
            story.append(Spacer(1, 10))

            # Best Model KPI Section
            story.append(Paragraph("Selected Champion Model", heading2))
            summary_text = (
                f"<b>Model Architecture:</b> {best_model_name}<br/>"
                f"<b>R² Accuracy Score:</b> {best_metrics.get('R2_Score', 0):.4f}<br/>"
                f"<b>Root Mean Squared Error (RMSE):</b> ${best_metrics.get('RMSE', 0):,.2f}<br/>"
                f"<b>Mean Absolute Error (MAE):</b> ${best_metrics.get('MAE', 0):,.2f}<br/>"
                f"<b>Mean Absolute Percentage Error (MAPE):</b> {best_metrics.get('MAPE', 0):.2f}%"
            )
            story.append(Paragraph(summary_text, styles["Normal"]))
            story.append(Spacer(1, 15))

            # Leaderboard Table
            story.append(Paragraph("Model Benchmark Leaderboard", heading2))
            
            table_data = [["Model", "MAE ($)", "RMSE ($)", "R² Score", "Adj R²"]]
            for _, row in leaderboard_df.iterrows():
                table_data.append([
                    str(row["Model"]),
                    f"${row['MAE']:,.0f}",
                    f"${row['RMSE']:,.0f}",
                    f"{row['R2_Score']:.4f}",
                    f"{row.get('Adj_R2', row['R2_Score']):.4f}"
                ])

            table = Table(table_data, colWidths=[150, 90, 90, 80, 80])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]))
            story.append(table)

            doc.build(story)
            logger.info(f"Generated PDF evaluation report at {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return str(pdf_path)
