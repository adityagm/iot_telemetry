import numpy as np
import plotly.graph_objects as go

def build_interactive_risk_chart(
    df, 
    device_id="DEV-CONTAINER-01", 
    threshold_low=0.30, 
    threshold_high=0.75
):
    """
    Generates interactive Plotly failure trajectory chart with ISO 13374 operational risk bands.
    """
    fig = go.Figure()
    max_time = df['time'].max()

    # Risk Zone Shading
    fig.add_shape(type="rect", x0=0, x1=max_time, y0=0.0, y1=threshold_low,
                  fillcolor="rgba(76, 175, 80, 0.15)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=max_time, y0=threshold_low, y1=threshold_high,
                  fillcolor="rgba(255, 193, 7, 0.15)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=max_time, y0=threshold_high, y1=1.0,
                  fillcolor="rgba(244, 67, 54, 0.15)", line_width=0, layer="below")

    # Threshold Lines
    fig.add_hline(y=threshold_low, line_dash="dash", line_color="#795548", line_width=1.5)
    fig.add_hline(y=threshold_high, line_dash="dash", line_color="#b71c1c", line_width=1.5)

    # Plot Line & Hover Tooltip
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['p_fault'], mode='lines',
        name='Failure/Risk Probability', line=dict(color='#0277bd', width=2.5),
        customdata=np.stack((df['p_accel'], df['p_temp'], df['p_batt']), axis=-1),
        hovertemplate=(
            "<b>Time:</b> %{x} Days<br><b>Combined Risk:</b> %{y:.1%}<br><extra>" +
            "<b>Sensors:</b><br>• Battery: %{customdata[2]:.1%}<br>" +
            "• Temp: %{customdata[1]:.1%}<br>• Accel: %{customdata[0]:.1%}</extra>"
        )
    ))

    fig.update_layout(
        title=f"<b>Device Failure Probability vs. Time with Risk Zones</b><br><sup>Device: {device_id}</sup>",
        xaxis=dict(title="Time (Days)", range=[0, max_time]),
        yaxis=dict(title="Probability", range=[0.0, 1.0]),
        template="plotly_white"
    )
    return fig