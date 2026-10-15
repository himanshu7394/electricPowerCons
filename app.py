import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Power Forecast System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Tech-Analytic Dark Theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --bg-primary: #121212;
        --bg-secondary: #1E1E1E;
        --bg-tertiary: #252525;
        --accent-teal: #00D9FF;
        --accent-blue: #3B82F6;
        --text-primary: #E5E7EB;
        --text-secondary: #9CA3AF;
        --shadow-soft: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    /* Main container styling */
    .main .block-container {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        padding: 2rem;
        border-radius: 12px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid #333;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: var(--bg-secondary);
    }

    /* Input containers with neumorphism-lite */
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: var(--bg-tertiary);
        border-radius: 8px;
        box-shadow: var(--shadow-soft);
    }

    /* Metric card styling */
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, var(--accent-teal), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
    }

    [data-testid="stMetricDelta"] {
        color: var(--accent-teal);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: var(--shadow-soft);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 217, 255, 0.3);
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Loading spinner animation */
    @keyframes pulse-glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        }
        50% {
            box-shadow: 0 0 40px rgba(0, 217, 255, 0.6);
        }
    }

    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        animation: pulse-glow 2s infinite;
    }

    /* Plotly chart container */
    .plotly-chart {
        background-color: var(--bg-tertiary);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: var(--shadow-soft);
    }

    /* Label styling */
    label {
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* Info box */
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 3px solid var(--accent-blue);
        border-radius: 4px;
    }

    /* Success box */
    .stSuccess {
        background-color: rgba(0, 217, 255, 0.1);
        border-left: 3px solid var(--accent-teal);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Custom Plotly template
custom_template = {
    'layout': {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {
            'color': '#E5E7EB',
            'family': 'Segoe UI, sans-serif'
        },
        'xaxis': {
            'gridcolor': '#333333',
            'linecolor': '#444444',
            'zerolinecolor': '#444444'
        },
        'yaxis': {
            'gridcolor': '#333333',
            'linecolor': '#444444',
            'zerolinecolor': '#444444'
        },
        'legend': {
            'bgcolor': 'rgba(30, 30, 30, 0.8)',
            'bordercolor': '#444444'
        }
    }
}

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load('lightgbm_time_series_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'lightgbm_time_series_model.pkl' not found. Please ensure the model is in the project directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main application
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
            ⚡ Power Forecast System
        </h1>
        <p style='color: #9CA3AF; font-size: 1.1rem;'>
            Advanced Time-Series Prediction Engine
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # Sidebar - Input Section
    st.sidebar.markdown("""
    <h2 style='color: #00D9FF; margin-bottom: 1rem;'>Configuration</h2>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Temporal Features")
        
        # Temporal inputs
        col1, col2 = st.columns(2)
        with col1:
            hour = st.number_input("Hour", min_value=0, max_value=23, value=12, help="Hour of the day (0-23)")
            day = st.number_input("Day", min_value=1, max_value=31, value=16, help="Day of the month")
        
        with col2:
            day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=4, help="0=Monday, 6=Sunday")
            month = st.number_input("Month", min_value=1, max_value=12, value=10, help="Month of the year")
        
        year = st.number_input("Year", min_value=2006, max_value=2030, value=2026, help="Year")

        st.markdown("---")
        st.markdown("### Power Consumption Features")

        # Power inputs
        current_power = st.number_input(
            "Current Power (kW)",
            min_value=0.0,
            max_value=10.0,
            value=1.2,
            step=0.01,
            help="Current global active power in kilowatts"
        )
        
        lag_1 = st.number_input(
            "Lag 1 (kW)",
            min_value=0.0,
            max_value=10.0,
            value=1.1,
            step=0.01,
            help="Power consumption 1 hour ago"
        )
        
        lag_2 = st.number_input(
            "Lag 2 (kW)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Power consumption 2 hours ago"
        )

        st.markdown("---")
        st.markdown("### Cluster Configuration")

        cluster_id = st.selectbox(
            "Cluster ID",
            options=[0, 1, 2],
            index=0,
            help="Consumption pattern cluster (derived from sub-metering data)"
        )

        st.markdown("---")
        
        # Execute button
        execute_button = st.button(
            "⚡ Execute Forecast",
            type="primary",
            use_container_width=True
        )

    # Main content area
    if execute_button:
        # Show loading animation
        with st.container():
            st.markdown("""
            <div class='loading-container'>
                <div style='text-align: center;'>
                    <div style='font-size: 1.2rem; color: #00D9FF; margin-bottom: 0.5rem;'>
                        Processing Forecast...
                    </div>
                    <div style='color: #9CA3AF; font-size: 0.9rem;'>
                        Running LightGBM inference engine
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Prepare input features
        input_features = pd.DataFrame({
            'hour': [hour],
            'day': [day],
            'day_of_week': [day_of_week],
            'month': [month],
            'year': [year],
            'Cluster_id': [cluster_id],
            'Current_Power': [current_power],
            'Lag_1': [lag_1],
            'Lag_2': [lag_2]
        })

        # Ensure Cluster_id is categorical as expected by the model
        input_features['Cluster_id'] = input_features['Cluster_id'].astype('category')

        try:
            # Make prediction
            prediction = model.predict(input_features)[0]
            
            # Clear loading and show results
            st.empty()

            # Results section
            st.markdown("---")
            st.markdown("""
            <h2 style='color: #00D9FF; margin-bottom: 1rem;'>Forecast Results</h2>
            """, unsafe_allow_html=True)

            # Metric card
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric(
                    label="Predicted Next Hour Power",
                    value=f"{prediction:.4f} kW",
                    delta=f"{((prediction - current_power) / current_power * 100):.2f}% vs Current"
                )

            st.markdown("---")

            # Visualization section
            st.markdown("""
            <h2 style='color: #00D9FF; margin-bottom: 1rem;'>Forecast Visualization</h2>
            """, unsafe_allow_html=True)

            # Create time series chart
            historical_hours = [lag_2, lag_1, current_power]
            historical_labels = ['T-2', 'T-1', 'Current (T)']
            forecast_value = prediction
            forecast_label = 'Forecast (T+1)'

            fig = go.Figure()

            # Historical data line
            fig.add_trace(go.Scatter(
                x=historical_labels + [forecast_label],
                y=historical_hours + [forecast_value],
                mode='lines+markers',
                name='Power Consumption',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=10, color='#3B82F6'),
                connectgaps=True
            ))

            # Highlight the forecast point
            fig.add_trace(go.Scatter(
                x=[forecast_label],
                y=[forecast_value],
                mode='markers',
                name='Forecast',
                marker=dict(size=15, color='#00D9FF', symbol='diamond'),
                showlegend=False
            ))

            # Update layout with custom template
            fig.update_layout(
                title={
                    'text': 'Power Consumption Forecast',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#E5E7EB'}
                },
                xaxis_title='Time',
                yaxis_title='Power (kW)',
                hovermode='x unified',
                **custom_template['layout']
            )

            # Add grid styling
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Additional insights
            st.markdown("---")
            st.markdown("""
            <h2 style='color: #00D9FF; margin-bottom: 1rem;'>Analysis Insights</h2>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Input Summary**
                - Hour: {hour}:00
                - Day: {day}/{month}/{year}
                - Cluster: {cluster_id}
                - Current Power: {current_power:.4f} kW
                """)

            with col2:
                trend = "↑ Increasing" if prediction > current_power else "↓ Decreasing"
                st.success(f"""
                **Forecast Analysis**
                - Next Hour: {prediction:.4f} kW
                - Trend: {trend}
                - Change: {abs(prediction - current_power):.4f} kW
                """)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please check that all input values are within valid ranges.")

    else:
        # Initial state - show instructions
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h3 style='color: #9CA3AF; margin-bottom: 1rem;'>System Ready</h3>
            <p style='color: #6B7280; font-size: 1.1rem;'>
                Configure parameters in the sidebar and click <strong>Execute Forecast</strong> to generate predictions.
            </p>
            <div style='margin-top: 2rem; padding: 1.5rem; background-color: #1E1E1E; border-radius: 8px; display: inline-block;'>
                <p style='color: #00D9FF; margin: 0; font-size: 0.9rem;'>
                    Model: LightGBM Regressor<br>
                    Features: 9 | Target: Next Hour Power
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
