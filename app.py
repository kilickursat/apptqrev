import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import gaussian_kde
import networkx as nx
from scipy.interpolate import griddata

# Constants for the second part
P_max = 132  # Maximum power in kW
nu = 0.7  # Efficiency coefficient
n_max = 20.1  # Maximum rpm
n_min = 10  # Minimum rpm for data filtering
x_axis_max = 20.3  # Extended x-axis for visualization
M_cont_value = 44  # Known continuous torque in kNm
M_max_Vg1 = 54  # Maximum torque for Vg1 in kNm

# Function to calculate M max Vg2
def M_max_Vg2(rpm):
    return np.minimum(M_max_Vg1, (P_max * 60 * nu) / (2 * np.pi * rpm))

# Helper functions from the first part
def calculate_torque(working_pressure, current_speed, torque_constant=0.14376997, n1=25.7):
    if current_speed < n1:
        return working_pressure * torque_constant
    else:
        return (n1 / current_speed) * torque_constant * working_pressure

def calculate_penetration_rate(speed, rpm):
    return round(speed / rpm if rpm != 0 else 0, 4)

def calculate_advance_rate_and_stats(df):
    distance_column = 'Weg VTP [mm]'
    time_column = 'Relative time'

    if len(df) > 1:
        weg = round(df[distance_column].max() - df[distance_column].min(), 2)
        zeit = round(df[time_column].max() - df[time_column].min(), 2)
    else:
        weg = df[distance_column].iloc[0]
        zeit = df[time_column].iloc[0]

    zeit = zeit * (0.000001 / 60)
    average_speed = round(weg / zeit, 2) if zeit != 0 else 0

    result = {
        "Total Distance (mm)": weg,
        "Total Time (min)": zeit,
        "Average Speed (mm/min)": average_speed
    }

    return result, average_speed

# Visualization functions from the first part
def visualize_machine_features(df):
    features = ['Working pressure [bar]', 'Penetration_Rate', 'Calculated torque [kNm]', 'Average Speed (mm/min)', 'Revolution [rpm]', 'Thrust force [kN]']
    fig = make_subplots(rows=len(features), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=features)

    for i, feature in enumerate(features, start=1):
        if feature == 'Average Speed (mm/min)':
            y_values = [df['Average Speed (mm/min)'].iloc[0]] * len(df)
        else:
            y_values = df[feature]
        fig.add_trace(go.Scatter(x=df['Relative time'], y=y_values, name=feature), row=i, col=1)
        fig.update_yaxes(title_text=feature, row=i, col=1)

    fig.update_layout(height=1200, title_text="Features vs Time", showlegend=False)
    fig.update_xaxes(title_text="Time", row=len(features), col=1)
    return fig

def create_correlation_heatmap(df):
    features = ['Revolution [rpm]', 'Thrust force [kN]', 'Chainage', 'Calculated torque [kNm]', 'Penetration_Rate', 'Working pressure [bar]']
    corr_matrix = df[features].corr()
    fig = px.imshow(corr_matrix, 
                    labels=dict(color="Correlation"),
                    x=features,
                    y=features,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)
    fig.update_layout(title='Correlation Heatmap of Selected Parameters')
    return fig

def perform_ols_regression(df, x_col, y_col):
    X = sm.add_constant(df[x_col])
    model = sm.OLS(df[y_col], X).fit()
    
    fig = px.scatter(df, x=x_col, y=y_col, trendline="ols")
    fig.update_layout(title=f'OLS Regression: {y_col} vs {x_col}',
                      xaxis_title=x_col,
                      yaxis_title=y_col)
    
    return model, fig

def create_polar_plot(df):
    pressure_column = 'Working pressure [bar]'
    time_normalized = np.linspace(0, 360, len(df))
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df[pressure_column],
        theta=time_normalized,
        mode='markers',
        marker=dict(color='blue', size=5),
        name='Pressure'
    ))

    fig.update_layout(
        title='Pressure Distribution Over Time',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, df[pressure_column].max() * 1.1],
            ),
        ),
    )
    return fig

def create_3d_scatter_plot(df):
    fig = px.scatter_3d(df, x='Working pressure [bar]', y='Revolution [rpm]', z='Calculated torque [kNm]',
                        color='Thrust force [kN]',
                        hover_data=['Working pressure [bar]', 'SR Position [Grad]'],
                        labels={'Working pressure [bar]': 'Working Pressure',
                                'Revolution [rpm]': 'Revolution [rpm]',
                                'Calculated torque [kNm]': 'Calculated Torque [kNm]',
                                'Thrust force [kN]': 'Thrust force [kN]'},
                        title='3D Visualization of Key Parameters')
    return fig

def create_density_heatmap(df):
    x = df['Working pressure [bar]']
    y = df['Penetration_Rate']

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    fig = go.Figure(data=go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=3, color=z, colorscale='Viridis', showscale=True)
    ))

    fig.update_layout(
        title='Density Heatmap: Working Pressure vs Penetration Rate',
        xaxis_title='Working Pressure',
        yaxis_title='Penetration Rate',
        coloraxis_colorbar=dict(title='Density')
    )
    return fig

# Function for the second part
def create_torque_rpm_plot(df):
    # Calculate the intersection points
    elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * M_max_Vg1)
    elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * M_cont_value)

    # Generate rpm values for the continuous curves
    rpm_curve = np.linspace(0.1, n_max, 1000)  # Avoid division by zero

    # Create the plot
    fig = go.Figure()

    # Plot torque curves
    fig.add_trace(go.Scatter(
        x=rpm_curve[rpm_curve <= elbow_rpm_cont],
        y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_cont], M_cont_value),
        mode='lines', name='M cont Max [kNm]', line=dict(color='green', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=rpm_curve[rpm_curve <= elbow_rpm_max],
        y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], M_max_Vg1),
        mode='lines', name='M max Vg1 [kNm]', line=dict(color='red', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=rpm_curve[rpm_curve <= n_max],
        y=M_max_Vg2(rpm_curve[rpm_curve <= n_max]),
        mode='lines', name='M max Vg2 [kNm]', line=dict(color='red', width=2, dash='dash')
    ))

    # Add vertical lines at the elbow points
    fig.add_trace(go.Scatter(
        x=[elbow_rpm_max, elbow_rpm_max], y=[0, M_max_Vg1],
        mode='lines', name='Elbow Max', line=dict(color='purple', width=2, dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=[elbow_rpm_cont, elbow_rpm_cont], y=[0, M_cont_value],
        mode='lines', name='Elbow Cont', line=dict(color='orange', width=2, dash='dot')
    ))

    # Add a truncated vertical line at n_max
    fig.add_trace(go.Scatter(
        x=[n_max, n_max], y=[0, M_cont_value],
        mode='lines', name='Max RPM', line=dict(color='black', width=2, dash='dash')
    ))

    # Plot calculated torque vs RPM, differentiating between normal and anomaly points
    normal_data = df[df['Working pressure [bar]'] < 250]
    anomaly_data = df[df['Working pressure [bar]'] >= 250]

    fig.add_trace(go.Scatter(
        x=normal_data['Revolution [rpm]'], y=normal_data['Calculated torque [kNm]'],
        mode='markers', name='Normal Data',
        marker=dict(
            size=8,
            color=normal_data['Calculated torque [kNm]'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Calculated Torque [kNm]')
        )
    ))

    fig.add_trace(go.Scatter(
        x=anomaly_data['Revolution [rpm]'], y=anomaly_data['Calculated torque [kNm]'],
        mode='markers', name='Anomaly (Pressure ≥ 250 bar)',
        marker=dict(color='red', size=10, symbol='x')
    ))

    # Update layout
    fig.update_layout(
        title='AVN800 DA975 - Hard Rock Test - 132kW (with Anomaly Detection)',
        xaxis_title='Drehzahl / speed / vitesse / revolutiones [1/min]',
        yaxis_title='Drehmoment / torque / couple / par de giro [kNm]',
        xaxis=dict(range=[0, x_axis_max]),
        yaxis=dict(range=[0, max(60, df['Calculated torque [kNm]'].max() * 1.1)]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Add annotations
    fig.add_annotation(x=elbow_rpm_max * 0.5, y=M_max_Vg1 * 1.05,
                       text=f'M max (max.): {M_max_Vg1} kNm',
                       showarrow=False, font=dict(color='red'))
    fig.add_annotation(x=elbow_rpm_cont * 0.5, y=M_cont_value * 0.95,
                       text=f'M cont (max.): {M_cont_value} kNm',
                       showarrow=False, font=dict(color='green'))
    fig.add_annotation(x=elbow_rpm_max, y=0,
                       text=f'{elbow_rpm_max:.2f}',
                       showarrow=False, font=dict(color='purple'))
    fig.add_annotation(x=elbow_rpm_cont, y=0,
                       text=f'{elbow_rpm_cont:.2f}',
                       showarrow=False, font=dict(color='orange'))
    fig.add_annotation(x=n_max, y=M_cont_value,
                       text=f'Max RPM: {n_max}',
                       showarrow=False, font=dict(color='black'), textangle=-90)

    return fig

def main():
    st.set_page_config(page_title="Data Analysis App", layout="wide")
    st.title("Data Analysis App")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, sep=';', decimal=',')

        # Rename columns for clarity
        df = df.rename(columns={
            'AzV.V13_SR_ArbDr_Z | DB    60.DBD    26': 'Working pressure [bar]',
            'AzV.V13_SR_Drehz_nach_Abgl_Z | DB    60.DBD    30': 'Revolution [rpm]',
            'AzV.V13_SR_Vorschub_Z | DB    60.DBD    32': 'Thrust force [kN]',
            'AzV.V13_SR_Weg | DB    60.DBD    18': 'Weg VTP [mm]',
            'AzV.V13_SR_Pos_Grad | DB    60.DBD   236': 'SR Position [Grad]',
            'Chainage_db': 'Chainage',
            'Relative Time_db': 'Relative time'
        })

        # Clean numeric columns
        numeric_columns = ['Working pressure [bar]', 'Revolution [rpm]', 'Thrust force [kN]', 'Chainage', 'Relative time', 'Weg VTP [mm]', 'SR Position [Grad]']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors=''coerce')

        # Remove rows with NaN values
        df = df.dropna(subset=numeric_columns)

        # Calculate new columns
        df['Calculated torque [kNm]'] = df.apply(lambda row: calculate_torque(row['Working pressure [bar]'], row['Revolution [rpm]']), axis=1)
        results, average_speed = calculate_advance_rate_and_stats(df)
        df['Average Speed (mm/min)'] = average_speed
        df['Penetration_Rate'] = df.apply(lambda row: calculate_penetration_rate(row['Average Speed (mm/min)'], row['Revolution [rpm]']), axis=1)

        # Display basic statistics
        st.subheader("Basic Statistics")
        st.write(results)

        # Display data overview
        st.subheader("Data Overview")
        st.write(df.describe())

        # Visualizations
        st.subheader("Visualizations")

        # Machine Features Plot
        st.plotly_chart(visualize_machine_features(df))

        # Correlation Heatmap
        st.plotly_chart(create_correlation_heatmap(df))

        # OLS Regression
        st.subheader("OLS Regression")
        x_col = st.selectbox("Select X-axis variable", df.columns)
        y_col = st.selectbox("Select Y-axis variable", df.columns)
        model, fig = perform_ols_regression(df, x_col, y_col)
        st.plotly_chart(fig)
        st.write(model.summary())

        # Polar Plot
        st.plotly_chart(create_polar_plot(df))

        # 3D Scatter Plot
        st.plotly_chart(create_3d_scatter_plot(df))

        # Density Heatmap
        st.plotly_chart(create_density_heatmap(df))

        # Torque-RPM Plot
        st.plotly_chart(create_torque_rpm_plot(df))

if __name__ == "__main__":
    main()
