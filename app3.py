import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Schedule Risk Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff6b6b;
        padding: 5px;
        border-radius: 3px;
        color: white;
    }
    .risk-medium {
        background-color: #ffd93d;
        padding: 5px;
        border-radius: 3px;
        color: black;
    }
    .risk-low {
        background-color: #6bcf7f;
        padding: 5px;
        border-radius: 3px;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class ScheduleRiskSimulator:
    def __init__(self):
        self.tasks = []
        self.simulation_results = []
        
    def add_task(self, name, optimistic, most_likely, pessimistic):
        """Add a task with three-point estimates"""
        self.tasks.append({
            'name': name,
            'optimistic': optimistic,
            'most_likely': most_likely,
            'pessimistic': pessimistic
        })
    
    def run_monte_carlo_simulation(self, num_simulations=10000):
        """Run Monte Carlo simulation using triangular distribution"""
        total_durations = []
        
        for _ in range(num_simulations):
            total_duration = 0
            for task in self.tasks:
                # Sample from triangular distribution for each task
                duration = np.random.triangular(
                    task['optimistic'],
                    task['most_likely'],
                    task['pessimistic']
                )
                total_duration += duration
            total_durations.append(total_duration)
        
        self.simulation_results = total_durations
        return total_durations
    
    def calculate_statistics(self):
        """Calculate key statistics from simulation results"""
        if not self.simulation_results:
            return {}
        
        results = np.array(self.simulation_results)
        return {
            'mean': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'p5': np.percentile(results, 5),
            'p50': np.percentile(results, 50),
            'p75': np.percentile(results, 75),
            'p85': np.percentile(results, 85),
            'p95': np.percentile(results, 95),
            'min': np.min(results),
            'max': np.max(results)
        }

def generate_ai_recommendations(statistics, tasks, risk_level):
    """Generate AI-powered recommendations based on simulation results"""
    recommendations = []
    
    # Risk assessment based on standard deviation
    cv = statistics['std'] / statistics['mean']  # Coefficient of variation
    
    if cv > 0.3:
        risk_category = "HIGH"
        risk_color = "risk-high"
    elif cv > 0.15:
        risk_category = "MEDIUM"
        risk_color = "risk-medium"
    else:
        risk_category = "LOW"
        risk_color = "risk-low"
    
    recommendations.append(f"📊 **Overall Risk Level**: <span class='{risk_color}'>{risk_category}</span>")
    
    # Duration analysis
    most_likely_estimate = sum(task['most_likely'] for task in tasks)
    if statistics['p95'] > most_likely_estimate * 1.3:
        recommendations.append("⏰ **Schedule Buffer Needed**: Consider adding 20-30% buffer to account for uncertainty")
    
    # Task-specific recommendations
    task_variances = []
    for task in tasks:
        variance = ((task['pessimistic'] - task['optimistic']) / 6) ** 2
        task_variances.append((task['name'], variance))
    
    # Identify high-risk tasks
    high_risk_tasks = sorted(task_variances, key=lambda x: x[1], reverse=True)[:3]
    if high_risk_tasks:
        recommendations.append("🔍 **High-Risk Tasks Focus**: ")
        for task_name, variance in high_risk_tasks:
            recommendations.append(f"   - {task_name}: High uncertainty - consider breaking down or adding contingency")
    
    # Resource recommendations
    if len(tasks) > 10:
        recommendations.append("👥 **Resource Management**: Large number of tasks detected. Consider parallel execution where possible")
    
    # Critical path analysis recommendation
    if statistics['p85'] - statistics['p50'] > statistics['mean'] * 0.2:
        recommendations.append("🛣️ **Critical Path Optimization**: Significant gap between median and 85th percentile suggests critical path optimization opportunities")
    
    # Contingency planning
    contingency = statistics['p85'] - statistics['mean']
    recommendations.append(f"🛡️ **Contingency Planning**: Recommended contingency: {contingency:.1f} days ({contingency/statistics['mean']*100:.1f}% of mean duration)")
    
    return recommendations

def main():
    st.markdown("<h1 class='main-header'>📊 Schedule Risk Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Initialize simulator
    if 'simulator' not in st.session_state:
        st.session_state.simulator = ScheduleRiskSimulator()
        st.session_state.simulation_run = False
        st.session_state.results = []
        st.session_state.statistics = {}
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Simulation Configuration")
    
    num_simulations = st.sidebar.slider(
        "Number of Simulations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )
    
    confidence_level = st.sidebar.slider(
        "Confidence Level (%)",
        min_value=80,
        max_value=99,
        value=85,
        step=5
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📋 Task Input")
        
        # Task input form
        with st.form("task_form"):
            task_name = st.text_input("Task Name")
            col1a, col2a, col3a = st.columns(3)
            with col1a:
                optimistic = st.number_input("Optimistic (days)", min_value=1, value=5)
            with col2a:
                most_likely = st.number_input("Most Likely (days)", min_value=1, value=10)
            with col3a:
                pessimistic = st.number_input("Pessimistic (days)", min_value=1, value=20)
            
            submitted = st.form_submit_button("Add Task")
            if submitted and task_name:
                st.session_state.simulator.add_task(task_name, optimistic, most_likely, pessimistic)
                st.success(f"Task '{task_name}' added!")
    
    with col2:
        st.header("📊 Current Tasks")
        
        if st.session_state.simulator.tasks:
            tasks_df = pd.DataFrame(st.session_state.simulator.tasks)
            tasks_df['Expected'] = (tasks_df['optimistic'] + 4 * tasks_df['most_likely'] + tasks_df['pessimistic']) / 6
            tasks_df['Std Dev'] = (tasks_df['pessimistic'] - tasks_df['optimistic']) / 6
            st.dataframe(tasks_df, use_container_width=True)
            
            # Calculate total most likely estimate
            total_most_likely = sum(task['most_likely'] for task in st.session_state.simulator.tasks)
            st.metric("Total Most Likely Estimate", f"{total_most_likely:.1f} days")
        else:
            st.info("No tasks added yet. Please add tasks using the form on the left.")
    
    # Run simulation button
    st.markdown("---")
    col_run1, col_run2, col_run3 = st.columns([1, 2, 1])
    with col_run2:
        if st.button("🚀 Run Risk Simulation", use_container_width=True):
            if not st.session_state.simulator.tasks:
                st.warning("Please add at least one task before running the simulation.")
            else:
                with st.spinner("Running Monte Carlo simulation..."):
                    results = st.session_state.simulator.run_monte_carlo_simulation(num_simulations)
                    statistics = st.session_state.simulator.calculate_statistics()
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.statistics = statistics
                    st.session_state.simulation_run = True
                
                st.success(f"Simulation completed with {num_simulations} iterations!")
    
    # Display results if simulation has been run
    if st.session_state.get('simulation_run', False) and st.session_state.results:
        st.header("📈 Simulation Results")
        
        statistics = st.session_state.statistics
        results = np.array(st.session_state.results)  # Convert to numpy array for proper comparison
        
        # Calculate probability of overrun - FIXED VERSION
        total_most_likely = sum(task['most_likely'] for task in st.session_state.simulator.tasks)
        probability_overrun = (np.sum(results > total_most_likely) / len(results)) * 100
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Duration", f"{statistics['mean']:.1f} days")
        with col2:
            st.metric("Standard Deviation", f"{statistics['std']:.1f} days")
        with col3:
            st.metric(f"P{confidence_level} Duration", f"{np.percentile(results, confidence_level):.1f} days")
        with col4:
            st.metric("Probability of Overrun", f"{probability_overrun:.1f}%")
        
        # Visualization
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=results, nbinsx=50, name="Duration Distribution",
                                           marker_color='#1f77b4', opacity=0.7))
            
            # Add confidence interval lines
            fig_hist.add_vline(x=statistics['p5'], line_dash="dash", line_color="green", 
                              annotation_text=f"P5: {statistics['p5']:.1f}")
            fig_hist.add_vline(x=statistics['p50'], line_dash="dash", line_color="orange",
                              annotation_text=f"P50: {statistics['p50']:.1f}")
            fig_hist.add_vline(x=statistics['p95'], line_dash="dash", line_color="red",
                              annotation_text=f"P95: {statistics['p95']:.1f}")
            
            fig_hist.update_layout(
                title="Project Duration Distribution",
                xaxis_title="Total Duration (days)",
                yaxis_title="Frequency",
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with fig_col2:
            # Cumulative probability
            sorted_results = np.sort(results)
            cumulative_prob = np.arange(1, len(sorted_results) + 1) / len(sorted_results)
            
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=sorted_results, y=cumulative_prob, 
                                        mode='lines', name='Cumulative Probability',
                                        line=dict(color='#ff7f0e', width=3)))
            
            fig_cum.update_layout(
                title="Cumulative Probability Distribution",
                xaxis_title="Total Duration (days)",
                yaxis_title="Cumulative Probability",
                yaxis=dict(tickformat=".0%")
            )
            st.plotly_chart(fig_cum, use_container_width=True)
        
        # Risk analysis and AI recommendations
        st.header("🤖 AI Recommendations & Risk Analysis")
        
        # Calculate most likely estimate for comparison
        most_likely_estimate = sum(task['most_likely'] for task in st.session_state.simulator.tasks)
        statistics['most_likely_estimate'] = most_likely_estimate
        
        # Determine risk level
        risk_ratio = statistics['std'] / statistics['mean']
        if risk_ratio > 0.3:
            risk_level = "HIGH"
        elif risk_ratio > 0.15:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        recommendations = generate_ai_recommendations(statistics, st.session_state.simulator.tasks, risk_level)
        
        for recommendation in recommendations:
            st.markdown(f"• {recommendation}", unsafe_allow_html=True)
        
        # Detailed statistics table
        st.header("📋 Detailed Statistics")
        stats_data = {
            'Metric': ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum', 
                      'P5', 'P50', f'P{confidence_level}', 'P95'],
            'Value (days)': [statistics['mean'], statistics['median'], statistics['std'],
                           statistics['min'], statistics['max'], statistics['p5'],
                           statistics['p50'], np.percentile(results, confidence_level),
                           statistics['p95']]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Export results
        st.header("💾 Export Results")
        if st.button("Export Simulation Data"):
            results_df = pd.DataFrame({'Duration': results})
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"risk_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()