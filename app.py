import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
def init_db():
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        age INTEGER,
        weight REAL,
        height REAL,
        gender TEXT,
        activity_level TEXT
    )
    ''')
    
    # Create activities table
    c.execute('''
    CREATE TABLE IF NOT EXISTS activities (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        activity_type TEXT NOT NULL,
        date TEXT NOT NULL,
        duration REAL NOT NULL,
        distance REAL,
        weight REAL,
        reps INTEGER,
        sets INTEGER,
        calories_burned REAL,
        notes TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create goals table
    c.execute('''
    CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        goal_type TEXT NOT NULL,
        target_value REAL NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        status TEXT DEFAULT 'In Progress',
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Authentication functions
def create_user(username, password, age, weight, height, gender, activity_level):
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, age, weight, height, gender, activity_level) VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (username, password, age, weight, height, gender, activity_level))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate(username, password):
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

def get_user_info(user_id):
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    return user

def update_user_info(user_id, age, weight, height, gender, activity_level):
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    c.execute("UPDATE users SET age = ?, weight = ?, height = ?, gender = ?, activity_level = ? WHERE id = ?",
             (age, weight, height, gender, activity_level, user_id))
    conn.commit()
    conn.close()

# Activity tracking functions
def log_activity(user_id, activity_type, date, duration, distance=None, weight=None, reps=None, sets=None, calories_burned=None, notes=None):
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    c.execute('''
    INSERT INTO activities (user_id, activity_type, date, duration, distance, weight, reps, sets, calories_burned, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, activity_type, date, duration, distance, weight, reps, sets, calories_burned, notes))
    conn.commit()
    conn.close()

def get_user_activities(user_id):
    conn = sqlite3.connect('fitness_tracker.db')
    df = pd.read_sql_query("SELECT * FROM activities WHERE user_id = ? ORDER BY date DESC", conn, params=(user_id,))
    conn.close()
    return df

def get_activity_summary(user_id):
    activities = get_user_activities(user_id)
    if activities.empty:
        return pd.DataFrame()
    
    summary = activities.groupby('activity_type').agg({
        'duration': 'sum',
        'calories_burned': 'sum',
        'id': 'count'
    }).reset_index()
    summary.rename(columns={'id': 'count'}, inplace=True)
    return summary

# Goal tracking functions
def set_goal(user_id, goal_type, target_value, start_date, end_date):
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    c.execute('''
    INSERT INTO goals (user_id, goal_type, target_value, start_date, end_date)
    VALUES (?, ?, ?, ?, ?)
    ''', (user_id, goal_type, target_value, start_date, end_date))
    conn.commit()
    conn.close()

def get_user_goals(user_id):
    conn = sqlite3.connect('fitness_tracker.db')
    df = pd.read_sql_query("SELECT * FROM goals WHERE user_id = ?", conn, params=(user_id,))
    conn.close()
    return df

def update_goal_status(goal_id, status):
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    c.execute("UPDATE goals SET status = ? WHERE id = ?", (status, goal_id))
    conn.commit()
    conn.close()

# Calorie calculation functions
def calculate_calories_burned(activity_type, duration, user_weight, user_gender):
    # MET values (Metabolic Equivalent of Task)
    met_values = {
        'Running': 9.8,
        'Walking': 3.5,
        'Cycling': 7.5,
        'Swimming': 8.3,
        'Weightlifting': 6.0,
        'Yoga': 2.5,
        'HIIT': 8.0,
        'Basketball': 6.5,
        'Soccer': 7.0,
        'Tennis': 7.3
    }
    
    # Default to moderate activity if not found
    met = met_values.get(activity_type, 5.0)
    
    # Calorie calculation formula: calories = MET * weight(kg) * duration(hours)
    # Convert duration from minutes to hours
    duration_hours = duration / 60
    
    # Adjust for gender (approximate adjustment)
    gender_factor = 1.0 if user_gender == 'Male' else 0.9
    
    calories = met * user_weight * duration_hours * gender_factor
    
    return round(calories, 2)

# Machine learning functions
def train_recommendation_model(user_id):
    activities = get_user_activities(user_id)
    if len(activities) < 5:  # Need enough data to train
        return None
    
    # Prepare data
    activities['date'] = pd.to_datetime(activities['date'])
    activities['day_of_week'] = activities['date'].dt.dayofweek
    activities['month'] = activities['date'].dt.month
    
    # Features and target
    X = activities[['duration', 'day_of_week', 'month']]
    y = activities['activity_type']
    
    # One-hot encode activity type
    activity_types = activities['activity_type'].unique()
    
    # Train a simple model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return model, activity_types

def predict_next_activity(model, user_data, activity_types):
    if model is None:
        return "Not enough data for prediction"
    
    # Prepare input data
    today = datetime.datetime.now()
    input_data = np.array([[
        user_data.get('preferred_duration', 30),
        today.weekday(),
        today.month
    ]])
    
    # Predict
    prediction = model.predict(input_data)[0]
    return prediction

# Dashboard components
def render_sidebar(user_id):
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Log Activity", "Set Goals", "Profile", "Analysis"])
    
    st.sidebar.divider()
    st.sidebar.subheader("User Info")
    user = get_user_info(user_id)
    if user:
        st.sidebar.text(f"Username: {user[1]}")
        st.sidebar.text(f"Age: {user[3]}")
        st.sidebar.text(f"Weight: {user[4]} kg")
        st.sidebar.text(f"Height: {user[5]} cm")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.rerun()
    
    return page

def render_dashboard(user_id):
    st.title("Fitness Dashboard")
    
    # Get user data
    user = get_user_info(user_id)
    activities = get_user_activities(user_id)
    goals = get_user_goals(user_id)
    
    # Display welcome message
    st.markdown(f"### Welcome back, {user[1]}!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_activities = len(activities) if not activities.empty else 0
        st.metric("Total Activities", total_activities)
    
    with col2:
        total_duration = activities['duration'].sum() if not activities.empty else 0
        st.metric("Total Minutes", f"{total_duration:.0f}")
    
    with col3:
        total_calories = activities['calories_burned'].sum() if not activities.empty else 0
        st.metric("Calories Burned", f"{total_calories:.0f}")
    
    with col4:
        active_goals = len(goals[goals['status'] == 'In Progress']) if not goals.empty else 0
        st.metric("Active Goals", active_goals)
    
    # Recent activities
    st.subheader("Recent Activities")
    if not activities.empty:
        recent = activities.head(5)
        st.dataframe(recent[['date', 'activity_type', 'duration', 'calories_burned']])
    else:
        st.info("No activities logged yet. Start by logging your first activity!")
    
    # Activity distribution
    st.subheader("Activity Distribution")
    if not activities.empty:
        activity_counts = activities['activity_type'].value_counts().reset_index()
        activity_counts.columns = ['Activity', 'Count']
        
        fig = px.pie(activity_counts, values='Count', names='Activity', hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log activities to see your distribution.")
    
    # Weekly progress
    st.subheader("Weekly Progress")
    if not activities.empty:
        activities['date'] = pd.to_datetime(activities['date'])
        activities['week'] = activities['date'].dt.isocalendar().week
        weekly_data = activities.groupby('week')['calories_burned'].sum().reset_index()
        
        fig = px.line(weekly_data, x='week', y='calories_burned', markers=True)
        fig.update_layout(
            xaxis_title="Week Number",
            yaxis_title="Calories Burned",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log activities to see your weekly progress.")
    
    # Goal progress
    st.subheader("Goal Progress")
    if not goals.empty and len(goals[goals['status'] == 'In Progress']) > 0:
        active_goals = goals[goals['status'] == 'In Progress']
        for _, goal in active_goals.iterrows():
            goal_type = goal['goal_type']
            target = goal['target_value']
            end_date = pd.to_datetime(goal['end_date'])
            
            # Calculate progress based on goal type
            if goal_type == 'Total Activities':
                current = len(activities)
                progress = min(current / target, 1.0) if target > 0 else 0
            elif goal_type == 'Total Duration':
                current = activities['duration'].sum()
                progress = min(current / target, 1.0) if target > 0 else 0
            elif goal_type == 'Calories Burned':
                current = activities['calories_burned'].sum()
                progress = min(current / target, 1.0) if target > 0 else 0
            else:
                current = 0
                progress = 0
            
            # Display progress bar
            st.text(f"{goal_type}: {current:.0f}/{target:.0f}")
            st.progress(float(progress))
            st.text(f"Target Date: {end_date.strftime('%Y-%m-%d')}")
            
            # Check if goal is completed
            if progress >= 1.0:
                update_goal_status(goal['id'], 'Completed')
                st.success(f"Goal achieved! 🎉")
    else:
        st.info("No active goals. Set new goals to track your progress!")
    
    # Recommendations
    st.subheader("Personalized Recommendations")
    if not activities.empty and len(activities) >= 5:
        try:
            model, activity_types = train_recommendation_model(user_id)
            user_data = {'preferred_duration': activities['duration'].mean()}
            next_activity = predict_next_activity(model, user_data, activity_types)
            
            st.markdown(f"""
            Based on your activity patterns, we recommend:
            - **Next activity**: {next_activity}
            - **Duration**: {user_data['preferred_duration']:.0f} minutes
            - **Focus area**: {'Cardio' if next_activity in ['Running', 'Swimming', 'Cycling'] else 'Strength' if next_activity == 'Weightlifting' else 'Flexibility'}
            """)
        except Exception as e:
            st.warning("Could not generate recommendations. Please log more diverse activities.")
    else:
        st.info("Log at least 5 activities to get personalized recommendations.")

def render_log_activity(user_id):
    st.title("Log Activity")
    
    user = get_user_info(user_id)

    # ✅ Initialize session state for activity_type if it does not exist
    if "activity_type" not in st.session_state:
        st.session_state.activity_type = "Running"

    # ✅ Use session state to store the selected activity type
    activity_type = st.selectbox(
        "Activity Type",
        ["Running", "Walking", "Cycling", "Swimming", "Weightlifting", "Yoga", "HIIT", "Basketball", "Soccer", "Tennis"],
        index=["Running", "Walking", "Cycling", "Swimming", "Weightlifting", "Yoga", "HIIT", "Basketball", "Soccer", "Tennis"].index(st.session_state.activity_type),
        key="activity_selector"
    )

    # ✅ Detect change in activity type and trigger a rerun
    if activity_type != st.session_state.activity_type:
        st.session_state.activity_type = activity_type
        st.rerun()

    with st.form("activity_form"):
        st.subheader("Activity Details")
        
        date = st.date_input("Date", datetime.datetime.now())
        duration = st.number_input("Duration (minutes)", min_value=1, value=30)

        # ✅ Show relevant fields based on activity type
        if activity_type in ["Running", "Walking", "Cycling"]:
            distance = st.number_input("Distance (km)", min_value=0.1, value=5.0, step=0.1)
        else:
            distance = None

        if activity_type == "Weightlifting":
            weight = st.number_input("Weight (kg)", min_value=0.0, value=20.0, step=0.5)
            sets = st.number_input("Sets", min_value=1, value=3)
            reps = st.number_input("Reps per set", min_value=1, value=10)
        else:
            weight, sets, reps = None, None, None

        notes = st.text_area("Notes (optional)")

        # ✅ Calculate calories burned
        if user:
            calories_burned = calculate_calories_burned(
                activity_type, 
                duration, 
                user[4],  # user weight
                user[6]   # user gender
            )
            st.info(f"Estimated calories burned: {calories_burned} kcal")
        else:
            calories_burned = None

        submitted = st.form_submit_button("Log Activity")

        if submitted:
            log_activity(
                user_id=user_id,
                activity_type=activity_type,
                date=date.strftime("%Y-%m-%d"),
                duration=duration,
                distance=distance,
                weight=weight,
                reps=reps,
                sets=sets,
                calories_burned=calories_burned,
                notes=notes
            )
            st.success("Activity logged successfully!")
            st.balloons()
             # ✅ Refresh UI after submission


def render_set_goals(user_id):
    st.title("Set Fitness Goals")
    
    # Display existing goals
    goals = get_user_goals(user_id)
    if not goals.empty:
        st.subheader("Current Goals")
        for _, goal in goals.iterrows():
            with st.expander(f"{goal['goal_type']} - {goal['status']}"):
                st.write(f"Target: {goal['target_value']}")
                st.write(f"Start Date: {goal['start_date']}")
                st.write(f"End Date: {goal['end_date']}")
                st.write(f"Status: {goal['status']}")
                
                if goal['status'] == 'In Progress':
                    if st.button(f"Mark as Completed", key=f"complete_{goal['id']}"):
                        update_goal_status(goal['id'], 'Completed')
                        st.success("Goal marked as completed!")
                        st.rerun()
    
    # Form to set new goals
    st.subheader("Set New Goal")
    with st.form("goal_form"):
        goal_type = st.selectbox(
            "Goal Type",
            ["Total Activities", "Total Duration", "Calories Burned", "Weight Loss", "Distance"]
        )
        
        target_value = st.number_input("Target Value", min_value=1.0, value=100.0)
        
        start_date = st.date_input("Start Date", datetime.datetime.now())
        end_date = st.date_input("End Date", datetime.datetime.now() + datetime.timedelta(days=30))
        
        submitted = st.form_submit_button("Set Goal")
        
        if submitted:
            if end_date <= start_date:
                st.error("End date must be after start date")
            else:
                set_goal(
                    user_id=user_id,
                    goal_type=goal_type,
                    target_value=target_value,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                st.success("Goal set successfully!")
                st.rerun()

def render_profile(user_id):
    st.title("User Profile")
    
    user = get_user_info(user_id)
    
    if user:
        with st.form("profile_form"):
            st.subheader("Update Profile")
            
            age = st.number_input("Age", min_value=1, max_value=120, value=user[3] if user[3] else 30)
            weight = st.number_input("Weight (kg)", min_value=1.0, max_value=500.0, value=user[4] if user[4] else 70.0)
            height = st.number_input("Height (cm)", min_value=1.0, max_value=300.0, value=user[5] if user[5] else 170.0)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if user[6] == "Male" else 1 if user[6] == "Female" else 2)
            activity_level = st.select_slider(
                "Activity Level",
                options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
                value=user[7] if user[7] else "Moderately Active"
            )
            
            submitted = st.form_submit_button("Update Profile")
            
            if submitted:
                update_user_info(user_id, age, weight, height, gender, activity_level)
                st.success("Profile updated successfully!")
                st.rerun()
        
        # Calculate and display BMI
        if user[4] and user[5]:
            bmi = user[4] / ((user[5]/100) ** 2)
            st.subheader("Body Mass Index (BMI)")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your BMI", f"{bmi:.1f}")
            
            with col2:
                if bmi < 18.5:
                    category = "Underweight"
                elif bmi < 25:
                    category = "Normal weight"
                elif bmi < 30:
                    category = "Overweight"
                else:
                    category = "Obesity"
                st.metric("Category", category)
            
            # BMI chart
            bmi_ranges = [
                {"range": "Underweight", "min": 0, "max": 18.5, "color": "blue"},
                {"range": "Normal weight", "min": 18.5, "max": 25, "color": "green"},
                {"range": "Overweight", "min": 25, "max": 30, "color": "orange"},
                {"range": "Obesity", "min": 30, "max": 40, "color": "red"}
            ]
            
            fig = go.Figure()
            
            # Add BMI ranges
            for r in bmi_ranges:
                fig.add_trace(go.Scatter(
                    x=[r["min"], r["max"]],
                    y=[0, 0],
                    mode="lines",
                    line=dict(width=10, color=r["color"]),
                    name=r["range"]
                ))
            
            # Add marker for user's BMI
            fig.add_trace(go.Scatter(
                x=[bmi],
                y=[0],
                mode="markers",
                marker=dict(size=15, color="black"),
                name="Your BMI"
            ))
            
            fig.update_layout(
                title="BMI Scale",
                xaxis=dict(
                    title="BMI",
                    range=[15, 40],
                    showgrid=False
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **BMI Categories:**
            - Underweight: BMI less than 18.5
            - Normal weight: BMI 18.5 to 24.9
            - Overweight: BMI 25 to 29.9
            - Obesity: BMI 30 or greater
            
            Note: BMI is a screening tool but does not diagnose body fatness or health.
            """)

def render_analysis(user_id):
    st.title("Fitness Analysis")
    
    activities = get_user_activities(user_id)
    
    if activities.empty:
        st.info("Log some activities to see your analysis!")
        return
    
    # Convert date to datetime
    activities['date'] = pd.to_datetime(activities['date'])
    
    # Time period filter
    time_period = st.selectbox(
        "Select Time Period",
        ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "This Year"]
    )
    
    if time_period == "Last 7 Days":
        filtered_activities = activities[activities['date'] >= (datetime.datetime.now() - datetime.timedelta(days=7))]
    elif time_period == "Last 30 Days":
        filtered_activities = activities[activities['date'] >= (datetime.datetime.now() - datetime.timedelta(days=30))]
    elif time_period == "Last 90 Days":
        filtered_activities = activities[activities['date'] >= (datetime.datetime.now() - datetime.timedelta(days=90))]
    elif time_period == "This Year":
        filtered_activities = activities[activities['date'].dt.year == datetime.datetime.now().year]
    else:
        filtered_activities = activities
    
    if filtered_activities.empty:
        st.info(f"No activities found for the selected time period: {time_period}")
        return
    
    # Activity type filter
    activity_types = ["All"] + list(filtered_activities['activity_type'].unique())
    selected_activity = st.selectbox("Select Activity Type", activity_types)
    
    if selected_activity != "All":
        filtered_activities = filtered_activities[filtered_activities['activity_type'] == selected_activity]
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Activity Summary", "Trends", "Performance", "Machine Learning Insights"])
    
    with tab1:
        st.subheader("Activity Summary")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_activities = len(filtered_activities)
            st.metric("Total Activities", total_activities)
        
        with col2:
            total_duration = filtered_activities['duration'].sum()
            st.metric("Total Duration (min)", f"{total_duration:.0f}")
        
        with col3:
            total_calories = filtered_activities['calories_burned'].sum()
            st.metric("Total Calories Burned", f"{total_calories:.0f}")
        
        # Activity breakdown
        st.subheader("Activity Breakdown")
        activity_summary = filtered_activities.groupby('activity_type').agg({
            'id': 'count',
            'duration': 'sum',
            'calories_burned': 'sum'
        }).reset_index()
        activity_summary.columns = ['Activity Type', 'Count', 'Total Duration (min)', 'Total Calories Burned']
        st.dataframe(activity_summary)
        
        # Activity distribution chart
        fig = px.pie(
            activity_summary, 
            values='Count', 
            names='Activity Type',
            title='Activity Distribution',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Activity Trends")
        
        # Time series analysis
        filtered_activities['year_month'] = filtered_activities['date'].dt.to_period('M')
        monthly_data = filtered_activities.groupby('year_month').agg({
            'id': 'count',
            'duration': 'sum',
            'calories_burned': 'sum'
        }).reset_index()
        monthly_data['year_month'] = monthly_data['year_month'].astype(str)
        
        # Monthly activity count
        fig1 = px.bar(
            monthly_data,
            x='year_month',
            y='id',
            title='Monthly Activity Count',
            labels={'year_month': 'Month', 'id': 'Number of Activities'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Monthly calories burned
        fig2 = px.line(
            monthly_data,
            x='year_month',
            y='calories_burned',
            title='Monthly Calories Burned',
            labels={'year_month': 'Month', 'calories_burned': 'Calories Burned'},
            markers=True
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Day of week analysis
        filtered_activities['day_of_week'] = filtered_activities['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_data = filtered_activities.groupby('day_of_week').agg({
            'id': 'count',
            'duration': 'mean',
            'calories_burned': 'mean'
        }).reset_index()
        
        # Reorder days
        day_data['day_of_week'] = pd.Categorical(day_data['day_of_week'], categories=day_order, ordered=True)
        day_data = day_data.sort_values('day_of_week')
        
        # Activity count by day of week
        fig3 = px.bar(
            day_data,
            x='day_of_week',
            y='id',
            title='Activity Count by Day of Week',
            labels={'day_of_week': 'Day', 'id': 'Number of Activities'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.subheader("Performance Analysis")
        
        if selected_activity != "All":
             
            st.subheader("Performance Analysis")
        
        if selected_activity != "All":
            # Performance metrics for specific activity
            st.write(f"Performance metrics for {selected_activity}")
            
            # Time series of performance
            performance_data = filtered_activities.sort_values('date')
            
            if selected_activity in ["Running", "Walking", "Cycling"] and 'distance' in performance_data.columns:
                # Calculate pace (min/km)
                performance_data['pace'] = performance_data['duration'] / performance_data['distance']
                
                fig = px.line(
                    performance_data,
                    x='date',
                    y='pace',
                    title=f'{selected_activity} Pace Over Time (min/km)',
                    labels={'date': 'Date', 'pace': 'Pace (min/km)'},
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Distance over time
                fig2 = px.line(
                    performance_data,
                    x='date',
                    y='distance',
                    title=f'{selected_activity} Distance Over Time',
                    labels={'date': 'Date', 'distance': 'Distance (km)'},
                    markers=True
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            elif selected_activity == "Weightlifting" and 'weight' in performance_data.columns:
                # Weight lifted over time
                fig = px.line(
                    performance_data,
                    x='date',
                    y='weight',
                    title='Weight Lifted Over Time',
                    labels={'date': 'Date', 'weight': 'Weight (kg)'},
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Duration over time for all activities
            fig3 = px.line(
                performance_data,
                x='date',
                y='duration',
                title=f'{selected_activity} Duration Over Time',
                labels={'date': 'Date', 'duration': 'Duration (min)'},
                markers=True
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Calories burned over time
            fig4 = px.line(
                performance_data,
                x='date',
                y='calories_burned',
                title=f'{selected_activity} Calories Burned Over Time',
                labels={'date': 'Date', 'calories_burned': 'Calories Burned'},
                markers=True
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            # Compare performance across activities
            st.write("Comparing performance across different activities")
            
            # Average duration by activity type
            avg_duration = filtered_activities.groupby('activity_type')['duration'].mean().reset_index()
            fig = px.bar(
                avg_duration,
                x='activity_type',
                y='duration',
                title='Average Duration by Activity Type',
                labels={'activity_type': 'Activity Type', 'duration': 'Average Duration (min)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Average calories burned by activity type
            avg_calories = filtered_activities.groupby('activity_type')['calories_burned'].mean().reset_index()
            fig2 = px.bar(
                avg_calories,
                x='activity_type',
                y='calories_burned',
                title='Average Calories Burned by Activity Type',
                labels={'activity_type': 'Activity Type', 'calories_burned': 'Average Calories Burned'}
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.subheader("Machine Learning Insights")
        
        if len(filtered_activities) >= 10:
            st.write("Using machine learning to analyze your fitness data")
            
            # Prepare data for ML
            ml_data = filtered_activities.copy()
            ml_data['day_of_week'] = ml_data['date'].dt.dayofweek
            ml_data['month'] = ml_data['date'].dt.month
            ml_data['day'] = ml_data['date'].dt.day
            ml_data['hour'] = 12  # Default hour if not available
            
            # Feature selection
            features = ['day_of_week', 'month', 'day', 'duration']
            
            # Predict calories burned
            if len(ml_data) >= 20:
                st.write("#### Calories Burned Prediction Model")
                st.write("This model predicts calories burned based on activity features")
                
                X = ml_data[features]
                y = ml_data['calories_burned']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # Feature importance
                importance = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    feature_importance,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance for Calories Burned Prediction'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Model performance
                y_pred = model.predict(X_test)
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                
                st.metric("Model RMSE", f"{rmse:.2f} calories")
                
                # Prediction vs Actual
                results = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred
                })
                
                fig2 = px.scatter(
                    results,
                    x='Actual',
                    y='Predicted',
                    title='Predicted vs Actual Calories Burned',
                    labels={'Actual': 'Actual Calories', 'Predicted': 'Predicted Calories'}
                )
                fig2.add_trace(
                    go.Scatter(
                        x=[results['Actual'].min(), results['Actual'].max()],
                        y=[results['Actual'].min(), results['Actual'].max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    )
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Activity recommendation
            st.write("#### Activity Recommendation")
            st.write("Based on your patterns, we recommend these activities:")
            
            try:
                # Train a simple classifier
                X = ml_data[['day_of_week', 'month']]
                y = ml_data['activity_type']
                
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                
                # Get today's features
                today = datetime.datetime.now()
                today_features = np.array([[today.weekday(), today.month]])
                
                # Predict
                prediction = model.predict(today_features)[0]
                
                st.success(f"Recommended activity for today: **{prediction}**")
                MODEL_PATH = "ml_model.pkl"
                with open(MODEL_PATH, "wb") as file:
                    pickle.dump(model, file) 
                # Get probabilities for all activities
                activity_probs = model.predict_proba(today_features)[0]
                activities = model.classes_
                
                # Create recommendations dataframe
                recommendations = pd.DataFrame({
                    'Activity': activities,
                    'Recommendation Score': activity_probs
                }).sort_values('Recommendation Score', ascending=False)
                
                # Plot recommendation scores
                fig = px.bar(
                    recommendations,
                    x='Activity',
                    y='Recommendation Score',
                    title='Activity Recommendations Based on Your Patterns',
                    labels={'Activity': 'Activity Type', 'Recommendation Score': 'Recommendation Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
              
            except Exception as e:
                st.warning(f"Could not generate activity recommendations. Try logging more diverse activities.")
        else:
            st.info("Log at least 10 activities to see machine learning insights!")

# Authentication UI
def render_auth():
    st.title("Personal Fitness Tracker")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                user = authenticate(username, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user[0]
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Register")
            new_username = st.text_input("Username", key="reg_username")
            new_password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            # Basic user info
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            weight = st.number_input("Weight (kg)", min_value=1.0, max_value=500.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=1.0, max_value=300.0, value=170.0)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            activity_level = st.select_slider(
                "Activity Level",
                options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
                value="Moderately Active"
            )
            
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif not new_username or not new_password:
                    st.error("Username and password are required")
                else:
                    success = create_user(new_username, new_password, age, weight, height, gender, activity_level)
                    if success:
                        st.success("Registration successful! You can now login.")
                    else:
                        st.error("Username already exists")

# Main app
def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    # Check authentication
    if st.session_state.authenticated and st.session_state.user_id:
        # Render sidebar and get selected page
        page = render_sidebar(st.session_state.user_id)
        
        # Render selected page
        if page == "Dashboard":
            render_dashboard(st.session_state.user_id)
        elif page == "Log Activity":
            render_log_activity(st.session_state.user_id)
        elif page == "Set Goals":
            render_set_goals(st.session_state.user_id)
        elif page == "Profile":
            render_profile(st.session_state.user_id)
        elif page == "Analysis":
            render_analysis(st.session_state.user_id)
    else:
        # Render authentication UI
        render_auth()

if __name__ == "__main__":
    main()