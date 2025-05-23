import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import random
from io import BytesIO
import base64
import io
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib.pyplot as plt
import hashlib
import os
import pickle
import json

# Set page configuration
st.set_page_config(
    page_title="AI-Solutions Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for generating test data
def generate_web_logs(num_entries=1000):
    """Generate sample web server logs for AI-Solutions"""
    
    # Define possible data
    ip_ranges = [
        "128.1.0.", "155.55.0.", "157.20.5.", "157.20.20.", "157.20.30.",
        "192.168.1.", "45.60.70.", "78.90.100.", "101.102.103.", "200.201.202."
    ]
    
    pages = [
        "/index.html", "/about.html", "/services.html", "/contact.html",
        "/images/logo.jpg", "/images/events.jpg", "/images/team.jpg", "/images/products.jpg",
        "/event.php", "/scheduledemo.php", "/prototype.php", "/ai-assistant.php",
        "/jobs/listing.php", "/jobs/apply.php", "/jobs/engineering.php", "/jobs/sales.php",
        "/products/ai-tools.html", "/products/prototyping.html"
    ]
    
    status_codes = [200, 200, 200, 200, 200, 304, 404, 500]  # Weighted for more 200s
    
    # Country mapping for IPs (simplified for demo)
    country_mapping = {
        "128.1.0": "USA",
        "155.55.0": "UK",
        "157.20.5": "Germany",
        "157.20.20": "France",
        "157.20.30": "Japan",
        "192.168.1": "Canada",
        "45.60.70": "Australia",
        "78.90.100": "India",
        "101.102.103": "Brazil",
        "200.201.202": "China"
    }
    
    # Generate timestamps for the past month
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    dates = [start_date + datetime.timedelta(seconds=x) for x in range(0, int((end_date - start_date).total_seconds()), int((end_date - start_date).total_seconds() / num_entries))]
    
    logs = []
    
    for i in range(num_entries):
        # Generate timestamp
        timestamp = dates[i].strftime("%H:%M:%S")
        
        # Generate IP
        ip_base = random.choice(ip_ranges)
        ip = ip_base + str(random.randint(1, 254))
        
        # Get country from IP
        country = country_mapping[ip_base[:-1]]
        
        # Generate request
        method = "GET"
        page = random.choice(pages)
        
        # Generate status code
        status = random.choice(status_codes)
        
        # Determine page type
        if "scheduledemo.php" in page:
            page_type = "Demo Request"
        elif "event.php" in page:
            page_type = "Event Registration"
        elif "prototype.php" in page:
            page_type = "Prototype Tool"
        elif "ai-assistant.php" in page:
            page_type = "AI Assistant"
        elif "jobs" in page:
            page_type = "Job Listing"
        elif "images" in page:
            page_type = "Image"
        else:
            page_type = "Content Page"
        
        # Add sales rep for demo and event requests
        sales_rep = None
        if page_type in ["Demo Request", "Event Registration"]:
            sales_reps = ["Sam", "Alex", "Jamie", "Taylor", "Morgan"]
            sales_rep = random.choice(sales_reps)
        
        # Add job type if applicable
        job_type = None
        if "jobs" in page:
            job_types = ["Engineering", "Sales", "Marketing", "Design", "Management"]
            job_type = random.choice(job_types)
        
        logs.append({
            "timestamp": timestamp,
            "date": dates[i].strftime("%Y-%m-%d"),
            "time": dates[i].strftime("%H:%M:%S"),
            "hour": dates[i].hour,
            "ip": ip,
            "country": country,
            "method": method,
            "page": page,
            "status": status,
            "page_type": page_type,
            "sales_rep": sales_rep,
            "job_type": job_type
        })
    
    return pd.DataFrame(logs)

# User authentication system
def init_auth_system():
    """Initialize the authentication system"""
    # Default users if no saved users exist
    default_users = {
        "admin": {
            "password": hashlib.sha256("admin123".encode()).hexdigest(),
            "role": "admin",
            "name": "Administrator"
        },
        "sales": {
            "password": hashlib.sha256("sales123".encode()).hexdigest(),
            "role": "sales",
            "name": "Sales Team"
        },
        "viewer": {
            "password": hashlib.sha256("viewer123".encode()).hexdigest(),
            "role": "viewer",
            "name": "Data Viewer"
        }
    }
    
    # Define role permissions
    role_permissions = {
        "admin": ["Overview", "Geographic Analysis", "Sales Performance", "User Engagement", 
                 "Technical Health", "Raw Data", "Navigation Flow", "User Management"],
        "sales": ["Overview", "Geographic Analysis", "Sales Performance", "User Engagement", 
                 "Navigation Flow"],
        "viewer": ["Overview", "Geographic Analysis", "User Engagement"]
    }
    
    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    
    if "users" not in st.session_state:
        # Try to load saved users
        try:
            with open("users.pickle", "rb") as f:
                st.session_state.users = pickle.load(f)
        except:
            st.session_state.users = default_users
    
    if "role_permissions" not in st.session_state:
        st.session_state.role_permissions = role_permissions
        
    return st.session_state.users, st.session_state.role_permissions

def login_form():
    """Display login form and handle authentication"""
    st.sidebar.title("Login")
    
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username in st.session_state.users:
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                if st.session_state.users[username]["password"] == hashed_password:
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    st.sidebar.success(f"Welcome {st.session_state.users[username]['name']}!")
                    # Rerun to refresh the page
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Invalid password")
            else:
                st.sidebar.error("User does not exist")

def logout():
    """Handle user logout"""
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        # Rerun to refresh the page
        st.experimental_rerun()

def user_management():
    """User management section for admins"""
    st.title("👥 User Management")
    
    # Display current users
    st.header("Current Users")
    
    user_data = []
    for username, user_info in st.session_state.users.items():
        user_data.append({
            "Username": username,
            "Role": user_info["role"],
            "Name": user_info["name"]
        })
    
    user_df = pd.DataFrame(user_data)
    st.dataframe(user_df)
    
    # Add new user
    st.header("Add New User")
    
    with st.form("add_user_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        new_name = st.text_input("Full Name")
        new_role = st.selectbox("Role", ["admin", "sales", "viewer"])
        
        submit_button = st.form_submit_button("Add User")
        
        if submit_button:
            if new_username in st.session_state.users:
                st.error("Username already exists")
            elif not new_username or not new_password or not new_name:
                st.error("All fields are required")
            else:
                hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                st.session_state.users[new_username] = {
                    "password": hashed_password,
                    "role": new_role,
                    "name": new_name
                }
                
                # Save users to file
                with open("users.pickle", "wb") as f:
                    pickle.dump(st.session_state.users, f)
                
                st.success(f"User {new_username} added successfully")
    
    # Change password
    st.header("Change User Password")
    
    with st.form("change_password_form"):
        username = st.selectbox("Select User", list(st.session_state.users.keys()))
        new_password = st.text_input("New Password", type="password")
        
        submit_button = st.form_submit_button("Change Password")
        
        if submit_button:
            if not new_password:
                st.error("Password cannot be empty")
            else:
                hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                st.session_state.users[username]["password"] = hashed_password
                
                # Save users to file
                with open("users.pickle", "wb") as f:
                    pickle.dump(st.session_state.users, f)
                
                st.success(f"Password for {username} changed successfully")
    
    # Delete user
    st.header("Delete User")
    
    with st.form("delete_user_form"):
        username_to_delete = st.selectbox("Select User to Delete", list(st.session_state.users.keys()))
        
        submit_button = st.form_submit_button("Delete User")
        
        if submit_button:
            if username_to_delete == st.session_state.current_user:
                st.error("You cannot delete your own account")
            elif len(st.session_state.users) <= 1:
                st.error("Cannot delete the last user")
            else:
                del st.session_state.users[username_to_delete]
                
                # Save users to file
                with open("users.pickle", "wb") as f:
                    pickle.dump(st.session_state.users, f)
                
                st.success(f"User {username_to_delete} deleted successfully")

def generate_pdf_report(logs_df, sales_df=None, country_filter="All"):
    """Generate a PDF report from the dashboard data"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = styles['Heading1']
    title = Paragraph("AI-Solutions Dashboard Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 20))
    
    # Date range and filters
    date_range = f"Date Range: {logs_df['date'].min()} to {logs_df['date'].max()}"
    country_info = f"Country Filter: {country_filter}"
    filter_text = Paragraph(f"{date_range}<br/>{country_info}", styles['Normal'])
    elements.append(filter_text)
    elements.append(Spacer(1, 20))
    
    # Traffic Overview
    subtitle = Paragraph("Traffic Overview", styles['Heading2'])
    elements.append(subtitle)
    elements.append(Spacer(1, 10))
    
    # Create traffic summary table
    traffic_data = [
        ["Metric", "Value"],
        ["Total Visitors", str(logs_df['ip'].nunique())],
        ["Geographic Reach", f"{logs_df['country'].nunique()} countries"],
        ["Total Page Views", str(len(logs_df))],
        ["Demo Requests", str(len(logs_df[logs_df['page_type'] == 'Demo Request']))],
    ]
    
    traffic_table = Table(traffic_data, colWidths=[200, 100])
    traffic_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(traffic_table)
    elements.append(Spacer(1, 20))
    
    # Country distribution
    if len(logs_df['country'].unique()) > 1:
        subtitle = Paragraph("Geographic Distribution", styles['Heading2'])
        elements.append(subtitle)
        elements.append(Spacer(1, 10))
        
        # Create a matplotlib chart and save it to the PDF
        plt.figure(figsize=(8, 4))
        country_counts = logs_df['country'].value_counts().head(10)
        country_counts.plot(kind='bar')
        plt.title('Top 10 Countries by Traffic')
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        image = Image(img_buffer)
        image.drawHeight = 200
        image.drawWidth = 400
        elements.append(image)
        elements.append(Spacer(1, 20))
    
    # Page type distribution
    subtitle = Paragraph("Page Type Distribution", styles['Heading2'])
    elements.append(subtitle)
    elements.append(Spacer(1, 10))
    
    page_type_counts = logs_df['page_type'].value_counts().reset_index()
    page_type_counts.columns = ['Page Type', 'Count']
    
    page_data = [["Page Type", "Count"]]
    for _, row in page_type_counts.iterrows():
        page_data.append([row['Page Type'], str(row['Count'])])
    
    page_table = Table(page_data, colWidths=[200, 100])
    page_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(page_table)
    elements.append(Spacer(1, 20))
    
    # Add sales data if available
    if sales_df is not None and not sales_df.empty:
        subtitle = Paragraph("Sales Overview", styles['Heading2'])
        elements.append(subtitle)
        elements.append(Spacer(1, 10))
        
        sales_data = [
            ["Metric", "Value"],
            ["Total Revenue", f"${sales_df['revenue'].sum():,.2f}"],
            ["Average Deal Size", f"${sales_df['revenue'].mean():,.2f}"],
            ["Deals Closed", str(len(sales_df))],
        ]
        
        sales_table = Table(sales_data, colWidths=[200, 100])
        sales_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(sales_table)
    
    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


def get_download_pdf_link(logs_df, sales_df=None, filename="report.pdf", text="Download PDF Report", country_filter="All"):
    """Generate a download link for a PDF report"""
    pdf = generate_pdf_report(logs_df, sales_df, country_filter)
    b64 = base64.b64encode(pdf.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">{text}</a>'
    return href

def generate_sales_data(logs_df):
    """Generate sales performance data based on the web logs"""
    
    # Extract demo requests and event registrations
    demo_logs = logs_df[logs_df['page_type'] == 'Demo Request'].copy()
    event_logs = logs_df[logs_df['page_type'] == 'Event Registration'].copy()
    
    # Generate unique sales
    sales_data = []
    
    # Process demo conversions
    for _, row in demo_logs.iterrows():
        if row['sales_rep'] and random.random() < 0.6:  # 60% conversion rate
            deal_size = random.randint(5000, 50000)
            product_type = random.choice(["AI Assistant", "Prototyping Tool"])
            days_to_close = random.randint(1, 30)
            close_date = datetime.datetime.strptime(row['date'], "%Y-%m-%d") + datetime.timedelta(days=days_to_close)
            
            # Only include closed deals
            if close_date <= datetime.datetime.now():
                sales_data.append({
                    "date": close_date.strftime("%Y-%m-%d"),
                    "sales_rep": row['sales_rep'],
                    "country": row['country'],
                    "product": product_type,
                    "revenue": deal_size,
                    "source": "Demo",
                    "lead_time_days": days_to_close
                })
    
    # Process event conversions
    for _, row in event_logs.iterrows():
        if row['sales_rep'] and random.random() < 0.3:  # 30% conversion rate
            deal_size = random.randint(5000, 50000)
            product_type = random.choice(["AI Assistant", "Prototyping Tool"])
            days_to_close = random.randint(7, 45)
            close_date = datetime.datetime.strptime(row['date'], "%Y-%m-%d") + datetime.timedelta(days=days_to_close)
            
            # Only include closed deals
            if close_date <= datetime.datetime.now():
                sales_data.append({
                    "date": close_date.strftime("%Y-%m-%d"),
                    "sales_rep": row['sales_rep'],
                    "country": row['country'],
                    "product": product_type,
                    "revenue": deal_size,
                    "source": "Event",
                    "lead_time_days": days_to_close
                })
    
    sales_df = pd.DataFrame(sales_data)
    
    # Add targets for each sales rep
    if not sales_df.empty:
        sales_targets = {
            "Sam": 250000,
            "Alex": 200000,
            "Jamie": 225000,
            "Taylor": 175000,
            "Morgan": 210000
        }
        
        sales_df['target'] = sales_df['sales_rep'].map(sales_targets)
        
        # Create a summary DataFrame
        summary_data = []
        for rep, target in sales_targets.items():
            rep_sales = sales_df[sales_df['sales_rep'] == rep]['revenue'].sum()
            performance = (rep_sales / target) * 100
            
            if performance >= 110:
                status = "Exceeding Target"
                color = "green"
            elif performance >= 100:
                status = "Meeting Target"
                color = "blue"
            elif performance >= 85:
                status = "Near Target"
                color = "yellow"
            else:
                status = "Below Target"
                color = "red"
                
            summary_data.append({
                "sales_rep": rep,
                "total_sales": rep_sales,
                "target": target,
                "performance": performance,
                "status": status,
                "color": color
            })
            
        summary_df = pd.DataFrame(summary_data)
        return sales_df, summary_df
    else:
        return pd.DataFrame(), pd.DataFrame()

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Set up sidebar
st.sidebar.image("https://img.freepik.com/free-vector/gradient-ai-logo-template_23-2149872595.jpg", width=100)
st.sidebar.title("AI-Solutions Dashboard")

# Dashboard sections
dashboard_section = st.sidebar.radio(
    "Dashboard Section",
    ["Overview", "Geographic Analysis", "Sales Performance", "User Engagement", "Technical Health", "Raw Data"]
)

# Generate or load data
if 'logs_df' not in st.session_state:
    st.session_state.logs_df = generate_web_logs(5000)
    st.session_state.sales_df, st.session_state.summary_df = generate_sales_data(st.session_state.logs_df)

# Add filters
st.sidebar.header("Filters")

# Date filter
date_min = pd.to_datetime(st.session_state.logs_df['date']).min().date()
date_max = pd.to_datetime(st.session_state.logs_df['date']).max().date()
selected_date_range = st.sidebar.date_input(
    "Date Range",
    [date_min, date_max],
    date_min,
    date_max
)

# Country filter
countries = ["All"] + sorted(st.session_state.logs_df['country'].unique().tolist())
selected_country = st.sidebar.selectbox("Country", countries)

# Sales rep filter (when applicable)
if not st.session_state.sales_df.empty:
    sales_reps = ["All"] + sorted(st.session_state.sales_df['sales_rep'].unique().tolist())
    selected_sales_rep = st.sidebar.selectbox("Sales Representative", sales_reps)

# View level for sales performance
view_level = st.sidebar.radio("View Level", ["Team Overview", "Individual Performance"])

# Apply filters to dataframes
filtered_logs = st.session_state.logs_df.copy()

# Apply date filter
filtered_logs = filtered_logs[
    (pd.to_datetime(filtered_logs['date']).dt.date >= selected_date_range[0]) & 
    (pd.to_datetime(filtered_logs['date']).dt.date <= selected_date_range[1])
]

# Apply country filter
if selected_country != "All":
    filtered_logs = filtered_logs[filtered_logs['country'] == selected_country]

# Filter sales data
if not st.session_state.sales_df.empty:
    filtered_sales = st.session_state.sales_df.copy()
    
    # Apply date filter
    filtered_sales = filtered_sales[
        (pd.to_datetime(filtered_sales['date']).dt.date >= selected_date_range[0]) & 
        (pd.to_datetime(filtered_sales['date']).dt.date <= selected_date_range[1])
    ]
    
    # Apply country filter
    if selected_country != "All":
        filtered_sales = filtered_sales[filtered_sales['country'] == selected_country]
    
    # Apply sales rep filter
    if 'selected_sales_rep' in locals() and selected_sales_rep != "All":
        filtered_sales = filtered_sales[filtered_sales['sales_rep'] == selected_sales_rep]

# Create regenerate data button
if st.sidebar.button("Regenerate Test Data"):
    st.session_state.logs_df = generate_web_logs(5000)
    st.session_state.sales_df, st.session_state.summary_df = generate_sales_data(st.session_state.logs_df)
    st.experimental_rerun()

# Main content area
if dashboard_section == "Overview":
    st.title("📊 AI-Solutions Overview Dashboard")
    
    # High-level KPIs
    st.header("Key Performance Indicators (KPIs)")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    # Total visitors (unique IPs)
    total_visitors = filtered_logs['ip'].nunique()
    kpi_col1.metric("Total Visitors", f"{total_visitors:,}")
    
    # Demo conversion rate
    demo_requests = filtered_logs[filtered_logs['page_type'] == 'Demo Request'].shape[0]
    total_content_views = filtered_logs[filtered_logs['page_type'] == 'Content Page'].shape[0]
    conversion_rate = (demo_requests / total_content_views * 100) if total_content_views > 0 else 0
    kpi_col2.metric("Demo Conversion Rate", f"{conversion_rate:.2f}%")
    
    # Tool engagement rate
    tool_usage = filtered_logs[(filtered_logs['page_type'] == 'AI Assistant') | 
                              (filtered_logs['page_type'] == 'Prototype Tool')].shape[0]
    tool_engagement_rate = (tool_usage / filtered_logs.shape[0] * 100)
    kpi_col3.metric("Tool Engagement Rate", f"{tool_engagement_rate:.2f}%")
    
    # Geographic reach
    geographic_reach = filtered_logs['country'].nunique()
    kpi_col4.metric("Geographic Reach", f"{geographic_reach} countries")
    
    st.markdown("---")
    
    # Row 1: Maps and Conversion
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Geographic Distribution")
        country_counts = filtered_logs['country'].value_counts().reset_index()
        country_counts.columns = ['country', 'count']
        
        fig = px.choropleth(
            country_counts,
            locations='country',
            locationmode='country names',
            color='count',
            color_continuous_scale='Viridis',
            hover_name='country',
            title='Traffic Distribution by Country'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Conversion Funnel")
        
        # Calculate funnel metrics
        content_views = filtered_logs[filtered_logs['page_type'] == 'Content Page'].shape[0]
        tool_usage = filtered_logs[(filtered_logs['page_type'] == 'AI Assistant') | 
                                  (filtered_logs['page_type'] == 'Prototype Tool')].shape[0]
        demo_requests = filtered_logs[filtered_logs['page_type'] == 'Demo Request'].shape[0]
        
        funnel_data = pd.DataFrame({
            'Stage': ['Content Views', 'Tool Usage', 'Demo Requests'],
            'Count': [content_views, tool_usage, demo_requests]
        })
        
        fig = px.funnel(
            funnel_data,
            x='Count',
            y='Stage'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Tool Usage and Job Interest
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tool Usage Breakdown")
        tool_data = filtered_logs[filtered_logs['page_type'].isin(['AI Assistant', 'Prototype Tool'])]
        tool_counts = tool_data['page_type'].value_counts().reset_index()
        tool_counts.columns = ['Tool', 'Count']
        
        fig = px.pie(
            tool_counts,
            values='Count',
            names='Tool',
            title='Tool Usage Distribution',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Job Interest Analysis")
        job_data = filtered_logs[filtered_logs['job_type'].notna()]
        if not job_data.empty:
            job_counts = job_data['job_type'].value_counts().reset_index()
            job_counts.columns = ['Job Type', 'Count']
            
            fig = px.bar(
                job_counts,
                x='Job Type',
                y='Count',
                title='Job Interest by Category',
                color='Job Type',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No job interest data available for the selected filters.")
    
    # Row 3: Traffic patterns and status codes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Traffic Pattern")
        hourly_traffic = filtered_logs.groupby('hour').size().reset_index()
        hourly_traffic.columns = ['Hour', 'Count']
        
        fig = px.bar(
            hourly_traffic,
            x='Hour',
            y='Count',
            title='Traffic by Hour of Day',
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Status Code Distribution")
        status_counts = filtered_logs['status'].value_counts().reset_index()
        status_counts.columns = ['Status Code', 'Count']
        
        # Map status codes to colors
        status_colors = {
            200: 'green',
            304: 'blue',
            404: 'red',
            500: 'darkred'
        }
        
        status_colors_list = [status_colors.get(code, 'gray') for code in status_counts['Status Code']]
        
        fig = px.pie(
            status_counts,
            values='Count',
            names='Status Code',
            title='HTTP Status Code Distribution',
            color='Status Code',
            color_discrete_map={str(k): v for k, v in status_colors.items()}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
#--------------------------------------------------------------------------------------------------------------------------------------------------
elif dashboard_section == "Geographic Analysis":
    st.title("🌎 Geographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Traffic by Country")
        country_traffic = filtered_logs['country'].value_counts().reset_index()
        country_traffic.columns = ['Country', 'Visits']
        
        fig = px.bar(
            country_traffic.head(10),
            x='Country',
            y='Visits',
            title='Top 10 Countries by Traffic',
            color='Country'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Conversion Rate by Country")
        
        country_conversions = {}
        for country in filtered_logs['country'].unique():
            country_data = filtered_logs[filtered_logs['country'] == country]
            content_views = country_data[country_data['page_type'] == 'Content Page'].shape[0]
            demo_requests = country_data[country_data['page_type'] == 'Demo Request'].shape[0]
            
            if content_views > 0:
                conversion_rate = (demo_requests / content_views) * 100
                country_conversions[country] = conversion_rate
        
        conversion_df = pd.DataFrame({
            'Country': list(country_conversions.keys()),
            'Conversion Rate (%)': list(country_conversions.values())
        }).sort_values('Conversion Rate (%)', ascending=False)
        
        fig = px.bar(
            conversion_df.head(10),
            x='Country',
            y='Conversion Rate (%)',
            title='Top 10 Countries by Conversion Rate',
            color='Conversion Rate (%)',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # World map with detailed view
    st.subheader("Interactive World Map")
    
    country_metrics = {}
    for country in filtered_logs['country'].unique():
        country_data = filtered_logs[filtered_logs['country'] == country]
        
        country_metrics[country] = {
            'Traffic': country_data.shape[0],
            'Demo Requests': country_data[country_data['page_type'] == 'Demo Request'].shape[0],
            'AI Assistant Usage': country_data[country_data['page_type'] == 'AI Assistant'].shape[0],
            'Prototype Tool Usage': country_data[country_data['page_type'] == 'Prototype Tool'].shape[0],
            'Job Interest': country_data[country_data['job_type'].notna()].shape[0]
        }
    
    metrics_df = pd.DataFrame.from_dict(country_metrics, orient='index').reset_index()
    metrics_df.columns = ['Country', 'Traffic', 'Demo Requests', 'AI Assistant Usage', 'Prototype Tool Usage', 'Job Interest']
    
    fig = px.choropleth(
        metrics_df,
        locations='Country',
        locationmode='country names',
        color='Traffic',
        hover_data=['Traffic', 'Demo Requests', 'AI Assistant Usage', 'Prototype Tool Usage', 'Job Interest'],
        color_continuous_scale='Viridis',
        projection='natural earth'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Job interest by country
    st.subheader("Job Interest by Country")
    
    job_country_data = filtered_logs[filtered_logs['job_type'].notna()]
    if not job_country_data.empty:
        job_country_pivot = pd.crosstab(job_country_data['country'], job_country_data['job_type'])
        
        fig = px.imshow(
            job_country_pivot,
            labels=dict(x="Job Type", y="Country", color="Count"),
            x=job_country_pivot.columns,
            y=job_country_pivot.index,
            color_continuous_scale='Viridis',
            aspect="auto"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No job interest data available for the selected filters.")

#--------------------------------------------------------------------------------------------------------------------------

elif dashboard_section == "Sales Performance":
    if not st.session_state.sales_df.empty and not st.session_state.summary_df.empty:
        st.title("💼 Sales Performance Dashboard")
        
        if view_level == "Team Overview":
            # Team level KPIs
            st.header("Team Performance")
            
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            # Total revenue
            total_revenue = filtered_sales['revenue'].sum()
            kpi_col1.metric("Total Revenue", f"${total_revenue:,.2f}")
            
            # Average deal size
            avg_deal = filtered_sales['revenue'].mean()
            kpi_col2.metric("Average Deal Size", f"${avg_deal:,.2f}")
            
            # Average lead time
            avg_lead_time = filtered_sales['lead_time_days'].mean()
            kpi_col3.metric("Avg Lead Time (Days)", f"{avg_lead_time:.1f}")
            
            # Deals closed
            deals_closed = filtered_sales.shape[0]
            kpi_col4.metric("Deals Closed", deals_closed)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sales Performance vs Target")
                
                # Calculate aggregated performance
                team_data = st.session_state.summary_df.copy()
                
                status_colors = {
                    "Exceeding Target": "green",
                    "Meeting Target": "blue",
                    "Near Target": "gold",
                    "Below Target": "red"
                }
                
                fig = go.Figure()
                
                for i, row in team_data.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row['sales_rep']],
                        y=[row['performance']],
                        name=row['sales_rep'],
                        marker_color=status_colors[row['status']],
                        text=f"{row['performance']:.1f}%",
                        textposition='auto'
                    ))
                
                # Add target line
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(team_data)-0.5,
                    y0=100,
                    y1=100,
                    line=dict(color="black", width=2, dash="dot")
                )
                
                fig.update_layout(
                    title="Team Performance vs Target (%)",
                    yaxis_title="Performance (%)",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Legend
                legend_col1, legend_col2 = st.columns(2)
                legend_col1.markdown("**Performance Legend:**")
                legend_col1.markdown("🟢 **Exceeding Target** (≥110%)")
                legend_col1.markdown("🔵 **Meeting Target** (100-109%)")
                
                legend_col2.markdown("&nbsp;")
                legend_col2.markdown("🟡 **Near Target** (85-99%)")
                legend_col2.markdown("🔴 **Below Target** (<85%)")
            
            with col2:
                st.subheader("Revenue by Product")
                
                product_revenue = filtered_sales.groupby('product')['revenue'].sum().reset_index()
                
                fig = px.pie(
                    product_revenue,
                    values='revenue',
                    names='product',
                    title='Revenue Distribution by Product',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Row 2
            col1, col2, = st.columns(2)
            
            with col1:
                st.subheader("Sales by Country")
                
                country_sales = filtered_sales.groupby('country')['revenue'].sum().reset_index()
                country_sales = country_sales.sort_values('revenue', ascending=False)
                
                fig = px.bar(
                    country_sales.head(10),
                    x='country',
                    y='revenue',
                    title='Top 10 Countries by Revenue',
                    color='country'
                )
                
                fig.update_layout(
                    xaxis_title="Country",
                    yaxis_title="Revenue ($)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Sales Trend Over Time")
                
                daily_sales = filtered_sales.groupby('date')['revenue'].sum().reset_index()
                daily_sales['date'] = pd.to_datetime(daily_sales['date'])
                daily_sales = daily_sales.sort_values('date')
                
                fig = px.line(
                    daily_sales,
                    x='date',
                    y='revenue',
                    title='Daily Sales Revenue'
                )
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Revenue ($)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Pipeline velocity
            st.subheader("Sales Pipeline Velocity")
            
            lead_time_product = filtered_sales.groupby('product')['lead_time_days'].mean().reset_index()
            lead_time_product.columns = ['Product', 'Average Days to Close']
            
            fig = px.bar(
                lead_time_product,
                x='Product',
                y='Average Days to Close',
                title='Average Days to Close by Product',
                color='Product'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Individual Performance view
            if 'selected_sales_rep' in locals() and selected_sales_rep != "All":
                rep_data = st.session_state.summary_df[st.session_state.summary_df['sales_rep'] == selected_sales_rep]
                if not rep_data.empty:
                    rep_row = rep_data.iloc[0]
                    
                    st.header(f"Individual Performance: {selected_sales_rep}")
                    
                    # Performance status with color
                    status_color = rep_row['color']
                    st.markdown(f"<h3 style='color: {status_color};'>Status: {rep_row['status']} ({rep_row['performance']:.1f}%)</h3>", unsafe_allow_html=True)
                    
                    # KPIs for this rep
                    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                    
                    rep_filtered_sales = filtered_sales[filtered_sales['sales_rep'] == selected_sales_rep]
                    
                    # Total sales
                    rep_total_sales = rep_filtered_sales['revenue'].sum()
                    kpi_col1.metric("Total Sales", f"${rep_total_sales:,.2f}")
                    
                    # Target
                    rep_target = rep_row['target']
                    kpi_col2.metric("Target", f"${rep_target:,.2f}")
                    
                    # Number of deals
                    rep_deals = rep_filtered_sales.shape[0]
                    kpi_col3.metric("Deals Closed", rep_deals)
                    
                    # Average deal size
                    rep_avg_deal = rep_filtered_sales['revenue'].mean() if rep_deals > 0 else 0
                    kpi_col4.metric("Avg Deal Size", f"${rep_avg_deal:,.2f}")
                    
                    st.markdown("---")
                    
                    # Row 1
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Performance Comparison")
                        
                        # Get all reps data for comparison
                        all_reps = st.session_state.summary_df.copy()
                        all_reps = all_reps.sort_values('performance', ascending=False)
                        
                        # Highlight the selected rep
                        colors = ['lightgrey'] * len(all_reps)
                        for i, rep in enumerate(all_reps['sales_rep']):
                            if rep == selected_sales_rep:
                                colors[i] = status_color
                        
                        fig = px.bar(
                            all_reps,
                            x='sales_rep',
                            y='performance',
                            title='Performance Comparison (All Reps)',
                            text=all_reps['performance'].apply(lambda x: f"{x:.1f}%")
                        )
                        
                        fig.update_traces(marker_color=colors, textposition='auto')
                        
                        # Add target line
                        fig.add_shape(
                            type="line",
                            x0=-0.5,
                            x1=len(all_reps)-0.5,
                            y0=100,
                            y1=100,
                            line=dict(color="black", width=2, dash="dot")
                        )
                        
                        fig.update_layout(
                            xaxis_title="Sales Rep",
                            yaxis_title="Performance (%)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Product Distribution")
                        
                        rep_product_sales = rep_filtered_sales.groupby('product')['revenue'].sum().reset_index()
                        
                        fig = px.pie(
                            rep_product_sales,
                            values='revenue',
                            names='product',
                            title=f'{selected_sales_rep}\'s Revenue by Product',
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Row 2
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Sales by Country")
                        
                        rep_country_sales = rep_filtered_sales.groupby('country')['revenue'].sum().reset_index()
                        rep_country_sales = rep_country_sales.sort_values('revenue', ascending=False)
                        
                        fig = px.bar(
                            rep_country_sales.head(10),
                            x='country',
                            y='revenue',
                            title=f'{selected_sales_rep}\'s Top Countries by Revenue',
                            color='country'
                        )
                        
                        fig.update_layout(
                            xaxis_title="Country",
                            yaxis_title="Revenue ($)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Sales Trend")
                        
                        rep_daily_sales = rep_filtered_sales.groupby('date')['revenue'].sum().reset_index()
                        rep_daily_sales['date'] = pd.to_datetime(rep_daily_sales['date'])
                        rep_daily_sales = rep_daily_sales.sort_values('date')
                        
                        fig = px.line(
                            rep_daily_sales,
                            x='date',
                            y='revenue',
                            title=f'{selected_sales_rep}\'s Daily Sales Revenue'
                        )
                        
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Revenue ($)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Lead time analysis
                    st.subheader("Lead Time Analysis")
                    
                    rep_lead_time = rep_filtered_sales.groupby('source')['lead_time_days'].mean().reset_index()
                    rep_lead_time.columns = ['Source', 'Average Days to Close']
                    
                    fig = px.bar(
                        rep_lead_time,
                        x='Source',
                        y='Average Days to Close',
                        title=f'{selected_sales_rep}\'s Average Days to Close by Source',
                        color='Source'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning(f"No data available for {selected_sales_rep} with the current filters.")
            else:
                st.info("Please select a specific sales representative to view individual performance.")
    else:
        st.warning("No sales data available. Please regenerate the test data.")

elif dashboard_section == "User Engagement":
    st.title("👥 User Engagement Analysis")
    
    # Tool engagement metrics
    st.header("Tool Engagement Metrics")
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    # AI Assistant usage
    ai_assistant_usage = filtered_logs[filtered_logs['page_type'] == 'AI Assistant'].shape[0]
    kpi_col1.metric("AI Assistant Usage", ai_assistant_usage)
    
    # Prototype tool usage
    prototype_usage = filtered_logs[filtered_logs['page_type'] == 'Prototype Tool'].shape[0]
    kpi_col2.metric("Prototype Tool Usage", prototype_usage)
    
    # Tool engagement rate
    tool_engagement = (ai_assistant_usage + prototype_usage) / filtered_logs.shape[0] * 100
    kpi_col3.metric("Tool Engagement Rate", f"{tool_engagement:.2f}%")
    
    #st.markdown("---")
    
    # Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tool Usage by Day")
        
        # Add date column if not present
        if 'date_day' not in filtered_logs.columns:
            filtered_logs['date_day'] = pd.to_datetime(filtered_logs['date']).dt.date
        
        # Get daily tool usage
        daily_tool_usage = filtered_logs[filtered_logs['page_type'].isin(['AI Assistant', 'Prototype Tool'])]
        daily_tool_pivot = pd.crosstab(daily_tool_usage['date_day'], daily_tool_usage['page_type'])
        
        # Reset index for plotting
        daily_tool_pivot = daily_tool_pivot.reset_index()
        
        # Melt the dataframe for easier plotting
        daily_tool_melted = pd.melt(
            daily_tool_pivot, 
            id_vars=['date_day'],
            value_vars=['AI Assistant', 'Prototype Tool'],
            var_name='Tool',
            value_name='Usage'
        )
        
        fig = px.line(
            daily_tool_melted,
            x='date_day',
            y='Usage',
            color='Tool',
            title='Daily Tool Usage',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Interactions"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tool Usage by Country")
        
        # Get country-wise tool usage
        country_tool_usage = filtered_logs[filtered_logs['page_type'].isin(['AI Assistant', 'Prototype Tool'])]
        country_tool_pivot = pd.crosstab(country_tool_usage['country'], country_tool_usage['page_type'])
        
        # Reset index for plotting
        country_tool_pivot = country_tool_pivot.reset_index()
        
        # Melt the dataframe for easier plotting
        country_tool_melted = pd.melt(
            country_tool_pivot,
            id_vars=['country'],
            value_vars=['AI Assistant', 'Prototype Tool'],
            var_name='Tool',
            value_name='Usage'
        )
        
        # Sort by total usage
        country_totals = country_tool_melted.groupby('country')['Usage'].sum().reset_index()
        top_countries = country_totals.sort_values('Usage', ascending=False).head(10)['country'].tolist()
        
        country_tool_filtered = country_tool_melted[country_tool_melted['country'].isin(top_countries)]
        
        fig = px.bar(
            country_tool_filtered,
            x='country',
            y='Usage',
            color='Tool',
            title='Tool Usage by Top 10 Countries',
            barmode='group'
        )
        
        fig.update_layout(
            xaxis_title="Country",
            yaxis_title="Number of Interactions"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tool Usage by Hour")
        
        # Get hourly tool usage
        hourly_tool_usage = filtered_logs[filtered_logs['page_type'].isin(['AI Assistant', 'Prototype Tool'])]
        hourly_tool_pivot = pd.crosstab(hourly_tool_usage['hour'], hourly_tool_usage['page_type'])
        
        # Reset index for plotting
        hourly_tool_pivot = hourly_tool_pivot.reset_index()
        
        # Melt the dataframe for easier plotting
        hourly_tool_melted = pd.melt(
            hourly_tool_pivot,
            id_vars=['hour'],
            value_vars=['AI Assistant', 'Prototype Tool'],
            var_name='Tool',
            value_name='Usage'
        )
        
        fig = px.line(
            hourly_tool_melted,
            x='hour',
            y='Usage',
            color='Tool',
            title='Tool Usage by Hour of Day',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Number of Interactions",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tool to Demo Conversion")
        
        # Analyze if tool users are more likely to request demos
        tool_users = filtered_logs[filtered_logs['page_type'].isin(['AI Assistant', 'Prototype Tool'])]['ip'].unique()
        
        # Get demo requests from tool users
        tool_user_demos = filtered_logs[
            (filtered_logs['page_type'] == 'Demo Request') & 
            (filtered_logs['ip'].isin(tool_users))
        ].shape[0]
        
        # Get demo requests from non-tool users
        non_tool_user_demos = filtered_logs[
            (filtered_logs['page_type'] == 'Demo Request') & 
            (~filtered_logs['ip'].isin(tool_users))
        ].shape[0]
        
        # Create comparison dataframe
        conversion_comparison = pd.DataFrame({
            'User Type': ['Tool Users', 'Non-Tool Users'],
            'Demo Requests': [tool_user_demos, non_tool_user_demos]
        })
        
        fig = px.bar(
            conversion_comparison,
            x='User Type',
            y='Demo Requests',
            title='Demo Requests: Tool Users vs Non-Tool Users',
            color='User Type'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Job interest analysis
    st.header("Job Interest Analysis")
    
    job_data = filtered_logs[filtered_logs['job_type'].notna()]
    
    if not job_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Job Interest Distribution")
            
            job_counts = job_data['job_type'].value_counts().reset_index()
            job_counts.columns = ['Job Type', 'Count']
            
            fig = px.pie(
                job_counts,
                values='Count',
                names='Job Type',
                title='Job Interest Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Job Interest Trend")
            
            # Add date column if not present
            if 'date_day' not in job_data.columns:
                job_data['date_day'] = pd.to_datetime(job_data['date']).dt.date
            
            # Get daily job interest
            daily_job_interest = pd.crosstab(job_data['date_day'], job_data['job_type'])
            
            # Reset index for plotting
            daily_job_interest = daily_job_interest.reset_index()
            
            # Melt the dataframe for easier plotting
            daily_job_melted = pd.melt(
                daily_job_interest,
                id_vars=['date_day'],
                var_name='Job Type',
                value_name='Interest'
            )
            
            fig = px.line(
                daily_job_melted,
                x='date_day',
                y='Interest',
                color='Job Type',
                title='Daily Job Interest by Type',
                markers=True
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Inquiries"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No job interest data available for the selected filters.")

elif dashboard_section == "Technical Health":
    st.title("🔧 Technical Health Dashboard")
    
    # HTTP status code metrics
    st.header("HTTP Status Code Analysis")
    
    # Calculate status code metrics
    status_counts = filtered_logs['status'].value_counts()
    total_requests = filtered_logs.shape[0]
    
    success_rate = (status_counts.get(200, 0) / total_requests) * 100 if total_requests > 0 else 0
    error_rate = (status_counts.get(404, 0) + status_counts.get(500, 0)) / total_requests * 100 if total_requests > 0 else 0
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    kpi_col1.metric("Total Requests", total_requests)
    kpi_col2.metric("Success Rate", f"{success_rate:.2f}%")
    kpi_col3.metric("Error Rate", f"{error_rate:.2f}%")
    
    st.markdown("---")
    
    # Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Status Code Distribution")
        
        status_df = pd.DataFrame({
            'Status Code': status_counts.index,
            'Count': status_counts.values
        })
        
        # Map status codes to descriptions
        status_descriptions = {
            200: "OK",
            304: "Not Modified",
            404: "Not Found",
            500: "Server Error"
        }
        
        status_df['Description'] = status_df['Status Code'].map(status_descriptions)
        status_df['Label'] = status_df['Status Code'].astype(str) + " - " + status_df['Description']
        
        # Map status codes to colors
        status_colors = {
            200: 'green',
            304: 'blue',
            404: 'orange',
            500: 'red'
        }
        
        status_df['Color'] = status_df['Status Code'].map(status_colors)
        
        fig = px.pie(
            status_df,
            values='Count',
            names='Label',
            title='HTTP Status Code Distribution',
            color='Status Code',
            color_discrete_map={str(k): v for k, v in status_colors.items()}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Error Rate by Day")
        
        # Add date column if not present
        if 'date_day' not in filtered_logs.columns:
            filtered_logs['date_day'] = pd.to_datetime(filtered_logs['date']).dt.date
        
        # Calculate daily error rates
        daily_status = filtered_logs.groupby(['date_day', 'status']).size().reset_index(name='count')
        daily_total = filtered_logs.groupby('date_day').size().reset_index(name='total')
        
        # Merge to get daily totals
        daily_status = daily_status.merge(daily_total, on='date_day')
        
        # Calculate percentages
        daily_status['percentage'] = (daily_status['count'] / daily_status['total']) * 100
        
        # Filter for error codes
        error_codes = [404, 500]
        daily_errors = daily_status[daily_status['status'].isin(error_codes)]
        
        fig = px.line(
            daily_errors,
            x='date_day',
            y='percentage',
            color='status',
            title='Daily Error Rate by Status Code',
            markers=True,
            labels={'percentage': 'Error Rate (%)', 'date_day': 'Date', 'status': 'Status Code'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Errors by Page")
        
        # Calculate errors by page
        page_errors = filtered_logs[filtered_logs['status'].isin([404, 500])].groupby('page').size().reset_index(name='error_count')
        page_errors = page_errors.sort_values('error_count', ascending=False).head(10)
        
        fig = px.bar(
            page_errors,
            x='page',
            y='error_count',
            title='Top 10 Pages with Errors',
            color='error_count',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            xaxis_title="Page",
            yaxis_title="Error Count",
            xaxis={'categoryorder':'total descending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Errors by Country")
        
        # Calculate errors by country
        country_errors = filtered_logs[filtered_logs['status'].isin([404, 500])].groupby('country').size().reset_index(name='error_count')
        country_total = filtered_logs.groupby('country').size().reset_index(name='total_count')
        
        # Merge for calculating percentages
        country_error_rates = country_errors.merge(country_total, on='country')
        country_error_rates['error_rate'] = (country_error_rates['error_count'] / country_error_rates['total_count']) * 100
        country_error_rates = country_error_rates.sort_values('error_rate', ascending=False).head(10)
        
        fig = px.bar(
            country_error_rates,
            x='country',
            y='error_rate',
            title='Top 10 Countries by Error Rate',
            color='error_rate',
            color_continuous_scale='Reds',
            text=country_error_rates['error_rate'].apply(lambda x: f"{x:.2f}%")
        )
        
        fig.update_layout(
            xaxis_title="Country",
            yaxis_title="Error Rate (%)",
            xaxis={'categoryorder':'total descending'}
        )
        
        fig.update_traces(textposition='auto')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Error distribution by hour
    st.subheader("Error Distribution by Hour")
    
    # Calculate hourly error rates
    hourly_status = filtered_logs.groupby(['hour', 'status']).size().reset_index(name='count')
    hourly_total = filtered_logs.groupby('hour').size().reset_index(name='total')
    
    # Merge to get hourly totals
    hourly_status = hourly_status.merge(hourly_total, on='hour')
    
    # Calculate percentages
    hourly_status['percentage'] = (hourly_status['count'] / hourly_status['total']) * 100
    
    # Filter for error codes
    hourly_errors = hourly_status[hourly_status['status'].isin(error_codes)]
    
    fig = px.line(
        hourly_errors,
        x='hour',
        y='percentage',
        color='status',
        title='Hourly Error Rate by Status Code',
        markers=True,
        labels={'percentage': 'Error Rate (%)', 'hour': 'Hour of Day', 'status': 'Status Code'}
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif dashboard_section == "Raw Data":
    st.title("📋 Raw Data")
    
    tab1, tab2 = st.tabs(["Web Server Logs", "Sales Data"])
    
    with tab1:
        st.header("Web Server Logs")
        st.dataframe(filtered_logs)
        
        # Download CSV link
        st.markdown(get_download_link(filtered_logs, "ai_solutions_web_logs.csv", "Download Web Logs CSV"), unsafe_allow_html=True)
    
    with tab2:
        if not st.session_state.sales_df.empty:
            st.header("Sales Data")
            st.dataframe(filtered_sales)
            
            # Download CSV link
            st.markdown(get_download_link(filtered_sales, "ai_solutions_sales_data.csv", "Download Sales Data CSV"), unsafe_allow_html=True)
        else:
            st.info("No sales data available. Please regenerate the test data.")
            
elif dashboard_section == "Navigation Flow":
    st.title("🔄 User Navigation Flow Analysis")
    
    # Generate navigation paths for IPs with multiple page views
    st.header("User Journey Analysis")
    
    # Get IPs with multiple page views
    ip_counts = filtered_logs.groupby('ip').size()
    multi_view_ips = ip_counts[ip_counts > 1].index.tolist()
    
    # Create user journey data
    user_journeys = []
    for ip in multi_view_ips[:500]:  # Limit to first 500 for performance
        journey = filtered_logs[filtered_logs['ip'] == ip].sort_values('timestamp')
        
        if len(journey) > 1:
            path = " → ".join(journey['page_type'].tolist())
            country = journey['country'].iloc[0]
            pages = len(journey)
            
            # Check for conversion (if journey ended with demo request)
            converted = 'Demo Request' in journey['page_type'].values
            
            user_journeys.append({
                'ip': ip,
                'country': country,
                'path': path,
                'pages': pages,
                'converted': converted
            })
    
    journey_df = pd.DataFrame(user_journeys)
    
    # Most common paths
    st.subheader("Most Common User Paths")
    
    path_counts = journey_df['path'].value_counts().reset_index()
    path_counts.columns = ['Path', 'Count']
    path_counts = path_counts.head(10)
    
    fig = px.bar(
        path_counts,
        x='Count',
        y='Path',
        title='Top 10 Navigation Paths',
        orientation='h',
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        yaxis_title=""
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Conversion Sankey Diagram
    st.subheader("Navigation Flow and Conversion")
    
    # Calculate transition frequencies
    sankey_data = {'source': [], 'target': [], 'value': [], 'source_country': []}
    
    # Process most frequent navigation paths
    for _, row in journey_df.head(200).iterrows():
        path_steps = row['path'].split(' → ')
        country = row['country']
        
        if len(path_steps) > 1:
            for i in range(len(path_steps) - 1):
                source = path_steps[i]
                target = path_steps[i + 1]
                
                sankey_data['source'].append(source)
                sankey_data['target'].append(target)
                sankey_data['value'].append(1)
                sankey_data['source_country'].append(country)
    
    # Convert to DataFrame
    sankey_df = pd.DataFrame(sankey_data)
    
    # Aggregate the flows
    sankey_agg = sankey_df.groupby(['source', 'target']).agg({'value': 'sum'}).reset_index()
    
    # Create a list of all unique nodes
    all_nodes = list(set(sankey_agg['source'].tolist() + sankey_agg['target'].tolist()))
    
    # Create a mapping of node name to node ID
    node_ids = {node: i for i, node in enumerate(all_nodes)}
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color="blue"
        ),
        link=dict(
            source=[node_ids[s] for s in sankey_agg['source']],
            target=[node_ids[t] for t in sankey_agg['target']],
            value=sankey_agg['value']
        )
    )])
    
    fig.update_layout(
        title_text="User Navigation Flow",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Path analysis by conversion
    st.subheader("Path Analysis by Conversion")
    
    # Average path length by conversion
    path_length_by_conversion = journey_df.groupby('converted')['pages'].mean().reset_index()
    path_length_by_conversion.columns = ['Converted', 'Average Path Length']
    path_length_by_conversion['Converted'] = path_length_by_conversion['Converted'].map({True: 'Converted', False: 'Not Converted'})
    
    fig = px.bar(
        path_length_by_conversion,
        x='Converted',
        y='Average Path Length',
        title='Average Path Length by Conversion Status',
        color='Converted',
        color_discrete_map={'Converted': 'green', 'Not Converted': 'red'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Entry and exit pages
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Common Entry Points")
        
        entry_pages = []
        for _, row in journey_df.iterrows():
            path_steps = row['path'].split(' → ')
            if path_steps:
                entry_pages.append(path_steps[0])
        
        entry_df = pd.DataFrame({'Entry Page': entry_pages})
        entry_counts = entry_df['Entry Page'].value_counts().reset_index()
        entry_counts.columns = ['Entry Page', 'Count']
        
        fig = px.pie(
            entry_counts.head(5),
            values='Count',
            names='Entry Page',
            title='Top 5 Entry Pages',
            hole=0.4
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Common Exit Points")
        
        exit_pages = []
        for _, row in journey_df.iterrows():
            path_steps = row['path'].split(' → ')
            if path_steps:
                exit_pages.append(path_steps[-1])
        
        exit_df = pd.DataFrame({'Exit Page': exit_pages})
        exit_counts = exit_df['Exit Page'].value_counts().reset_index()
        exit_counts.columns = ['Exit Page', 'Count']
        
        fig = px.pie(
            exit_counts.head(5),
            values='Count',
            names='Exit Page',
            title='Top 5 Exit Pages',
            hole=0.4
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**AI-Solutions Dashboard** | Developed for Sales Performance Analysis")
st.markdown("Data refreshed: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))