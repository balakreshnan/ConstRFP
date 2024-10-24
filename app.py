import streamlit as st

from rfpapp import rfpapp
from about import about
from watertech import watertech
#from rfpsem import rfpsem
from resumeiq import resumeiq
from qrcreator import qrcreator
# from rfpsem2 import agenticrfp

# Set page size
st.set_page_config(
    page_title="Microsoft Construction Copilot",
    page_icon=":rocket:",
    layout="wide",  # or "centered"
    initial_sidebar_state="expanded"  # or "collapsed"
)

# Load your CSS file
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function to load the CSS
load_css("styles.css")

st.logo("images/aeclogofy24.png")
st.sidebar.image("images/aeclogofy24.png", use_column_width=True)

# Sidebar navigation
nav_option = st.sidebar.selectbox("Navigation", ["Home", 
                                                 "RFP Assistant", #"WaterTech",
                                                 "PM Assistant", "Schedule Assistant",
                                                 "Project Status",
                                                 "RFP Agents", "QR Code Generator", 
                                                 # "Agentic RFP",
                                                 "About"])

# Display the selected page
if nav_option == "RFP Assistant":
    rfpapp()
elif nav_option == "PM Assistant":
    rfpapp()
elif nav_option == "Schedule Assistant":
    rfpapp()
elif nav_option == "Project Status":
    rfpapp()
elif nav_option == "QR Code Generator":
    qrcreator()
#elif nav_option == "Agentic RFP":
#    agenticrfp()
#elif nav_option == "WaterTech": 
#    watertech()
#elif nav_option == "RFP Agents":
#    rfpsem()
#elif nav_option == "ResumeIQ":
#    resumeiq()
elif nav_option == "About":
    about()

#st.sidebar.image("microsoft-logo-png-transparent-20.png", use_column_width=True)