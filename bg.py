import streamlit as st
from PIL import Image
import base64

# Function to add background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # Set the background image
    set_background(r"C:\Users\Ramtej\Desktop\Sales_Forcasting\bg.jpg")  # Use raw string for the path

    # Streamlit input
    st.title("Single Input Application")
    user_input = st.number_input("Enter a number:", min_value=0, step=1)

    if user_input is not None:
        st.write(f"You entered: {user_input}")

if __name__ == "__main__":
    main()
