import qrcode
import streamlit as st
from io import BytesIO
from PIL import Image

def qrcreator():
    # Data for the QR code (this could be a URL, text, etc.)
    # data = "https://microsoft.benevity.org/campaigns/82923"

    data = st.text_input("Enter your URL:", "https://www.microsoft.com")
                         
    if st.button("Create QR Code"):
        # Create a QR code instance
        qr = qrcode.QRCode(
            version=1,  # Version controls the size of the QR Code (1 is the smallest)
            error_correction=qrcode.constants.ERROR_CORRECT_L,  # Controls error correction level
            box_size=10,  # Size of each box in the QR code grid
            border=4,  # Thickness of the border (4 is the minimum)
        )

        # Add data to the instance
        qr.add_data(data)
        qr.make(fit=True)  # Fit the data to the size of the QR code

        # Create an image from the QR code instance
        img = qr.make_image(fill="black", back_color="white")

        # Save the image
        img.save("diwalievent2024.png")

        print("QR code generated and saved as 'diwalievent2024.png'")

        #img = Image.open("your_image_path_or_image_object")  # Example PIL Image

        # Convert the image to bytes
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')  # You can change the format if needed
        img_byte_arr = img_byte_arr.getvalue()  # Get the bytes-like object

        st.image(img_byte_arr, use_column_width=True)

        # Convert the image to bytes for downloading
        img_byte_arr_download = BytesIO()
        img.save(img_byte_arr_download, format='PNG')  # Save the image again for download
        img_byte_arr_download.seek(0)

        # Add download button
        st.download_button(
            label="Download image",
            data=img_byte_arr_download,
            file_name="downloaded_image.png",  # You can name it as you want
            mime="image/png"
        )

