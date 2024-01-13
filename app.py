"""
    Testing Model
"""
import streamlit as st
from PIL import Image
import numpy as np
from numpy import asarray
import tensorflow as tf
from keras.models import load_model

st.set_page_config(page_title="Gender Classification", initial_sidebar_state="expanded")


st.markdown(
    "<h1 style='text-align: center; '>Gender Classification</h1>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """
    Load model
    """
    model_ = load_model("./models/models.h5")
    return model_


def preprocess_image(img):
    """
    Melakukan preprocessing image
    """
    gbr = Image.open(img)
    gbr = gbr.convert("RGB")
    gbr = gbr.resize((160, 160))
    gbr = asarray(gbr)
    gbr = np.expand_dims(gbr, axis=0)
    return gbr


def predict_gender(img, models):
    """
    Predict Gender
    """
    yhat = models.predict(img)
    yhat = np.round(yhat.item())
    if yhat == 1:
        return "Male"
    elif yhat == 0:
        return "Female"
    else:
        return "UNRECOGNIZED"


model = load_models()
# with st.sidebar:
# st.write("Input Image")
st.header("Use Your Webcam")


with st.sidebar:
    st.subheader("About")
    # st.markdown(
    #     "harusnya sampe evaluasi aja tapi nanggung jadi tak deploying.cape mikirin desain daripada modelingnya"
    # )
    # st.markdown("Kalau salah prediksi harap dimaklumi model kurang bagus ato mukkk")
    st.markdown(
        "<p>Bisa dikasi bintang jika berkenan <a href='https://github.com/ahdithya/gender-classification'>Gender Classification</a> </p>",
        unsafe_allow_html=True,
    )

with st.container():
    if "image" not in st.session_state:
        st.session_state["image"] = None

    picture = st.camera_input(
        "Take a picture",
        key="webcam",
        help="Tekan Tombol dibawah untuk mengambil gambar.",
    )
    if picture:
        st.session_state["image"] = picture

    if st.session_state["image"]:
        image = preprocess_image(st.session_state["image"])
        YHAT_ = predict_gender(image, model)

        fileName = st.session_state["image"].name

        with st.expander(f"Your Gender is {YHAT_}"):
            st.image(picture)
            if YHAT_ == "Male":
                st.snow()
            elif YHAT_ == "Female":
                st.balloons()
            st.download_button(
                "Download Image", data=st.session_state["image"], file_name=fileName
            )


st.write(
    """
    ## Notes
    Harap dimaklumi tampilan minimalis, lagi malas mikir (gabisa desain)
    
    -hehe
    
    """
)
