import streamlit as st
import cv2
import numpy as np
import pandas as pd

from image_processing import preprocess_image
from ocr import extract_text
from nlp import extract_information
from ml_model import classify_item


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Receipt AI | Smart Scanner",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# DARK PROFESSIONAL THEME
# =====================================================

st.markdown("""
<style>
    .stApp {
        background: #0b0d12;
        color: #f2f4f8;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 13, 18, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #11141c;
        border-right: 1px solid #262a36;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        background: linear-gradient(135deg, #191d29, #11131a);
        border: 1px solid #303443;
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 35px;
        margin: 0;
        color: #ffffff;
        font-weight: 800;
    }

    .hero p {
        color: #a8afc0;
        font-size: 15px;
        margin-bottom: 0;
    }

    .accent {
        color: #ff4057;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #f4f5f9;
        margin: 8px 0 14px 0;
    }

    .small-muted {
        color: #9ca4b5;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: #151923;
        border: 1px solid #2a3040;
        border-radius: 14px;
        padding: 17px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aab1c2;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    div[data-testid="stFileUploader"] {
        background: #151923;
        border: 1px dashed #454b60;
        border-radius: 15px;
        padding: 12px;
    }

    .stButton > button {
        background: #ff4057;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    .stButton > button:hover {
        background: #e92d47;
        color: white;
        border: none;
    }

    div[data-testid="stTabs"] button {
        color: #aab1c2;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #ff5267;
    }

    hr {
        border-color: #292e3b;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🧾 Receipt<span class="accent">AI</span></h1>
    <p>Smart Receipt & Invoice Scanner</p>
    <p>Digital Image Processing · OCR · NLP · Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## ⚡ ReceiptAI")
    st.caption("INTELLIGENT DOCUMENT ANALYSIS")
    st.divider()

    st.markdown("### Processing pipeline")
    st.markdown("🖼️ **01** — Image processing")
    st.markdown("🔤 **02** — Text recognition")
    st.markdown("🤖 **03** — Information extraction")
    st.markdown("🧠 **04** — Expense classification")

    st.divider()
    st.markdown("### Supported formats")
    st.markdown("`JPG`  `JPEG`  `PNG`")

    st.caption("Built by Ankith Babu S. & Arpitha M.")


# =====================================================
# UPLOAD SECTION
# =====================================================

st.markdown(
    '<div class="section-title">📤 Upload your document</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="small-muted">Upload a clear receipt or invoice image to begin analysis.</p>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a receipt or invoice",
    type=["jpg", "jpeg", "png"],
    help="For best results, use a clear, well-lit image."
)


# =====================================================
# PROCESSING
# =====================================================

if uploaded_file is not None:

    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read this image. Please upload another JPG or PNG file.")

    else:
        st.success(f"Document ready: {uploaded_file.name}")

        preview_col, details_col = st.columns([1.15, 0.85], gap="large")

        with preview_col:
            st.markdown("### 👁️ Original receipt")
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with details_col:
            st.markdown("### 📋 Document details")

            st.metric("Image width", f"{image.shape[1]} px")
            st.metric("Image height", f"{image.shape[0]} px")

            st.caption(
                "Click the button below to run image processing, "
                "OCR, information extraction and ML classification."
            )

        st.divider()

        scan_clicked = st.button(
            "🔍  SCAN & ANALYZE RECEIPT",
            use_container_width=True,
            type="primary"
        )

        if scan_clicked:

            try:
                # -----------------------------------------
                # 1. DIGITAL IMAGE PROCESSING
                # -----------------------------------------

                with st.spinner("🖼️ Processing and enhancing the image..."):
                    resized, processed_image = preprocess_image(image)

                st.success("Image processing completed.")

                with st.expander("View image processing results"):
                    image_col1, image_col2 = st.columns(2)

                    with image_col1:
                        st.markdown("**Resized image**")
                        st.image(
                            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            if len(resized.shape) == 3
                            else resized,
                            use_container_width=True
                        )

                    with image_col2:
                        st.markdown("**Enhanced image for OCR**")
                        st.image(
                            processed_image,
                            use_container_width=True
                        )

                # -----------------------------------------
                # 2. OCR
                # -----------------------------------------

                with st.spinner("🔤 Extracting text with OCR..."):
                    extracted_text = extract_text(processed_image)

                # -----------------------------------------
                # 3. NLP / INFORMATION EXTRACTION
                # -----------------------------------------

                with st.spinner("🤖 Extracting receipt information..."):
                    information = extract_information(extracted_text)

                if not isinstance(information, dict):
                    raise ValueError(
                        "The information extraction function did not return a dictionary."
                    )

                items = information.get("items", []) or []

                # -----------------------------------------
                # 4. ML CLASSIFICATION
                # -----------------------------------------

                ml_results = []

                if items:
                    with st.spinner("🧠 Classifying expense categories..."):
                        for item in items:
                            category = classify_item(item.get("item", ""))

                            ml_results.append({
                                "Item": item.get("item", "Unknown"),
                                "Quantity": item.get("quantity", 1),
                                "Price (₹)": item.get("price", 0),
                                "Expense Category": category
                            })

                st.session_state["receipt_ai_results"] = {
                    "filename": uploaded_file.name,
                    "text": extracted_text or "",
                    "information": information,
                    "items": items,
                    "ml_results": ml_results
                }

            except Exception as error:
                st.error(
                    "Something went wrong during scanning. "
                    "Please check the existing processing functions."
                )
                st.exception(error)


# =====================================================
# DISPLAY RESULTS
# =====================================================

results = st.session_state.get("receipt_ai_results")

if results is not None:

    information = results["information"]
    items = results["items"]
    extracted_text = results["text"]
    ml_results = results["ml_results"]

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Receipt analysis dashboard</div>',
        unsafe_allow_html=True
    )

    total_amount = sum(
        float(item.get("price", 0) or 0)
        for item in items
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Detected items", len(items))

    with metric2:
        st.metric("Total of detected item prices", f"₹{total_amount:,.2f}")

    with metric3:
        st.metric(
            "OCR status",
            "Text found" if extracted_text.strip() else "No text found"
        )

    st.markdown(" ")

    tab1, tab2, tab3 = st.tabs([
        "🧾 Receipt information",
        "🔤 Extracted OCR text",
        "🧠 ML classification"
    ])

    # -----------------------------------------
    # TAB 1: INFORMATION
    # -----------------------------------------

    with tab1:
        st.markdown("### Extracted information")

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.metric("Store", str(information.get("store") or "Not detected"))

        with info_col2:
            st.metric("Date", str(information.get("date") or "Not detected"))

        with info_col3:
            st.metric("Invoice number", str(information.get("invoice") or "Not detected"))

        st.markdown("### 🛒 Purchased items")

        if items:
            item_rows = []

            for item in items:
                item_rows.append({
                    "Item": item.get("item", "Unknown"),
                    "Quantity": item.get("quantity", 1),
                    "Price (₹)": item.get("price", 0)
                })

            item_df = pd.DataFrame(item_rows)
            st.dataframe(item_df, use_container_width=True, hide_index=True)

            csv_data = item_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download item details (CSV)",
                data=csv_data,
                file_name="receipt_items.csv",
                mime="text/csv"
            )

            st.caption(
                "The displayed total is the sum of detected item prices. "
                "It may differ from the printed receipt total if OCR misses items, "
                "taxes, discounts or fees."
            )
        else:
            st.warning(
                "No purchased items were detected. Check the OCR text and "
                "try a clearer image if needed."
            )

    # -----------------------------------------
    # TAB 2: OCR TEXT
    # -----------------------------------------

    with tab2:
        st.markdown("### 🔤 Raw text recognised from the receipt")

        if extracted_text.strip():
            st.text_area(
                "Recognised text",
                value=extracted_text,
                height=320
            )

            st.download_button(
                "⬇️ Download OCR text",
                data=extracted_text,
                file_name="receipt_ocr.txt",
                mime="text/plain"
            )
        else:
            st.warning(
                "No text was recognised. Try a clearer image with better lighting."
            )

    # -----------------------------------------
    # TAB 3: ML CLASSIFICATION
    # -----------------------------------------

    with tab3:
        st.markdown("### 🧠 Expense categorization")

        st.caption(
            "Categories are produced by your existing classify_item() function."
        )

        if ml_results:
            ml_df = pd.DataFrame(ml_results)

            st.dataframe(
                ml_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download classified results (CSV)",
                data=ml_df.to_csv(index=False).encode("utf-8"),
                file_name="receipt_classification.csv",
                mime="text/csv"
            )
        else:
            st.info("No detected items are available for classification.")

    st.divider()

    st.markdown("### ✅ Processing pipeline")

    pipeline_cols = st.columns(4)

    pipeline_labels = [
        "1. Image processing",
        "2. OCR",
        "3. NLP extraction",
        "4. ML classification"
    ]

    for column, label in zip(pipeline_cols, pipeline_labels):
        with column:
            st.success(label)

    st.caption("Receipt analysis finished. Please verify extracted values before using them.")


# =====================================================
# EMPTY STATE
# =====================================================

else:
    st.divider()

    st.markdown("### 🚀 How it works")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("#### 🖼️ DIP")
        st.caption("Prepares and enhances the receipt image.")

    with step2:
        st.markdown("#### 🔤 OCR")
        st.caption("Recognises text from the image.")

    with step3:
        st.markdown("#### 🤖 NLP")
        st.caption("Extracts store, date, invoice and item details.")

    with step4:
        st.markdown("#### 🧠 ML")
        st.caption("Predicts expense categories for detected items.")

    st.info("Upload a receipt image above to get started.")