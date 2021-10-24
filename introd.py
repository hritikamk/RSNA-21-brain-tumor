import streamlit as st


def intro():
    st.balloons()
    col1, col2 = st.columns(2)
    col1.image(
        "https://storage.googleapis.com/kaggle-media/competitions/RSNA-2021/image2.png")
    col2.markdown("""
    A malignant tumor in the brain is a life-threatening condition. Known as glioblastoma, it's both the most common form of brain cancer in adults and the one with the worst prognosis, with median survival being less than a year. 
    """)
    st.markdown("""The presence of a specific genetic sequence in the tumor known as MGMT promoter methylation has been shown to be a favorable prognostic factor and a strong predictor of responsiveness to chemotherapy.
                Currently, genetic analysis of cancer requires surgery to extract a tissue sample. Then it can take several weeks to determine the genetic characterization of the tumor. Depending upon the results and type of initial therapy chosen, a subsequent surgery may be necessary. If an accurate method to predict the genetics of the cancer through imaging(i.e., radiogenomics) alone could be developed, this would potentially minimize the number of surgeries and refine the type of therapy required.
                We'll help brain cancer patients receive less invasive diagnoses and treatments. The introduction of new and customized treatment strategies before surgery has the potential to improve the management, survival, and prospects of patients with brain cancer.
                """)
    return None
