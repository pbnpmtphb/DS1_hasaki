from wordcloud import WordCloud
from test_preprocessing import process_text, process_special_word, process_postag_thesea, remove_stopwords
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns


# from imblearn.over_sampling import SMOTE


# Load dictionary and list
def load_dictionary(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        lines = file.read().split('\n')
    dictionary = {}
    for line in lines:
        if '\t' in line:  
            key, value = line.split('\t')
            dictionary[key] = str(value)
    return dictionary

def load_list(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        return file.read().split('\n')

# Load resource
emoji_dict = load_dictionary('./pre_processing/emojicon.txt')  
teen_dict = load_dictionary('./pre_processing/teencode.txt')  
english_dict = load_dictionary('./pre_processing/english-vnmese.txt')  
wrong_lst = load_list('./pre_processing/wrong-word.txt')  
stopword_lst = load_list('./pre_processing/vietnamese-stopwords.txt')  


# Load models
with open('tfidf.pkl', 'rb') as file:
    tfidf = pickle.load(file)
with open('logistic_regression_model.pkl', 'rb') as file:
    logistics_regression = pickle.load(file)


#--------------
# Title & Sidebar
st.title("Data Science Project")
st.markdown("# :rainbow[Sentiment Analysis]")

menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Lương Nhã Hoàng Hà & Phạm Bích Nhật""")
st.sidebar.write("""#### Giảng viên hướng dẫn: 
                 Cô Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 
                 14/12/2024""")

#Menu bar
if choice == 'Business Objective':    
    #Business Objective page:
    st.markdown("## Vấn đề kinh doanh")
    multi="""
    ##### Nhận diện, phân loại phản hồi của khách hàng trên website Hasaki.vn.
    ##### Từ những đánh giá của khách hàng, giúp nhãn hàng hiểu khách hàng đánh giá gì về sản phẩm, từ đó đưa ra các kế hoạch cải thiện chất lượng sản phẩm cũng như các dịch vụ đi kèm.
    """
    st.markdown(multi)  
    st.write("""##### Thực hiện: Sử dụng Machine Learning (Random Forest).""")
    st.image("Sentiment-analysis.jpg")

elif choice == 'Build Project':
    # Build Project page
    st.markdown("1. Tiền xử lý dữ liệu ")
    data = pd.read_csv("Danh_gia.csv", encoding='utf-8')
    st.subheader("Build Project")
    st.write("##### 1. Thu thập và đọc data")
    st.dataframe(data.head(10))
    st.image('PhanPhoiRating.png')
    st.write("##### 2. Data preprocessing")
    multi2='''##### 
    Các bước xử lý dữ liệu bao gồm:
    * Chuyển nhận xét từ viết hoa thành viết thường
    * Loại bỏ các kí tự đặc biệt trong câu (dấu, số, stop-word, khoảng trắng)
    * Thay emoji và teen-code bằng từ tiếng việt chuẩn
    * Nối các câu lại với nhau, và ngăn cách giữa các câu bằng dấu chấm
    Phân loại nhận xét thành positive (từ 4 ⭐ ⬆️) và negative (từ 3 ⭐⬇️)
    Over-sampling các nhận xét negative để xử lý mất cân bằng dữ liệu (hơn 70% nhận xét là positive)
    '''
    st.markdown(multi2)
    st.write("##### Output dữ liệu sau khi tiền xử lý")
    Output = pd.read_csv("Comment_df.csv", encoding='utf-8')
    st.dataframe(Output.head(10))

    st.write("##### 3. Build model")

    st.write("##### 3.1. Modeling with traditional machine learning")
    multi3= '''
    models = { \n
    "Naive Bayes": MultinomialNB(),\n
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42) \n
    "Random Forest": RandomForestClassifier(random_state=42), \n
    }
    '''
    st.markdown( multi3)
    st.image('ROC_curve.png')
    st.write("##### Summary: This model is good enough for classification.")
    st.write("##### 3.2. Modeling with ...") # Còn thiếu
    
elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)            
            lines = lines[0]     
            flag = True                          
    if type=="Input":        
        content = st.text_area(label="Input your content:")
        if content!="":
            
            preprocessed_content = process_text(content, emoji_dict, teen_dict, english_dict, wrong_lst)
            preprocessed_content = process_special_word(preprocessed_content)
            preprocessed_content = process_postag_thesea(preprocessed_content)
            preprocessed_content = remove_stopwords(preprocessed_content, stopword_lst)
            
            lines = np.array([preprocessed_content])
            flag = True

    if flag:
        st.write("Processed content:")
        st.code(preprocessed_content)
        x_new = tfidf.transform(lines)
        y_pred_new = logistics_regression.predict(x_new)
        sentiment = "Positive" if y_pred_new[0] == 1 else "Negative"

        # Display Prediction
        st.write("### Prediction:")
        st.code(f"Sentiment: {sentiment}")

