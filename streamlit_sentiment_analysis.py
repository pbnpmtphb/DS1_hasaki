import numpy as np
import pandas as pd
import pickle
import streamlit as st
from test_preprocessing import process_text, process_special_word, process_postag_thesea, remove_stopwords
from wordcloud import WordCloud


# Load a dictionary file as key-value pairs
def load_dictionary(file_path):
    """Load a dictionary from a file with tab-separated key-value pairs."""
    with open(file_path, 'r', encoding="utf8") as file:
        lines = file.read().split('\n')
    dictionary = {}
    for line in lines:
        if '\t' in line:
            key, value = line.split('\t')
            dictionary[key] = str(value)
    return dictionary


# Load a list from a file
def load_list(file_path):
    """Load a list from a file."""
    with open(file_path, 'r', encoding="utf8") as file:
        return file.read().split('\n')


# Load resources only once
emoji_dict = load_dictionary('./emojicon.txt')  
teen_dict = load_dictionary('./teencode.txt')  
english_dict = load_dictionary('./english-vnmese.txt')  
wrong_lst = load_list('./wrong-word.txt')  
stopword_lst = load_list('./vietnamese-stopwords.txt')  

# Load models once
with open('tfidf.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('logistic_regression_model.pkl', 'rb') as file:
    logistics_regression = pickle.load(file)


# Streamlit UI Setup
st.title("Data Science Project")
st.markdown("# :rainbow[Sentiment Analysis]")

menu = ["Vấn đề kinh doanh", "Xây dựng model", "Tìm kiếm mới","Phân tích đánh giá sản phẩm"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Lương Nhã Hoàng Hà & Phạm Bích Nhật""")
st.sidebar.write("""#### Giảng viên hướng dẫn: 
                 Cô Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 
                 14/12/2024""")

# Business problem page
if choice == 'Vấn đề kinh doanh':    
    st.markdown("## Vấn đề kinh doanh")
    multi = """
    ##### Nhận diện, phân loại phản hồi của khách hàng trên website Hasaki.vn.
    ##### Từ những đánh giá của khách hàng, giúp nhãn hàng hiểu khách hàng đánh giá gì về sản phẩm, từ đó đưa ra các kế hoạch cải thiện chất lượng sản phẩm cũng như các dịch vụ đi kèm.
    """
    st.markdown(multi)  
    st.write("""##### Thực hiện: Sử dụng Machine Learning (Random Forest).""")
    st.image("Sentiment-analysis.jpg")


# Model Building page
elif choice == 'Xây dựng model':
    data = pd.read_csv("Danh_gia.csv", encoding='utf-8')
    st.subheader("Xây dựng model")
    st.write("##### 1. Thu thập và đọc data")
    st.dataframe(data.head(10))
    st.image('PhanPhoiRating.png')

    st.write("##### 2. Data preprocessing")
    multi2 = '''##### Các bước xử lý dữ liệu bao gồm:
    * Chuyển nhận xét từ viết hoa thành viết thường
    * Loại bỏ các kí tự đặc biệt trong câu (dấu, số, stop-word, khoảng trắng)
    * Thay emoji và teen-code bằng từ tiếng việt chuẩn
    * Nối các câu lại với nhau, và ngăn cách giữa các câu bằng dấu chấm
    Phân loại nhận xét thành positive (từ 4 ⭐ ⬆️) và negative (từ 3 ⭐⬇️)
    Over-sampling các nhận xét negative để xử lý mất cân bằng dữ liệu (hơn 70% nhận xét là positive)
    '''
    st.markdown(multi2)
    st.write("##### Output dữ liệu sau khi tiền xử lý")
    Output = pd.read_csv("comment_df.csv", encoding='utf-8')
    st.dataframe(Output.head(10))

    st.write("##### 3. Build model")
    st.write("##### 3.1. Modeling with traditional machine learning")
    multi3 = '''
    models = { \n
    "Naive Bayes": MultinomialNB(),\n
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42), \n
    "Random Forest": RandomForestClassifier(random_state=42), \n
    }
    '''
    st.markdown(multi3)
    st.image('ROC_curve.png')


# New search page
elif choice == 'Tìm kiếm mới':
    st.subheader("Chọn input")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    # File upload
        
    if type=="Upload":
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file is not None:
            # Read and decode the file content using UTF-8
            content = uploaded_file.read().decode('utf-8')
            
            # Process the content based on file type
            if uploaded_file.name.endswith('.txt'):
                # Split the content into lines (create a list from lines)
                lines = content.splitlines()  # This splits by newlines
                st.write("Loaded lines from text file:")
                st.write(lines)
            
            elif uploaded_file.name.endswith('.csv'):
                # Load CSV file into a DataFrame and convert to list
                # Assuming the CSV has one column, or you're interested in the first column
                lines = pd.read_csv(io.StringIO(content), header=None)[0].tolist()  # Convert first column to list
                st.write("Loaded data from CSV file:")
                st.write(lines)
            flag = True

    # Direct input
    if type == "Input":        
        content = st.text_area(label="Input your content:")
        if content:
            lines = content.split('\n')
            flag = True

    # Process and predict
    if flag:
        preprocessed_lines = [
            remove_stopwords(
                process_postag_thesea(
                    process_special_word(process_text(line, emoji_dict, teen_dict, english_dict, wrong_lst))
                ),
                stopword_lst
            )
            for line in lines
        ]
        # After loading, you can process the list (e.g., preprocessing)
        st.write("Danh sách comment:")
        st.write(lines)
        st.write("Nội dung đã xử lý:")
        st.code("\n".join(preprocessed_lines))  

        # Predict sentiment
        x_new = tfidf.transform(preprocessed_lines)
        y_pred_new = logistics_regression.predict(x_new)
        sentiments = ["Positive" if pred == 1 else "Negative" for pred in y_pred_new]

        # Display prediction
        result_df = pd.DataFrame({
            'Original Text': lines,
            'Processed Text': preprocessed_lines,
            'Sentiment': sentiments
        })
        st.write("### Dự đoán:")
        st.dataframe(result_df)
elif choice == 'Phân tích đánh giá sản phẩm':
    st.markdown("## Phân tích đánh giá sản phẩm")
    # Positive and negative words and emojis
    positive_words = [
        "thích", "tốt", "xuất sắc", "tuyệt vời", "ổn", "tuyệt", "ok", "okay",
        "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "nhanh", "tiện lợi", "dễ sử dụng", 
        "hiệu quả", "ấn tượng", "nổi bật", "thân thiện", "cao cấp", "độc đáo", "rất tốt", 
        "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp", "hấp dẫn", "an tâm", "thúc đẩy", 
        "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội", "sáng tạo", 
        "phù hợp", "tận tâm", "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận", 
        "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền"
    ]
    negative_words = [
        "kém", "tệ", "buồn", "chán", "không dễ chịu", "không chất lượng", "kém chất lượng", 
        "không thích", "không ổn", "không hợp", "không đáng tin cậy", "không chuyên nghiệp",
        "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", 
        "không đáng giá", "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp", "khó chịu", 
        "gây khó dễ", "rườm rà", "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ",
        "không rõ ràng", "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền", 'không hài lòng', 
        'không đáng', 'quá tệ', 'rất tệ', 'thất vọng', 'chán', 'tệ hại', 'kinh khủng', 'không ưng ý'
    ]
    # Emoji lists
    positive_emojis = ["😄", "😃", "😀", "😁", "😆", "😅", "🤣", "😂", "🙂", "🙃", "😉", "😊", "😇", "🥰", "😍"]
    negative_emojis = ["😞", "😔", "🙁", "☹️", "😕", "😢", "😭", "😖", "😣", "😩", "😠", "😡", "🤬", "😤", "😰", "😨"]

    def find_words(document, list_of_words):
        document_lower = document.lower()
        word_count = 0
        word_list = []
        for word in list_of_words:
            if word in document_lower:
                word_count += document_lower.count(word)
                word_list.append(word)
        return word_count, word_list

    # Display list of products
    df_products = pd.read_csv('San_pham.csv')
    st.session_state.df_products = df_products

    if 'selected_ma_san_pham' not in st.session_state:
        st.session_state.selected_ma_san_pham = None

    product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in df_products.iterrows()]
    selected_product = st.selectbox(
        "Chọn sản phẩm",
        options=product_options,
        format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
    )

    st.session_state.selected_ma_san_pham = selected_product[1]
    if st.session_state.selected_ma_san_pham:
        st.write("ma_san_pham:", st.session_state.selected_ma_san_pham)
        selected_product_info = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]
        if not selected_product_info.empty:
            st.write('### Bạn vừa chọn:')
            st.write('### ', selected_product_info['ten_san_pham'].values[0])
            product_description = selected_product_info['mo_ta'].values[0]
            truncated_description = ' '.join(product_description.split()[:100])
            st.write('#### Thông tin:')
            st.write(truncated_description, '...')

        # Load the 'comment_df.csv' file containing the ratings and comments
        comment_data = pd.read_csv('comment_df.csv', encoding='utf-8')
        selected_product_comments = comment_data[comment_data['ma_san_pham'] == st.session_state.selected_ma_san_pham]

        # Calculate the average rating for the selected product
        average_rating = selected_product_comments['so_sao'].mean()
        st.write(f"### Điểm trung bình của sản phẩm {selected_product_info['ten_san_pham'].values[0]}: {average_rating:.2f} ")

        # Split comments into positive and negative based on ratings
        positive_comments = selected_product_comments[selected_product_comments['so_sao'] >= 4]['noi_dung_binh_luan']
        negative_comments = selected_product_comments[selected_product_comments['so_sao'] <= 3]['noi_dung_binh_luan']

        positive_comments_text = ' '.join(positive_comments)
        negative_comments_text = ' '.join(negative_comments)

        # Optionally: Display the comments themselves
        st.subheader("Positive Comments")
        st.write(positive_comments)

        st.subheader("Negative Comments")
        st.write(negative_comments)

        # Count words and emojis for positive and negative words
        positive_word_count, positive_word_list = find_words(' '.join(positive_comments), positive_words)
        st.write("Số lượng từ tích cực:", positive_word_count)
        st.write("Danh sách từ tích cực:", positive_word_list)

        negative_word_count, negative_word_list = find_words(' '.join(negative_comments), negative_words)
        st.write("Số lượng từ tiêu cực:", negative_word_count)
        st.write("Danh sách từ tiêu cực:", negative_word_list)

        positive_emoji_count, positive_emoji_list = find_words(' '.join(positive_comments), positive_emojis)
        st.write("Số lượng emoji tích cực:", positive_emoji_count)
        st.write("Danh sách emoji tích cực:", positive_emoji_list)

        negative_emoji_count, negative_emoji_list = find_words(' '.join(negative_comments), negative_emojis)
        st.write("Số lượng emoji tiêu cực:", negative_emoji_count)
        st.write("Danh sách emoji tiêu cực:", negative_emoji_list)
        
    # Generate word clouds for positive and negative comments
    if not positive_comments.empty:
        # Join all positive comments into a single string before generating the word cloud
        positive_comments_text = ' '.join(positive_comments)
        positive_wordcloud = WordCloud(width=800, height=400, background_color='white',max_words=30).generate(positive_comments_text)
        st.subheader("Word Cloud for Positive Comments")
        st.image(positive_wordcloud.to_array())

    if not negative_comments.empty:
        # Join all negative comments into a single string before generating the word cloud
        negative_comments_text = ' '.join(negative_comments)
        negative_wordcloud = WordCloud(width=800, height=400, background_color='white',max_words=30).generate(negative_comments_text)
        st.subheader("Word Cloud for Negative Comments")
        st.image(negative_wordcloud.to_array())
