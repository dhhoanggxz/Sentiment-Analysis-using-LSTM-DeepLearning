# Các bước thực hiện phân tích cảm xúc trên IMDB Reviews với LSTM, cùng với những thư viện đã nêu sẽ như sau:

### 1. Chuẩn bị và nạp dữ liệu:
   - **`os`** và **`zipfile`**: Nếu dữ liệu được lưu trữ trong file nén `.zip`, bạn sẽ cần giải nén file và đọc các file bên trong.
     ```python
     # Giải nén file zip nếu có
     with ZipFile('data.zip', 'r') as zip_ref:
         zip_ref.extractall('data_folder')
     ```

   - **`pandas`**: Đọc dữ liệu và chuyển nó vào `DataFrame` để tiện quản lý và xử lý dữ liệu.
     ```python
     import pandas as pd

     data = pd.read_csv('data_folder/IMDB_Dataset.csv')
     print(data.head())
     ```

### 2. Xử lý văn bản:
   - **`Tokenizer`**: Để chuẩn bị văn bản cho mô hình, bạn dùng `Tokenizer` để mã hóa từ ngữ thành số. Điều này giúp mô hình xử lý văn bản dễ dàng hơn.
     ```python
     from tensorflow.keras.preprocessing.text import Tokenizer

     # Khởi tạo tokenizer và giới hạn từ điển ở 5000 từ phổ biến nhất
     tokenizer = Tokenizer(num_words=5000)
     tokenizer.fit_on_texts(data['review'])
     sequences = tokenizer.texts_to_sequences(data['review'])
     ```

   - **`pad_sequences`**: Dữ liệu văn bản có thể có độ dài khác nhau, nên cần làm đồng nhất độ dài các câu để phù hợp với mô hình.
     ```python
     from tensorflow.keras.preprocessing.sequence import pad_sequences

     # Đảm bảo tất cả các chuỗi có độ dài là 500
     X = pad_sequences(sequences, maxlen=500)
     ```

### 3. Chia dữ liệu:
   - **`train_test_split`**: Chia dữ liệu thành hai tập: tập huấn luyện và tập kiểm tra để đánh giá mô hình.
     ```python
     from sklearn.model_selection import train_test_split

     y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

### 4. Xây dựng mô hình LSTM:
   - **`Sequential`**, **`Embedding`**, **`LSTM`**, và **`Dense`**: Xây dựng mô hình LSTM bằng Keras với cấu trúc Sequential, thêm lớp `Embedding` để chuyển đổi các từ thành vector, lớp `LSTM` để học các mối liên hệ trong chuỗi từ, và lớp `Dense` cuối cùng để dự đoán.
     ```python
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import Dense, Embedding, LSTM

     model = Sequential()
     model.add(Embedding(input_dim=5000, output_dim=128, input_length=500))
     model.add(LSTM(64))
     model.add(Dense(1, activation='sigmoid'))

     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
     ```

### 5. Huấn luyện mô hình:
   ```python
   model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
   ```

### Về Keras và tại sao chọn Keras
Keras là một thư viện mã nguồn mở, được xây dựng trên nền tảng của **TensorFlow**. Nó giúp đơn giản hóa việc xây dựng và huấn luyện các mạng nơ-ron thông qua các API thân thiện. Một số lý do phổ biến khi chọn Keras:

1. **Dễ sử dụng**: Keras có cú pháp đơn giản, dễ đọc và dễ mở rộng. Điều này giúp tiết kiệm thời gian, đặc biệt là cho các dự án nghiên cứu hay thử nghiệm.

2. **Tích hợp với TensorFlow**: Keras là API chính thức cho TensorFlow, tận dụng tối đa khả năng và hiệu suất của TensorFlow cho các ứng dụng lớn hoặc trong sản xuất.

3. **Khả năng mở rộng**: Keras cho phép tùy chỉnh và hỗ trợ nhiều loại lớp và kiến trúc mô hình khác nhau.

Ngoài Keras, bạn cũng có thể dùng các thư viện khác như **PyTorch** và **TensorFlow (raw)**:
   - **PyTorch**: Thích hợp hơn khi cần tùy chỉnh cao và linh hoạt, phổ biến trong nghiên cứu.
   - **TensorFlow**: Nếu bạn cần điều chỉnh chi tiết toàn bộ quá trình huấn luyện và tối ưu hóa, hoặc muốn triển khai mô hình trong sản xuất, sử dụng TensorFlow có thể thích hợp hơn.

Tuy nhiên, **Keras** vẫn là một lựa chọn hợp lý và phổ biến, đặc biệt cho các dự án deep learning cấp độ ban đầu và trung cấp do tính đơn giản của nó.
