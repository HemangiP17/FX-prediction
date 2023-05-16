# TensorFlow Data Preprocessing Project
This is a Python project that demonstrates data preprocessing using TensorFlow. It includes code to read a dataset, perform data transformation, and create TFRecord files for further processing.

Project Overview
The project consists of the following main components:
<ol>
<li>Dataset Retrieval: The script checks if the dataset file exists locally. If not, it downloads it from a specified URL.</li>
<li>Data Preprocessing: The dataset is loaded using pandas, and various data transformations are applied. These transformations include calculating the change between 'CAD-high' and 'CAD-close', digitizing the change into target classes using specified bins, and extracting date-related features.</li>
<li>TFRecord Creation: The preprocessed data is then serialized into TFRecord files using TensorFlow's TFRecordWriter. Each record in the TFRecord file contains features such as tickers, day of the week, month of the year, hour of the day, and the target class.</li>
<li>Custom Imputer: A custom TensorFlow preprocessing layer, myImputer, is implemented to handle missing values in the data. The layer replaces NaN values with precomputed imputation values.</li>
</ol>

## Dependencies
The project requires the following dependencies:
<ol>
<li>TensorFlow</li>
<li>Pandas</li>
<li>NumPy</li>
</ol>

## Contributing
Contributions to the project are welcome. If you find any issues or have suggestions for improvements, please create an issue or submit a pull request.

## License
The project is licensed under the MIT License.

## Acknowledgments
This project was inspired by the need for efficient data preprocessing in machine learning tasks. Special thanks to the contributors and maintainers of the TensorFlow and pandas libraries for providing excellent tools for data manipulation and preprocessing.

## Contact
For any inquiries or questions, please contact Hemangi Patel at hemangipatel171998@gmail.com or hp493@drexel.edu.
