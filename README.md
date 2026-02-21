# Crop Prediction and Recommendation System

This project helps users predict crop yield, suggest the best fertilizers, and recommend suitable crops for specific conditions using machine learning models.

## Features

1. **Crop Yield Prediction**:
   - Utilizes **Linear Regression** for basic predictions based on factors like rainfall, temperature, and soil nutrients.
   - For more complex scenarios, a **Random Forest** model is employed to handle non-linear relationships between multiple variables.
   
2. **Fertilizer Recommendation**:
   - Recommends appropriate fertilizers by analyzing soil properties such as nitrogen, phosphorus, potassium (NPK), pH level, and moisture content.
   
3. **Crop Type Suggestion**:
   - Suggests the best crop to grow based on environmental data, soil type, weather, and historical crop performance using machine learning models.

## Installation

1. Clone the repository and navigate into it:

   ```bash
   git clone https://github.com/dhruvdaberao/AGRIGAINS.git
   cd AGRIGAINS
   ```

2. (Optional) Create and activate a Python virtual environment.

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:

   ```bash
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`.
   Use the interface to select the desired tool (crop yield prediction, fertilizer recommendation, or crop type suggestion) and provide the required parameters.

## Data

This project uses publicly available agricultural datasets for model training and evaluation. See the notebooks in the repository for details.

## Contributing

Contributions are welcome! Feel free to open issues for bug reports or feature requests, and submit pull requests to propose changes.

## License

This project is provided under the MIT License. See the `LICENSE` file for details.
