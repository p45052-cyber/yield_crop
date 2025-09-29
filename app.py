import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
try:
    model = joblib.load('random_forest_regression_model_very_small.pkl')
    scaler = joblib.load('scaler_model.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'random_forest_regression_model_very_small.pkl' and 'scaler_model.pkl' are in the same directory.")
    st.stop()

# Create the Streamlit app
st.title("Crop Yield Prediction")

st.write("Enter the features to predict crop yield.")

# Create input fields for features
area = st.selectbox("Area", ['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bangladesh', 'Belarus', 'Belgium', 'Bhutan', 'Bolivia (Plurinational State of)', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Ethiopia', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'India', 'Indonesia', 'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kuwait', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Serbia', 'Sierra Leone', 'Slovakia', 'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Tajikistan', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United Republic of Tanzania', 'United States of America', 'Uruguay', 'Uzbekistan', 'Venezuela (Bolivarian Republic of)', 'Zambia', 'Zimbabwe'])
item = st.selectbox("Item", ['Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Sweet potatoes', 'Wheat', 'Cassava', 'Yams', 'Beans, dry', 'Peas, dry', 'Groundnuts with shell', 'Rye', 'Oats', 'Millet', 'Lentils', 'Soybeans', 'Sunflower seed', 'Rape seed', 'Sesame seed', 'Olives', 'Sugar beet', 'Sugar cane', 'Cotton lint', 'Tobacco, unmanufactured', 'Cloves', 'Apples', 'Bananas', 'Oranges', 'Lemons, limes', 'Grapefruit, pomelos', 'Grapes', 'Strawberries', 'Raspberries', 'Apples', 'Dates', 'Figs', 'Olives', 'Pears', 'Peaches', 'Plums', 'Cherries', 'Apricots', 'Nectarines', 'Kiwifruit', 'Mangoes', 'Avocados', 'Pineapples', 'Bananas', 'Plantains', 'Pawpaws', 'Guavas', 'Cashew nuts', 'Pistachios', 'Almonds, in shell', 'Walnuts, in shell', 'Hazelnuts, in shell', 'Chestnuts', 'Coffee, green', 'Cocoa beans', 'Tea leaves', 'Hops', 'Pepper (piper spp.)', 'Chillies and peppers, dry', 'Vanilla', 'Cinnamon (canella spp.)', 'Nutmeg, mace and cardamoms', 'Anise, badian, fennel, coriander, cumin, caraway', 'Ginger', 'Turmeric', 'other spices', 'Coconut, in shell', 'Oil palm fruit', 'Soybean oil', 'Groundnut oil', 'Sunflower-seed oil', 'Rape or colza seed oil', 'Olive oil', 'Palm oil', 'Coconut (copra) oil', 'Palm kernel oil', 'Castor oil seed', 'Tung oil seed', 'Mustard seed', 'Poppy seed', 'Linseed', 'Hempseed', 'Safflower seed', 'Cottonseed', 'Sesame seed', 'Vegetable oils, n.e.c.', 'Karite nuts (shea nuts)', 'Tallow tree seed', 'Candlenut', 'Kapok fruit', 'Other oilseeds', 'Cabbages', 'Cauliflowers and broccoli', 'Lettuce and chicory', 'Spinach', 'Artichokes', 'Asparagus', 'Onions, dry', 'Garlic', 'Leeks, other alliaceous vegetables', 'Tomatoes', 'Cucumbers and gherkins', 'Eggplants (aubergines)', 'Chillies and peppers, green', 'Pumpkins, squash and gourds', 'Watermelons', 'Cantaloupes and other melons', 'Vegetables, fresh nes', 'Cucumbers and gherkins', 'Okra', 'Beans, green', 'Peas, green', 'Broad beans and horse beans, dry', 'Chick peas', 'Cow peas, dry', 'Pigeon peas', 'Lupins', 'Vetches', 'Minor pulses, dry, nes', 'Potatoes', 'Sweet potatoes', 'Yams', 'Cassava', 'Taro (cocoyam)', 'Roots and tubers, nes', 'Sugar beet', 'Sugar cane', 'Sugar Crops Primary', 'Sugar non-centrifugal', 'Maize', 'Rice, paddy', 'Sorghum', 'Millet', 'Wheat', 'Barley', 'Oats', 'Cereals, other', 'Cereals, Total'])
year = st.slider("Year", 1990, 2013, 2000)
avg_rain_fall = st.number_input("Average Rainfall (mm per year)", min_value=0.0, max_value=3500.0, value=1000.0)
pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, max_value=400000.0, value=50000.0)
avg_temp = st.number_input("Average Temperature (°C)", min_value=0.0, max_value=35.0, value=20.0)

# Create a button to predict
if st.button("Predict Yield"):
    # Create a dataframe from the input
    input_data = pd.DataFrame({
        'Year': [year],
        'average_rain_fall_mm_per_year': [avg_rain_fall],
        'pesticides_tonnes': [pesticides],
        'avg_temp': [avg_temp],
        'Area': [area],
        'Item': [item]
    })

    # Standardize the numerical features using the loaded scaler
    numerical_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # One-hot encode categorical features using the same columns as training
    # We need to ensure the columns match the training data, even if a category isn't present in the input
    # This requires knowing the columns after one-hot encoding from the training data
    # For simplicity here, we'll re-create the one-hot encoding and align columns.
    # In a real application, you would save and load the list of columns from training.
    X_train_cols = model.feature_names_in_ # Get the column names from the trained model
    input_data = pd.get_dummies(input_data, columns=['Area', 'Item'], drop_first=True)

    # Reindex input_data to match the training columns, filling missing with 0
    input_data = input_data.reindex(columns=X_train_cols, fill_value=0)


    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.subheader("Predicted Crop Yield (hg/ha):")
    st.write(f"{prediction[0]:.2f}")
