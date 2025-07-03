
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib




def load_data():
    """Load sample house data (replace with real dataset)."""                  #  Load & Preprocess Data
    data = {
        'Area': [1200, 1500, 1800, 2000, 2200],  # Numerical feature
        'Bedrooms': [2, 3, 3, 4, 4],             # Numerical feature
        'Location': ['A', 'B', 'A', 'C', 'B'],    # Categorical feature
        'Price': [300000, 450000, 500000, 600000, 700000]  # Target
    }
    return pd.DataFrame(data)

df = load_data()
df_encoded = pd.get_dummies(df, columns=['Location'], drop_first=True)#  Load & Preprocess Data


X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']


# Train-Test Split 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


y_pred = lin_reg.predict(X_test)          # Predict & Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"\n📊 Linear Regression Results:")
print(f"- Mean Squared Error (MSE): ${mse:,.2f}")
print(f"- Predicted Prices: {y_pred}")


joblib.dump(lin_reg, 'house_price_lin_reg.joblib')
print("\n💾 Model saved as 'house_price_lin_reg.joblib'!")


price_threshold = 500000  # Example threshold
y_class = np.where(y > price_threshold, 1, 0)  # 1=Expensive, 0=Affordable


X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)  # Split data for classification


log_reg = LogisticRegression()
log_reg.fit(X_train_cls, y_train_cls)

y_pred_cls = log_reg.predict(X_test_cls)
accuracy = accuracy_score(y_test_cls, y_pred_cls)
print(f"\n🎯 Logistic Regression Results (Classification):")
print(f"- Accuracy: {accuracy * 100:.2f}%")




def predict_house_price(area, bedrooms, location):        #  Make Predictions 
    """Predict house price using saved model."""
    model = joblib.load('house_price_lin_reg.joblib')
    
    # Create DataFrame with one-hot encoding
    loc_cols = ['Location_B', 'Location_C']            #  Create DataFrame with one-hot encoding  
    loc_encoded = [0] * len(loc_cols)                  #Adjust based on your data
    
    
    if location == 'B':
        loc_encoded[0] = 1
    elif location == 'C':
        loc_encoded[1] = 1
    
    new_house = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        **dict(zip(loc_cols, loc_encoded))
    })
    
    predicted_price = model.predict(new_house)
    return predicted_price[0]


pred_price = predict_house_price(1600, 3, 'B')             # Example Prediction
print(f"\n🔮 Predicted Price for 1600 sqft, 3 BR, Location 'B': ${pred_price:,.2f}")

if __name__ == "__main__":
    print("\n✅ Project executed successfully!")
