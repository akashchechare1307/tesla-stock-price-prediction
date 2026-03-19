# 🚗 Tesla AI Stock Forecaster

> *An Interactive Deep Learning Web Application for Tesla (TSLA) Stock Price Prediction*

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## ✨ Features

- 📈 **Interactive Stock Charts** - Beautiful Plotly visualizations with hover effects
- 🤖 **Dual Model Support** - Compare SimpleRNN vs LSTM architectures
- 🔮 **Future Forecasting** - Predict TSLA prices for 1-30 days ahead
- 🎨 **Neon-Themed UI** - Eye-catching animated gradient backgrounds
- 📊 **Training Metrics** - Real-time loss and MAE monitoring
- 📱 **Responsive Design** - Works on desktop and mobile

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|--------|
| **Streamlit** | Web UI Framework |
| **TensorFlow/Keras** | Deep Learning Models |
| **SimpleRNN & LSTM** | Recurrent Neural Networks |
| **Plotly** | Interactive Charts |
| **yfinance** | Stock Data API |
| **Scikit-learn** | Data Preprocessing |
| **Pandas & NumPy** | Data Manipulation |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/tesla-stock-price-prediction.git
cd tesla-stock-price-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🎯 Project Overview

This project uses **Deep Learning** to predict Tesla (TSLA) closing prices based on historical stock data. It compares two neural network architectures:

- **SimpleRNN**: Basic recurrent neural network for sequence modeling
- **LSTM**: Long Short-Term Memory network for capturing long-term dependencies

The models learn from the last 60 days of price history and forecast future prices.

---

## 📊 How It Works

```
📥 Input: Last 60 days of TSLA prices
  ↓
⚙️ Preprocessing: MinMaxScaler normalization
  ↓
🤖 Model: SimpleRNN or LSTM architecture
  ↓
📤 Output: Predicted prices for selected horizon
```

---

## 🎨 UI Features

- **Animated gradient background** that flows continuously
- **Neon glow title** with rainbow color transitions
- **Glowing cards** for metrics and statistics
- **Custom styled buttons** with hover animations
- **Dark theme** optimized for data visualization

---

## ⚠️ Disclaimer

This application is built for **educational and learning purposes only**. It should NOT be used for making financial or trading decisions. Stock market predictions are inherently uncertain and influenced by many factors beyond historical price data.

---

## 👨‍💻 About the Author

**Akash Chechare**  
Electronics & Computer Engineering Student  
Pravara Rural Engineering College, Maharashtra

- 📧 Contact: akashchechare1307@gmail.com
- 💼 LinkedIn: [Connect with me](https://linkedin.com/in/akashchechare)
- 🐙 GitHub: [My Projects](https://github.com/akashchechare1307)

---

## 📜 License

MIT License - Feel free to use, modify, and distribute!

---

<div align="center">

**Built with ❤️ using Streamlit, TensorFlow & Plotly**

Made in India 🇮🇳

</div>
