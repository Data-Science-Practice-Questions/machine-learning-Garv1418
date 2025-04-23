import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tkinter import Tk, Label, Entry, Button, messagebox

# Load the dataset
file_path = 'project of ml\\film_script_success_data.csv'  
data = pd.read_csv(file_path)

# Define feature columns and target column
features = ['sentiment_score', 'keyword_count', 'dialogue_length']
target = 'success'

# Encode 'Genre' using one-hot encoding
data = pd.get_dummies(data, columns=['genre'], drop_first=True)
features.extend([col for col in data.columns if col.startswith('genre_')])

# Prepare X (features) and y (target)
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Tkinter GUI
def predict_success():
    try:
        # Collect inputs
        sentiment = float(sentiment_entry.get())
        keywords = int(keywords_entry.get())
        dialogues = int(dialogues_entry.get())
        genre = genre_entry.get().capitalize()

        # Create a DataFrame for the input script
        genre_columns = {col: 0 for col in data.columns if col.startswith('genre_')}
        if f'genre_{genre}' in genre_columns:
            genre_columns[f'genre_{genre}'] = 1
        else:
            messagebox.showerror("Error", f"Unknown genre: {genre}")
            return

        example_script = pd.DataFrame([{
            'sentiment_score': sentiment,
            'keyword_count': keywords,
            'dialogue_length': dialogues,
            **genre_columns
        }])

        # Predict success
        prediction = model.predict(example_script)
        success_message = "The film script is predicted to succeed!" if prediction[0] == 1 else "The film script is predicted to fail."
        messagebox.showinfo("Prediction", success_message)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the Tkinter window
root = Tk()
root.title("Film Script Success Predictor")

# Input labels and fields
Label(root, text="Sentiment Score -1.0~1.0").grid(row=0, column=0, padx=10, pady=5)
sentiment_entry = Entry(root)
sentiment_entry.grid(row=0, column=1, padx=10, pady=5)

Label(root, text="Keyword Count 0~50").grid(row=1, column=0, padx=10, pady=5)
keywords_entry = Entry(root)
keywords_entry.grid(row=1, column=1, padx=10, pady=5)

Label(root, text="Dialogue Length 0~1000").grid(row=2, column=0, padx=10, pady=5)
dialogues_entry = Entry(root)
dialogues_entry.grid(row=2, column=1, padx=10, pady=5)

Label(root, text="Genre:").grid(row=3, column=0, padx=10, pady=5)
genre_entry = Entry(root)
genre_entry.grid(row=3, column=1, padx=10, pady=5)

# Predict button
predict_button = Button(root, text="Predict", command=predict_success)
predict_button.grid(row=4, columnspan=2, pady=10)

# Run the application
root.mainloop()
