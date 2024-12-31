import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Define thresholds for the parameters
THRESHOLDS = {
    "O2": 95,
    "Oxygen": 95,
    "Pulse": 100,
    "Temperature": 37,
    "HeartRate": 120,
}


# Function to check thresholds and classify parameters
def check_thresholds(row):
    above = []
    below = []
    for parameter, threshold in THRESHOLDS.items():
        if parameter in row:
            if row[parameter] > threshold:
                above.append(parameter)
            else:
                below.append(parameter)
    return above, below


# Sample data with hospitalization labels
SAMPLE_DATA = {
    "O2": [92, 96, 93, 97, 91, 89, 88, 99],
    "Oxygen": [97, 94, 96, 98, 95, 93, 90, 100],
    "Pulse": [78, 110, 85, 102, 88, 120, 98, 99],
    "Temperature": [36.5, 37.5, 36.8, 38, 36.2, 38.1, 36.9, 37.0],
    "HeartRate": [115, 125, 118, 130, 119, 135, 121, 115],
    "Hospitalization": [1, 1, 0, 1, 0, 1, 1, 0],  # 1 = Hospitalization Needed, 0 = Not Needed
}


# Train the Decision Tree model
def train_model():
    df = pd.DataFrame(SAMPLE_DATA)
    X = df.drop("Hospitalization", axis=1)
    y = df["Hospitalization"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Training Complete")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model


# Function to process the file, display alerts, and predict hospitalization
def process_file(file_path):
    try:
        data = pd.read_csv(file_path)  # Load the CSV file
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file: {e}")
        return

    above_counts = {key: 0 for key in THRESHOLDS.keys()}
    below_counts = {key: 0 for key in THRESHOLDS.keys()}

    alert_messages = []
    hospitalization_predictions = []

    for index, row in data.iterrows():
        above, below = check_thresholds(row)
        for param in above:
            above_counts[param] += 1
        for param in below:
            below_counts[param] += 1

        if above:
            alert_messages.append(f"Alert for row {index + 1}: {', '.join(above)} exceeded thresholds.")
        else:
            alert_messages.append(f"Row {index + 1} is within thresholds.")

        # Predict hospitalization
        try:
            prediction = model.predict([row[THRESHOLDS.keys()].values])[0]
            hospitalization_predictions.append(
                f"Row {index + 1}: {'Hospitalization Needed' if prediction else 'No Hospitalization Needed'}"
            )
        except Exception as e:
            hospitalization_predictions.append(f"Row {index + 1}: Prediction failed due to {e}")

    messagebox.showinfo("Processing Complete", "\n".join(alert_messages + hospitalization_predictions))

    # Visualization
    plot_thresholds(above_counts, below_counts)


# Function to plot parameters above and below thresholds
def plot_thresholds(above_counts, below_counts):
    parameters = list(THRESHOLDS.keys())
    above_values = [above_counts[param] for param in parameters]
    below_values = [below_counts[param] for param in parameters]

    x = range(len(parameters))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x, above_values, width, label='Above Threshold', color='red')
    plt.bar([p + width for p in x], below_values, width, label='Below Threshold', color='green')

    plt.xlabel('Parameters')
    plt.ylabel('Counts')
    plt.title('Parameters Above and Below Thresholds')
    plt.xticks([p + width / 2 for p in x], parameters)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Function to open file dialog and select a CSV file
def select_file():
    file_path = filedialog.askopenfilename(
        title="Select a CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if file_path:
        process_file(file_path)


# Train model globally
model = train_model()


# GUI Setup
def safe_tkinter_execution():
    try:
        root = tk.Tk()
        root.title("Parameter Threshold Checker with Decision Tree")

        frame = tk.Frame(root, padx=20, pady=20)
        frame.pack()

        label = tk.Label(frame, text="Upload a CSV file to check thresholds and predict hospitalization",
                         font=("Arial", 14))
        label.pack(pady=10)

        upload_button = tk.Button(frame, text="Upload File", command=select_file, font=("Arial", 12))
        upload_button.pack(pady=5)

        exit_button = tk.Button(frame, text="Exit", command=root.quit, font=("Arial", 12))
        exit_button.pack(pady=5)

        root.mainloop()
    except tk.TclError:
        print("Error: Tkinter GUI is not supported in this environment.")


# Execute safely
safe_tkinter_execution()
