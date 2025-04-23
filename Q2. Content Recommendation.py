import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from tkinter import Tk, Label, Entry, Button, messagebox, Toplevel
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'project of ml\\content_recommendation_data.csv'  
data = pd.read_csv(file_path)

# Preprocessing
features = ['scroll_depth', 'time_spent']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linked = linkage(X_scaled, method='ward')
data['cluster'] = fcluster(linked, t=4, criterion='maxclust')  


# Tkinter GUI
def visualize_clusters():
    top = Toplevel(window)
    top.title("Dendrogram Visualization")
    
    # Visualize Dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    # Save plot to display in tkinter window
    plt.savefig("dendrogram.png")
    plt.close()

    # Embed the dendrogram image in the Tkinter window
    from PIL import Image, ImageTk
    img = Image.open("dendrogram.png")
    img = ImageTk.PhotoImage(img)
    lbl = Label(top, image=img)
    lbl.image = img  
    lbl.pack()


def recommend_articles():
    try:
        cluster_num = int(cluster_entry.get())
        if cluster_num not in data['cluster'].unique():
            messagebox.showerror("Error", f"Cluster {cluster_num} does not exist.")
            return
        recommendations = data[data['cluster'] == cluster_num]['topic'].unique()
        messagebox.showinfo("Recommended Articles", f"Articles for Cluster {cluster_num}: {', '.join(recommendations)}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Create Tkinter window
window = Tk()
window.title("Content Recommendation Engine")

# Input for cluster selection
Label(window, text="Enter Cluster Number (1-4):").grid(row=0, column=0, padx=10, pady=5)
cluster_entry = Entry(window)
cluster_entry.grid(row=0, column=1, padx=10, pady=5)

# Buttons for functionalities
recommend_button = Button(window, text="Recommend Articles", command=recommend_articles)
recommend_button.grid(row=1, column=0, padx=10, pady=5)

dendrogram_button = Button(window, text="Visualize Dendrogram", command=visualize_clusters)
dendrogram_button.grid(row=1, column=1, padx=10, pady=5)

# Run the application
window.mainloop()
