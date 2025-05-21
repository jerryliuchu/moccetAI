import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(df):
    sns.countplot(data=df, x='sentiment')
    plt.title("Sentiment Distribution")
    plt.show()
