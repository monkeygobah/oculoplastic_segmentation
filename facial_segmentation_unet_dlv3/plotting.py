import matplotlib.pyplot as plt

# 2. 'l_eye', 'r_eye', 'l_brow', 'r_brow', = 1,2,3,4


# make histograms of dice scores for sclera and brows
def plot_dice_histograms(dice_scores_list, title):
    # Extracting scores for each class
    eye_scores = [d[1] for d in dice_scores_list if 1 in d]
    # r_eye_scores = [d[2] for d in dice_scores_list if 2 in d]
    brow_scores = [d[2] for d in dice_scores_list if 2 in d]
    # r_brow_scores = [d[4] for d in dice_scores_list if 4 in d]

    # Setting up the plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle('Histograms of Dice Scores by Class')

    # Plotting histograms for each class
    axs[0, 0].hist(eye_scores, bins=20, color='skyblue', edgecolor='black')
    axs[0, 0].set_title('Eye Dice')

    # axs[0, 1].hist(r_eye_scores, bins=20, color='orange', edgecolor='black')
    # axs[0, 1].set_title('Right Eye')

    axs[0, 1].hist(brow_scores, bins=20, color='lightgreen', edgecolor='black')
    axs[0, 1].set_title('Brow Dice')

    # axs[1, 1].hist(r_brow_scores, bins=20, color='pink', edgecolor='black')
    # axs[1, 1].set_title('Right Brow')

    # Setting labels
    for ax in axs.flat:
        ax.set(xlabel='Dice Score', ylabel='Frequency')
        
    ax.set_title(f'Histograms of Dice Scores by Class for {title}')

    # Adjusting layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"histograms_{title}.jpg")

# make box plots of dice scores 
def plot_dice_boxplots(dice_scores_list, title):
    # Extracting scores for each class
    eye_scores = [d[1] for d in dice_scores_list if 1 in d]
    # r_eye_scores = [d[2] for d in dice_scores_list if 2 in d]
    brow_scores = [d[2] for d in dice_scores_list if 2 in d]
    # r_brow_scores = [d[4] for d in dice_scores_list if 4 in d]

    scores = [eye_scores, brow_scores]
    labels = ['Sclera', 'Brows']

    # Setting up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(scores, labels=labels, patch_artist=True)

    # Adding title and labels
    ax.set_title(f'Box Plots of Dice Scores by Class for {title}')
    ax.set_ylabel('Dice Score')
    ax.set_xlabel('Class')

    # Customizing box colors
    colors = ['skyblue', 'orange']
    for patch, color in zip(ax.artists, colors):
        patch.set_facecolor(color)

    plt.xticks(rotation=45)  # Optional: Rotate class labels for better readability
    plt.tight_layout()
    plt.savefig(f"boxplots_{title}.jpg")

