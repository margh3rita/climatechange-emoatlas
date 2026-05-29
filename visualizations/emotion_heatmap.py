import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ND_DIR = Path("./data/emoatlas_natural_democratics")
NR_DIR = Path("./data/emoatlas_natural_republicans")
SD_DIR = Path("./data/emoatlas_synthetic_democratics")
SR_DIR = Path("./data/emoatlas_synthetic_republicans")

OUTPUT_DIR = Path("./data/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

groups = ['NAT', 'SYN']
titles = ['Z-score Emotion Heatmap — Democrat titles', 'Z-score Emotion Heatmap — Republican titles']
cmap = plt.cm.PRGn


for year in ['2016', '2017']:
    

    nd_df = pd.read_csv(ND_DIR / f"zscores_{year}.csv")
    nr_df = pd.read_csv(NR_DIR / f"zscores_{year}.csv")
    sd_df = pd.read_csv(SD_DIR / f"zscores_{year}.csv")
    sr_df = pd.read_csv(SR_DIR / f"zscores_{year}.csv")
    
    emotions = nd_df['Unnamed: 0'].tolist()
    
    values_dem = np.array([nd_df['zscore'].tolist(), sd_df['zscore'].tolist()])
    values_rep = np.array([nr_df['zscore'].tolist(), sr_df['zscore'].tolist()])
    
    datasets = [values_dem, values_rep]
    
    filenames = [
        OUTPUT_DIR / f'emotion_heatmap_dem_{year}.png', 
        OUTPUT_DIR / f'emotion_heatmap_rep_{year}.png'
    ]
    
    for i in range(2):
        values = datasets[i]

        fig, ax = plt.subplots(figsize=(11, 4.5))
        im = ax.imshow(values, cmap=cmap, vmin=-3, vmax=5, aspect='auto')

        # Configurazione degli assi e dei testi
        ax.set_xticks(range(len(emotions)))
        ax.set_xticklabels(emotions, fontsize=11)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=11)
        ax.set_xlabel('Emotion', fontsize=12, labelpad=8)
        ax.set_ylabel('Group', fontsize=12, labelpad=8)
        ax.set_title(f'{titles[i]} - {year}', fontsize=13, fontweight='bold', pad=12)

        for row in range(len(groups)):
            for col in range(len(emotions)):
                v = values[row, col]
          
                t = (v - (-3)) / (5 - (-3))
                color = 'white' if t < 0.2 or t > 0.85 else 'black'
                label = str(int(v)) if v == int(v) else str(round(v, 2))
                ax.text(col, row, label, ha='center', va='center', fontsize=10, color=color, fontweight='500')

        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
        cbar.ax.tick_params(labelsize=10)

   
        plt.tight_layout()
        plt.savefig(filenames[i], dpi=150, bbox_inches='tight')
        plt.close()

