#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PU Learning Happiness Point Modeling - Fixed Version
Key fix: Census missing values (negative millions) properly handled
Background color adjusted to website color scheme #FAFAF7
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cdist
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Website background color
SITE_BG_COLOR = '#FAFAF7'


def clean_census_data(df, census_cols):
    """
    Clean Census data
    Census API uses negative numbers for missing values (e.g., -666666666)
    """
    print("\n  Cleaning Census anomalies...")
    
    for col in census_cols:
        if col not in df.columns:
            continue
        
        # Count anomalies (negative or extremely large values)
        n_negative = (df[col] < 0).sum()
        
        if n_negative > 0:
            print(f"    {col}: Found {n_negative} negative values → Replacing with NaN")
            df.loc[df[col] < 0, col] = np.nan
    
    return df


class PULearning:
    """PU Learning Classifier"""
    
    def __init__(self, reliable_neg_ratio=0.3):
        self.reliable_neg_ratio = reliable_neg_ratio
        self.scaler = StandardScaler()
        
    def identify_reliable_negatives(self, X_positive, X_unlabeled):
        """Identify reliable negatives (distance-based)"""
        print("\n  [Step 1] Identifying reliable negatives...")
        
        self.positive_centroid = X_positive.mean(axis=0)
        distances = cdist(X_unlabeled, [self.positive_centroid], metric='euclidean').flatten()
        
        n_reliable_neg = int(len(X_unlabeled) * self.reliable_neg_ratio)
        reliable_neg_indices = np.argsort(distances)[-n_reliable_neg:]
        
        n_suspect = int(len(X_unlabeled) * 0.05)
        suspect_indices = np.argsort(distances)[:n_suspect]
        
        print(f"    Positive samples: {len(X_positive)}")
        print(f"    Reliable negatives: {n_reliable_neg} (farthest {self.reliable_neg_ratio*100:.0f}%)")
        print(f"    Suspect positives: {n_suspect} (closest 5%)")
        
        return reliable_neg_indices, suspect_indices, distances
    
    def fit(self, X_positive, X_unlabeled):
        """Train model"""
        X_all = np.vstack([X_positive, X_unlabeled])
        self.scaler.fit(X_all)
        
        X_pos_scaled = self.scaler.transform(X_positive)
        X_unl_scaled = self.scaler.transform(X_unlabeled)
        
        reliable_neg_idx, suspect_idx, self.distances = self.identify_reliable_negatives(
            X_pos_scaled, X_unl_scaled
        )
        self.reliable_neg_idx = reliable_neg_idx
        self.suspect_idx = suspect_idx
        
        X_reliable_neg = X_unl_scaled[reliable_neg_idx]
        self.X_train = np.vstack([X_pos_scaled, X_reliable_neg])
        self.y_train = np.concatenate([
            np.ones(len(X_pos_scaled)),
            np.zeros(len(X_reliable_neg))
        ])
        
        print(f"\n  [Step 2] Training classifiers...")
        print(f"    Training set: positives={int(self.y_train.sum())}, negatives={int(len(self.y_train)-self.y_train.sum())}")
        
        self.classifier = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.classifier.fit(self.X_train, self.y_train)
        
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.rf_classifier.fit(self.X_train, self.y_train)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lr_scores = cross_val_score(self.classifier, self.X_train, self.y_train, cv=cv, scoring='roc_auc')
        rf_scores = cross_val_score(self.rf_classifier, self.X_train, self.y_train, cv=cv, scoring='roc_auc')
        
        self.lr_auc_mean = lr_scores.mean()
        self.lr_auc_std = lr_scores.std()
        self.rf_auc_mean = rf_scores.mean()
        self.rf_auc_std = rf_scores.std()
        
        print(f"    Logistic Regression AUC: {lr_scores.mean():.4f} (±{lr_scores.std()*2:.4f})")
        print(f"    Random Forest AUC: {rf_scores.mean():.4f} (±{rf_scores.std()*2:.4f})")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        lr_coef = self.classifier.coef_[0]
        rf_importance = self.rf_classifier.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'lr_coefficient': lr_coef,
            'rf_importance': rf_importance,
            'direction': ['Positive' if c > 0 else 'Negative' for c in lr_coef]
        })
        
        importance_df = importance_df.sort_values('lr_coefficient', ascending=False)
        return importance_df
    
    def plot_roc_curve(self, output_dir):
        """Plot ROC curve - Nature style"""
        print("\n  Plotting ROC curve...")
        
        # Color scheme
        COLOR_1 = '#2F2D54'  # Dark purple-blue
        COLOR_2 = '#BD9AAD'  # Pink-purple
        COLOR_GRAY = '#9193B4'  # Light purple-gray
        
        # Set Nature style
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.linewidth': 1.2,
            'axes.facecolor': SITE_BG_COLOR,
            'figure.facecolor': SITE_BG_COLOR,
            'figure.dpi': 150,
            'savefig.facecolor': SITE_BG_COLOR,
        })
        
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor(SITE_BG_COLOR)
        ax.set_facecolor(SITE_BG_COLOR)
        
        # 5-fold cross validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Store results for each fold
        lr_tprs, rf_tprs = [], []
        lr_aucs, rf_aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]
            
            # Logistic Regression
            lr_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
            lr_clf.fit(X_train_fold, y_train_fold)
            lr_pred = lr_clf.predict_proba(X_val_fold)[:, 1]
            lr_fpr, lr_tpr, _ = roc_curve(y_val_fold, lr_pred)
            lr_aucs.append(auc(lr_fpr, lr_tpr))
            lr_tprs.append(np.interp(mean_fpr, lr_fpr, lr_tpr))
            lr_tprs[-1][0] = 0.0
            
            # Random Forest
            rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                            class_weight='balanced', random_state=42, n_jobs=-1)
            rf_clf.fit(X_train_fold, y_train_fold)
            rf_pred = rf_clf.predict_proba(X_val_fold)[:, 1]
            rf_fpr, rf_tpr, _ = roc_curve(y_val_fold, rf_pred)
            rf_aucs.append(auc(rf_fpr, rf_tpr))
            rf_tprs.append(np.interp(mean_fpr, rf_fpr, rf_tpr))
            rf_tprs[-1][0] = 0.0
        
        # Calculate mean curves
        lr_mean_tpr = np.mean(lr_tprs, axis=0)
        lr_mean_tpr[-1] = 1.0
        lr_mean_auc = np.mean(lr_aucs)
        lr_std_auc = np.std(lr_aucs)
        
        rf_mean_tpr = np.mean(rf_tprs, axis=0)
        rf_mean_tpr[-1] = 1.0
        rf_mean_auc = np.mean(rf_aucs)
        rf_std_auc = np.std(rf_aucs)
        
        # Plot mean ROC curves
        ax.plot(mean_fpr, lr_mean_tpr, color=COLOR_1, linewidth=2.5,
                label=f'Logistic Regression (AUC = {lr_mean_auc:.3f} ± {lr_std_auc:.3f})')
        ax.plot(mean_fpr, rf_mean_tpr, color=COLOR_2, linewidth=2.5,
                label=f'Random Forest (AUC = {rf_mean_auc:.3f} ± {rf_std_auc:.3f})')
        
        # Plot confidence intervals
        lr_std_tpr = np.std(lr_tprs, axis=0)
        ax.fill_between(mean_fpr, 
                        np.maximum(lr_mean_tpr - lr_std_tpr, 0),
                        np.minimum(lr_mean_tpr + lr_std_tpr, 1),
                        color=COLOR_1, alpha=0.15)
        
        rf_std_tpr = np.std(rf_tprs, axis=0)
        ax.fill_between(mean_fpr, 
                        np.maximum(rf_mean_tpr - rf_std_tpr, 0),
                        np.minimum(rf_mean_tpr + rf_std_tpr, 1),
                        color=COLOR_2, alpha=0.15)
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], linestyle='--', color=COLOR_GRAY, linewidth=1.5, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title('ROC Curves: 5-Fold Cross-Validation', fontsize=16, fontweight='bold', pad=15)
        
        legend = ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        legend.get_frame().set_facecolor(SITE_BG_COLOR)
        legend.get_frame().set_edgecolor('#CCCCCC')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches='tight',
                    facecolor=SITE_BG_COLOR, edgecolor='none')
        plt.close()
        print(f"  ✓ ROC curve saved")


def main():
    """Main function"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    analysis_file = data_dir / "analysis_data" / "all_points_full.csv"
    output_dir = data_dir / "modeling_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("PU Learning Happiness Point Modeling")
    print("=" * 70)
    
    # Read data
    print("\n[1/6] Reading data...")
    try:
        df = pd.read_csv(analysis_file)
        print(f"  ✓ Read {len(df)} records")
        print(f"  Happiness points: {df['is_happy'].sum()}")
        print(f"  Other points: {len(df) - df['is_happy'].sum()}")
    except Exception as e:
        print(f"  ✗ Read failed: {e}")
        return
    
    # Define features
    print("\n[2/6] Defining features...")
    
    street_features = [
        'sky_ratio',           # Sky visibility
        'green_view_index',    # Vegetation coverage
        'building_ratio',      # Building density
        'road_ratio',          # Road coverage
        'vehicle_ratio',       # Vehicle presence
        'person_ratio',        # Pedestrian presence
    ]
    
    census_features = [
        'median_income',       # Economic level
        'poverty_rate',        # Poverty level
        'pct_college',         # Education level
        'pct_white',           # Racial composition (simplified to single indicator)
        'median_age',          # Age structure
        'pct_owner_occupied',  # Home ownership rate
        'unemployment_rate',   # Employment status
    ]
    
    street_features = [f for f in street_features if f in df.columns]
    census_features = [f for f in census_features if f in df.columns]
    
    print(f"  Street features ({len(street_features)}): {street_features}")
    print(f"  Census features ({len(census_features)}): {census_features}")
    
    # Key fix: Clean Census anomalies
    print("\n[3/6] Cleaning data...")
    df = clean_census_data(df, census_features)
    
    all_features = street_features + census_features
    
    # Drop rows with too many missing values
    df_model = df.dropna(subset=street_features).copy()
    for col in census_features:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())
    
    print(f"  Valid records after cleaning: {len(df_model)}")
    print(f"  Happiness points: {df_model['is_happy'].sum()}")
    
    # Diagnostics: Feature mean comparison
    print("\n[4/6] Feature diagnostics (cleaned data)...")
    print("-" * 70)
    
    happy = df_model[df_model['is_happy'] == 1]
    other = df_model[df_model['is_happy'] == 0]
    
    print(f"{'Feature':<22} {'Happy Mean':>12} {'Other Mean':>12} {'Direction':>10} {'p-value':>10}")
    print("-" * 70)
    
    real_directions = {}
    
    for feat in all_features:
        h_mean = happy[feat].mean()
        o_mean = other[feat].mean()
        
        t_stat, p_val = stats.ttest_ind(happy[feat].dropna(), other[feat].dropna())
        direction = "↑" if h_mean > o_mean else "↓"
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        
        real_directions[feat] = direction
        print(f"{feat:<22} {h_mean:>12.4f} {o_mean:>12.4f} {direction:>10} {p_val:>8.4f} {sig}")
    
    # PU Learning
    print("\n[5/6] PU Learning modeling...")
    
    positive_mask = df_model['is_happy'] == 1
    X_positive = df_model.loc[positive_mask, all_features].values
    X_unlabeled = df_model.loc[~positive_mask, all_features].values
    
    pu_model = PULearning(reliable_neg_ratio=0.3)
    pu_model.fit(X_positive, X_unlabeled)
    
    # Plot ROC curve
    pu_model.plot_roc_curve(output_dir)
    
    # Feature importance
    print("\n  Feature importance (compared with actual direction)...")
    importance = pu_model.get_feature_importance(all_features)
    importance['real_direction'] = importance['feature'].map(real_directions)
    importance['match'] = importance.apply(
        lambda row: '✓' if (row['lr_coefficient'] > 0 and row['real_direction'] == '↑') or 
                          (row['lr_coefficient'] < 0 and row['real_direction'] == '↓') else '✗',
        axis=1
    )
    
    print("\n  " + "-" * 75)
    print(f"  {'Feature':<22} {'Model Dir':>10} {'Actual Dir':>10} {'Match':>6} {'Coefficient':>12}")
    print("  " + "-" * 75)
    
    for idx, row in importance.iterrows():
        print(f"  {row['feature']:<22} {row['direction']:>10} {row['real_direction']:>10} "
              f"{row['match']:>6} {row['lr_coefficient']:>+12.4f}")
    
    importance.to_csv(output_dir / "feature_importance_pu.csv", index=False)
    
    # City-wide scoring
    print("\n[6/6] City-wide scoring...")
    
    X_all = df_model[all_features].values
    raw_probs = pu_model.predict_proba(X_all)
    
    min_prob = raw_probs.min()
    max_prob = raw_probs.max()
    normalized_scores = (raw_probs - min_prob) / (max_prob - min_prob)
    normalized_scores[df_model['is_happy'].values == 1] = 1.0
    
    df_model['happiness_prob'] = raw_probs
    df_model['happiness_score'] = normalized_scores.round(4)
    
    df_model['sample_type'] = 'unlabeled'
    df_model.loc[df_model['is_happy'] == 1, 'sample_type'] = 'positive'
    
    unlabeled_indices = df_model[df_model['is_happy'] == 0].index
    reliable_neg_actual_idx = unlabeled_indices[pu_model.reliable_neg_idx]
    suspect_actual_idx = unlabeled_indices[pu_model.suspect_idx]
    
    df_model.loc[reliable_neg_actual_idx, 'sample_type'] = 'reliable_negative'
    df_model.loc[suspect_actual_idx, 'sample_type'] = 'suspect_positive'
    
    print(f"\n  Scoring statistics:")
    for stype in ['positive', 'suspect_positive', 'unlabeled', 'reliable_negative']:
        subset = df_model[df_model['sample_type'] == stype]['happiness_score']
        if len(subset) > 0:
            print(f"    {stype:<20}: n={len(subset):>6}, mean={subset.mean():.4f}, median={subset.median():.4f}")
    
    output_cols = ['point_id', 'longitude', 'latitude', 'is_happy', 'sample_type',
                   'happiness_prob', 'happiness_score'] + all_features
    output_cols = [c for c in output_cols if c in df_model.columns]
    df_model[output_cols].to_csv(output_dir / "happiness_predictions_pu.csv", index=False)
    
    # Visualization
    print("\nGenerating visualizations...")
    
    # Set global background color
    plt.rcParams.update({
        'axes.facecolor': SITE_BG_COLOR,
        'figure.facecolor': SITE_BG_COLOR,
        'savefig.facecolor': SITE_BG_COLOR,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.patch.set_facecolor(SITE_BG_COLOR)
    
    ax1 = axes[0]
    ax1.set_facecolor(SITE_BG_COLOR)
    plot_data = importance.sort_values('lr_coefficient')
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in plot_data['lr_coefficient']]
    ax1.barh(plot_data['feature'], plot_data['lr_coefficient'], color=colors)
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Coefficient')
    ax1.set_title('Logistic Regression Coefficients\n(Green=Positive Effect, Red=Negative Effect)', fontweight='bold')
    
    ax2 = axes[1]
    ax2.set_facecolor(SITE_BG_COLOR)
    plot_data2 = importance.sort_values('rf_importance')
    ax2.barh(plot_data2['feature'], plot_data2['rf_importance'], color='#3498db')
    ax2.set_xlabel('Importance')
    ax2.set_title('Random Forest Feature Importance', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_pu.png", dpi=150, bbox_inches='tight',
                facecolor=SITE_BG_COLOR, edgecolor='none')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(SITE_BG_COLOR)
    ax.set_facecolor(SITE_BG_COLOR)
    
    colors_map = {'positive': '#e74c3c', 'suspect_positive': '#f39c12', 
                  'unlabeled': '#3498db', 'reliable_negative': '#95a5a6'}
    
    for stype in ['reliable_negative', 'unlabeled', 'suspect_positive', 'positive']:
        subset = df_model[df_model['sample_type'] == stype]['happiness_score']
        if len(subset) > 0:
            ax.hist(subset, bins=50, alpha=0.6, label=f'{stype} (n={len(subset)})', 
                   color=colors_map[stype], range=(0, 1))
    
    ax.set_xlabel('Happiness Score (0-1)')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)
    ax.set_title('Distribution of Happiness Scores', fontweight='bold')
    legend = ax.legend()
    legend.get_frame().set_facecolor(SITE_BG_COLOR)
    
    plt.savefig(output_dir / "score_distribution_pu.png", dpi=150, bbox_inches='tight',
                facecolor=SITE_BG_COLOR, edgecolor='none')
    plt.close()
    
    print(f"  Charts saved to {output_dir}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)
    
    print("\n【Key Findings】")
    positive_feats = importance[importance['lr_coefficient'] > 0].head(6)
    negative_feats = importance[importance['lr_coefficient'] < 0].head(6)
    
    print("\n  Positive effects (higher = happier):")
    for idx, row in positive_feats.iterrows():
        match_mark = row['match']
        print(f"    {match_mark} {row['feature']:<20} (coefficient: {row['lr_coefficient']:+.4f})")
    
    print("\n  Negative effects (higher = less happy):")
    for idx, row in negative_feats.iterrows():
        match_mark = row['match']
        print(f"    {match_mark} {row['feature']:<20} (coefficient: {row['lr_coefficient']:+.4f})")
    
    print(f"\n【Output Files】")
    print(f"  {output_dir / 'feature_importance_pu.csv'}")
    print(f"  {output_dir / 'happiness_predictions_pu.csv'}")
    print(f"  {output_dir / 'roc_curve.png'}")
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")


if __name__ == "__main__":
    main()
