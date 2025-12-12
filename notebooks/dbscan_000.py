import argparse, os, gzip, pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig
from rdkit.Chem import rdMolAlign
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import Counter

# ============== SETUP ==============
fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder',
        'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')


# ============== LOAD AND CENTER WITH GLOBAL CENTROID ==============
def get_feat_map_centered(path_to_dir: str):
    dir_list = os.listdir(path_to_dir)
    sdf_list = [os.path.join(path_to_dir, f) for f in dir_list if f.endswith('.sdf')]
    
    # First pass: load all mols and find global centroid
    mol_list = []
    all_centroids = []
    for sdf_path in sdf_list:
        sup = Chem.SDMolSupplier(sdf_path)
        for m in sup:
            if m is not None:
                mol_list.append(m)
                coords = m.GetConformer(0).GetPositions()
                all_centroids.append(coords.mean(axis=0))
    
    # Global centroid across all molecules
    global_centroid = np.mean(all_centroids, axis=0)
    print(f"    Global centroid: {global_centroid}")
    
    # Second pass: shift all mols by the SAME offset
    for mol in mol_list:
        conf = mol.GetConformer(0)
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (pos.x - global_centroid[0], 
                                      pos.y - global_centroid[1], 
                                      pos.z - global_centroid[2]))
    
    feat_list = []
    for m in mol_list:
        raw_feats = fdef.GetFeaturesForMol(m)
        feat_list.append([f for f in raw_feats if f.GetFamily() in keep])
    
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in feat_list]
    
    print(f"    Loaded {len(mol_list)} molecules, extracted {sum(len(f) for f in feat_list)} features")
    return feat_list, fms, mol_list, global_centroid


# ============== CLUSTERING BY FEATURE TYPE ==============
def cluster_features_by_type(feat_list, eps=0.5, min_samples=10):
    all_coords = []
    mol_indices = []
    feat_indices = []
    feat_types = []
    
    for mol_idx, feats in enumerate(feat_list):
        for feat_idx, f in enumerate(feats):
            all_coords.append(f.GetPos())
            mol_indices.append(mol_idx)
            feat_indices.append(feat_idx)
            feat_types.append(f.GetFamily())
    
    X = np.array([[c.x, c.y, c.z] for c in all_coords])
    feat_types = np.array(feat_types)
    mol_indices = np.array(mol_indices)
    feat_indices = np.array(feat_indices)
    
    all_labels = np.full(len(X), -1)
    cluster_offset = 0
    
    for feat_type in keep:
        mask = feat_types == feat_type
        if mask.sum() < min_samples:
            continue
        
        X_subset = X[mask]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_subset)
        
        labels_subset = clustering.labels_.copy()
        labels_subset[labels_subset != -1] += cluster_offset
        all_labels[mask] = labels_subset
        
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        cluster_offset += n_clusters
        print(f"    {feat_type}: {mask.sum()} features -> {n_clusters} clusters")
    
    return all_labels, mol_indices, feat_indices, X, feat_types


# ============== HOTSPOT EXTRACTION ==============
def get_cluster_hotspots(X, labels, mol_indices, feat_indices, feat_list, min_count=10):
    cluster_centers = []
    cluster_radii = []
    cluster_ids = []
    cluster_counts = []
    cluster_features = []
    
    unique_labels = [l for l in np.unique(labels) if l != -1]
    
    for label in unique_labels:
        mask = labels == label
        count = mask.sum()
        if count < min_count:
            continue
        
        points = X[mask]
        center = points.mean(axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1))
        
        feat_types_in_cluster = []
        for i, is_in_cluster in enumerate(mask):
            if is_in_cluster:
                mol_idx = mol_indices[i]
                feat_idx = feat_indices[i]
                feat_types_in_cluster.append(feat_list[mol_idx][feat_idx].GetFamily())
        
        dominant_feat = Counter(feat_types_in_cluster).most_common(1)[0][0]
        
        cluster_centers.append(center)
        cluster_radii.append(radius)
        cluster_ids.append(label)
        cluster_counts.append(count)
        cluster_features.append(dominant_feat)
    
    sorted_idx = np.argsort(cluster_counts)[::-1]
    cluster_centers = np.array(cluster_centers)[sorted_idx]
    cluster_radii = [cluster_radii[i] for i in sorted_idx]
    cluster_ids = [cluster_ids[i] for i in sorted_idx]
    cluster_counts = [cluster_counts[i] for i in sorted_idx]
    cluster_features = [cluster_features[i] for i in sorted_idx]
    
    return cluster_centers, cluster_radii, cluster_ids, cluster_counts, cluster_features


# ============== CLUSTER COMPOSITION ==============
def get_cluster_composition(labels, mol_indices, feat_indices, feat_list):
    cluster_info = {}
    
    for i, label in enumerate(labels):
        if label == -1:
            continue
        
        mol_idx = mol_indices[i]
        feat_idx = feat_indices[i]
        feat_type = feat_list[mol_idx][feat_idx].GetFamily()
        
        if label not in cluster_info:
            cluster_info[label] = []
        cluster_info[label].append(feat_type)
    
    cluster_summary = {}
    for label, feat_types_list in cluster_info.items():
        cluster_summary[label] = dict(Counter(feat_types_list))
    
    return cluster_summary


# ============== DATASET ANALYSIS ==============
def analyse_dataset(mol_list):
    all_feat_counts = []
    
    for mol in mol_list:
        raw_feats = fdef.GetFeaturesForMol(mol)
        mol_feats = [f for f in raw_feats if f.GetFamily() in keep]
        
        feat_counts = {}
        for f in mol_feats:
            ft = f.GetFamily()
            feat_counts[ft] = feat_counts.get(ft, 0) + 1
        all_feat_counts.append(feat_counts)
    
    print(f"\n    Analysed {len(all_feat_counts)} molecules\n")
    
    target_profile = {}
    for feat_type in keep:
        counts = [fc.get(feat_type, 0) for fc in all_feat_counts]
        median = int(np.median(counts))
        std = max(1, int(np.std(counts)))
        target_profile[feat_type] = (median, std)
        print(f"    {feat_type}: median={median}, std={std}, range=[{np.min(counts)}, {np.max(counts)}]")
    
    return target_profile


# ============== PLOTTING ==============
def plot_cluster_summary(cluster_ids, cluster_counts, cluster_features, output_path='hotspot_summary.png'):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    feature_colours = {
        'Donor': 'blue',
        'Acceptor': 'red',
        'Aromatic': 'orange',
        'NegIonizable': 'green',
        'PosIonizable': 'purple',
        'ZnBinder': 'cyan',
        'Hydrophobe': 'brown',
        'LumpedHydrophobe': 'pink'
    }
    
    colours = [feature_colours.get(f, 'grey') for f in cluster_features]
    xlabels = [f'C{cid} ({feat})' for cid, feat in zip(cluster_ids, cluster_features)]
    
    ax.bar(range(len(cluster_counts)), cluster_counts, color=colours)
    ax.set_xticks(range(len(cluster_counts)))
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_xlabel('Cluster (Feature Type)')
    ax.set_ylabel('Count')
    ax.set_title('Pharmacophore Hotspots by Density (Global Centered)')
    
    from matplotlib.patches import Patch
    present_features = list(set(cluster_features))
    legend_handles = [Patch(color=feature_colours[f], label=f) for f in present_features]
    ax.legend(handles=legend_handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    
    return fig, ax


def plot_all_features(X, labels, mol_indices, feat_indices, feat_list, min_count=10, output_path='features_3d.png'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    feature_colours = {
        'Donor': 'blue',
        'Acceptor': 'red',
        'Aromatic': 'orange',
        'NegIonizable': 'green',
        'PosIonizable': 'purple',
        'ZnBinder': 'cyan',
        'Hydrophobe': 'brown',
        'LumpedHydrophobe': 'pink'
    }
    
    feature_points = {ft: [] for ft in feature_colours}
    
    for i, label in enumerate(labels):
        if label == -1:
            continue
        mol_idx = mol_indices[i]
        feat_idx = feat_indices[i]
        feat_type = feat_list[mol_idx][feat_idx].GetFamily()
        if feat_type in feature_colours:
            feature_points[feat_type].append(X[i])
    
    for feat_type, points in feature_points.items():
        if len(points) >= min_count:
            points = np.array(points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c=feature_colours[feat_type], label=f'{feat_type} (n={len(points)})', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('All Features (Global Centered)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    
    return fig, ax


# ============== MAIN ==============
if __name__ == "__main__":
    # PARAMETERS - EDIT THESE
    SDF_DIR = '/Users/sanazkazeminia/Documents/mol_test_suite/data/project_data/data/AMPC_beta_lactamase/FEATURE_MAP_ALIGNED/'
    OUTPUT_PKL = 'ampc_hotspot_global_centered_eps0.5_min10.pkl'
    EPS = 0.5
    MIN_SAMPLES = 10
    MIN_COUNT = 10
    
    print("=" * 60)
    print("DBSCAN Pharmacophore Hotspot Generation")
    print("(Global Centroid Centering - Preserves Alignment)")
    print("=" * 60)
    
    # 1. Load and center with global centroid
    print("\n[1] Loading molecules and centering with global centroid...")
    feat_list, fms, mol_list, global_centroid = get_feat_map_centered(SDF_DIR)
    
    # 2. Cluster by feature type
    print("\n[2] Clustering features by type...")
    labels, mol_indices, feat_indices, X, feat_types = cluster_features_by_type(feat_list, eps=EPS, min_samples=MIN_SAMPLES)
    
    # 3. Get composition
    print("\n[3] Cluster composition:")
    summary = get_cluster_composition(labels, mol_indices, feat_indices, feat_list)
    for cluster_id in sorted(summary.keys(), reverse=True):
        print(f"    Cluster {cluster_id}: {summary[cluster_id]}")
    
    # 4. Extract hotspots
    print("\n[4] Extracting hotspots...")
    cluster_centers, cluster_radii, cluster_ids, cluster_counts, cluster_features = get_cluster_hotspots(
        X, labels, mol_indices, feat_indices, feat_list, min_count=MIN_COUNT
    )
    
    print("\n    Hotspot Summary:")
    for i in range(len(cluster_ids)):
        print(f"    Cluster {cluster_ids[i]}: {cluster_features[i]}, count={cluster_counts[i]}, radius={cluster_radii[i]:.2f}A, center={cluster_centers[i]}")
    
    # 5. Target profile
    print("\n[5] Analysing dataset for target profile...")
    target_profile = analyse_dataset(mol_list)
    
    # 6. Plot
    print("\n[6] Plotting...")
    plot_cluster_summary(cluster_ids, cluster_counts, cluster_features)
    plot_all_features(X, labels, mol_indices, feat_indices, feat_list, min_count=MIN_COUNT)
    
    # 7. Save
    print("\n[7] Saving pkl...")
    hotspot_data = {
        'cluster_centers': cluster_centers,
        'cluster_radii': cluster_radii,
        'cluster_ids': cluster_ids,
        'cluster_counts': cluster_counts,
        'cluster_features': cluster_features,
        'target_profile': target_profile,
        'keep': keep,
        'global_centroid': global_centroid,
        'metadata': {
            'eps': EPS,
            'min_samples': MIN_SAMPLES,
            'sigma': 0.5,
            'cutoff_distance': 2.0,
            'n_clusters': len(cluster_centers),
            'n_molecules_analysed': len(mol_list),
            'source_dir': SDF_DIR,
            'centering_method': 'global_centroid',
            'global_centroid': global_centroid.tolist(),
            'notes': 'Clustered by feature type. All ligands shifted by same global centroid to preserve alignment.'
        }
    }
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(hotspot_data, f)
    
    print(f"    Saved to {OUTPUT_PKL}")
    print(f"    Global centroid used: {global_centroid}")
    print(f"    Number of clusters: {len(cluster_centers)}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)