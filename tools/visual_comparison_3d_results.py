import os
import SimpleITK as sitk
import numpy as np
import plotly.graph_objects as go

def show_3d(image_np, threshold=0.5):
    verts = np.argwhere(image_np > threshold)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=1, color=z, colorscale='Viridis', opacity=0.5)
    )])
    fig.update_layout(scene=dict(aspectmode='data'))
    return fig

def process_folder(nii_folder, output_folder, threshold=0):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(nii_folder):
        if filename.endswith('.nii.gz'):
            path = os.path.join(nii_folder, filename)
            img = sitk.ReadImage(path)
            img_np = sitk.GetArrayFromImage(img)  # shape: [z, y, x]
            img_np = np.transpose(img_np, (1, 2, 0))  # to [x, y, z]
            fig = show_3d(img_np, threshold)
            out_file = os.path.join(output_folder, filename.replace('.nii.gz', '.html'))
            fig.write_html(out_file)
            print(f"Saved 3D view for {filename} -> {out_file}")

# 示例调用
nii_folder = r"E:\view\MedNeXt"
output_folder = r"D:\output\3d_views"
process_folder(nii_folder, output_folder, threshold=0.5)
