import os
from load_datadir_re  import load_datadir_re
from myPMS import myPMS
from myPCA import myPCA
from myRobustPMS import myRobustPMS
# from mynormal2depth import 
from mynormal2depth import mynormal2depth, save_results

dataFormat = 'PNG'

dataNameStack = ['bear', 'cat', 'pot', 'buddha']

for testId in range(4):
    dataName = f"{dataNameStack[testId]}{dataFormat}"
    datadir = os.path.join('../pmsData', dataName)
    bitdepth = 16
    gamma = 1
    resize = (512,512)
    data = load_datadir_re(datadir, bitdepth, resize, gamma)
    data.name = dataNameStack[testId]  # 设置数据集名称

    # # 1.2.1  Least Squares-Based Method
    # Normal = myPMS(data, resize)
    # #  save Normal mat and png file to results folder
    # print(f"Normal map for {dataNameStack[testId]} saved.")

    # 1.2.1  Robust Photometric Stereo
    Normal = myRobustPMS(data,resize)
    #  save Normal mat and png file to results folder
    
    # 1.2.2 PCA-Based Method
    # Normal = myPCA(data, resize)
    #  save Normal mat and png file to results folder

    # # 1.2.3 Calculate the depth map and convert it into mesh
    # pc = mynormal2depth(Normal, data)
    # 1.2.3 Convert PC to mesh
    depth_map, mesh = mynormal2depth(Normal, data)
    # 1.2.3 Save results mesh in standard mesh format
    depth_map_path, mesh_path = save_results(depth_map, mesh, data, "pms")
    print("Mesh statistics:")
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Number of triangles: {len(mesh.triangles)}")