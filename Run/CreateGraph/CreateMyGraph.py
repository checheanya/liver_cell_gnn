import os
import warnings

import cv2
import dgl
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from dgl.data.utils import save_graphs
from fuzzywuzzy import process
from histocartography.visualization import OverlayGraphVisualization
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from options import parse_args

warnings.filterwarnings('ignore')
np.random.seed(123)

CellTypes = ['Tumor cells',
             'Vascular endothelial cells',
             'Lymphocytes',
             'Fibroblasts',
             'Biliary epithelial cells',
             'Hepatocytes',
             'Others']


def img_energy(GrayImage):
    tmp = np.abs(cv2.Sobel(GrayImage, cv2.CV_32F, 1, 1))
    energy = tmp.sum() / (2048 * 2048)
    return energy

from collections import defaultdict

def replace_with_best_match_vectorized(cell_names):
    unique_names = cell_names.unique()
    best_matches = {}

    for name in unique_names:
        best_match, score = process.extractOne(name, CellTypes)
        if score > 0:  # Adjust this threshold based on your needs
            best_matches[name] = best_match
        else:
            best_matches[name] = name

    return cell_names.map(best_matches)

def load_txt(LabelDataPath, Patient):
    inTXTDataPath = os.path.join(LabelDataPath, Patient, Patient + '.txt')
    inTXTData = pd.read_csv(inTXTDataPath, sep='\t', engine='python', encoding='utf-8')

    inTXTData['Class'] = replace_with_best_match_vectorized(inTXTData['Class'])
    print('Finished load_txt')
    return inTXTData


def Distance(x, y):
    return pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2)


def polymerization(inTXTData, tr):
    final_data = pd.DataFrame(columns=inTXTData.columns)
    for type in CellTypes:
        data = inTXTData.loc[inTXTData['Class'] == type]
        if data.shape[0] == 0:
            continue
        data = data.reset_index(drop=True)
        # print("count before", len(data))
        i = 0
        length = data.shape[0]
        while i < length:
            j = i + 1
            while j < length:
                # print(i,j,data.index)
                x = pd.concat([data.iloc[i, 5: 13], data.iloc[i, 25: 31]], axis=0)
                y = pd.concat([data.iloc[j, 5: 13], data.iloc[j, 25: 31]], axis=0)
                x = np.array(x)
                y = np.array(y)
                dis = Distance(x, y)
                if dis <= tr:
                    # print(cos_sim)
                    new_feature = (x + y) / 2
                    data.drop(axis=0, index=j, inplace=True)
                    length -= 1
                    data.iloc[i][5:13] = new_feature[:8]
                    data.iloc[i][25:31] = new_feature[8:]
                    data = data.reset_index(drop=True)
                    j -= 1
                j += 1
            i += 1
        # print("count after", len(data))
        final_data = pd.concat([final_data, data], axis=0)
    return final_data


def get_range_for_every_feature(LabelDataPath):
    """
    Min/max per feature across all patients' TXT tables (for later normalization).
    :param LabelDataPath: Root folder of labeling projects
    :return: Array shape (2, 41): row 0 = min, row 1 = max; one column per feature
    """
    Patients = os.listdir(LabelDataPath)
    MinAndMax = np.empty(shape=(2, 41))
    MinAndMax[0, :] = 10000
    MinAndMax[1, :] = -10000
    for Patient in Patients:
        TXTData = load_txt(LabelDataPath, Patient)
        Features = TXTData.iloc[:, 7:]
        for i in range(41):
            if Features.min()[i] < MinAndMax[0, i]:
                MinAndMax[0, i] = Features.min()[i]
            if Features.max()[i] > MinAndMax[1, i]:
                MinAndMax[1, i] = Features.max()[i]

    print('Finished get_range_for_every_feature')
    return MinAndMax


def load_image(LabelDataPath, Patient):
    """
    Load cell-label and nuclei-label images for the patient.
    :param LabelDataPath: Root folder of all labeling projects
    :param Patient: Patient ID
    :return: NumPy arrays for cell labels and nuclei labels
    """
    Image.MAX_IMAGE_PIXELS = None
    LabeledCellImagePath = os.path.join(LabelDataPath, Patient, Patient + '-CellLabels.png')
    LabeledCellImage = Image.open(LabeledCellImagePath)
    LabeledCellImage = np.array(LabeledCellImage)
    LabeledNucleiImagePath = os.path.join(LabelDataPath, Patient, Patient + '-NucleiLabels.png')
    LabeledNucleiImage = Image.open(LabeledNucleiImagePath)
    LabeledNucleiImage = np.array(LabeledNucleiImage)

    print('Finished load_image')
    return LabeledCellImage, LabeledNucleiImage,


def load_wsi(WSIDataPath, Patient):
    """
    Load WSI for the patient.
    :param WSIDataPath: Root folder of all WSI files
    :param Patient: Patient ID
    :return: OpenSlide handle; use slide.read_region then np.array(...) for pixels
    """
    path = os.path.join(WSIDataPath, Patient, Patient + '.ndpi')
    slide = openslide.OpenSlide(path)

    print('Finished load_wsi')
    return slide


def find_patch_boxes_new(LabeledCellImage, patch_size=256, ratio=0.1):
    """
    Select patches with the most diverse cell types from the labeled cell image.
    :param LabeledCellImage: Cell type label image
    :param patch_size: Patch side length in pixels
    :param ratio: patches count / total area (see usage)
    :return:
    """
    # Foreground mask
    CellImage = ((LabeledCellImage > 0) * 255).astype('uint8')
    close_kernel = np.ones((50, 50), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(CellImage), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((50, 50), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    # Total foreground area
    image_binary = np.array(image_open > 0, dtype='int')
    S = image_binary.sum()

    # Foreground bounding box
    contours, _ = cv2.findContours(image_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = np.array([cv2.boundingRect(c) for c in contours])

    xmin = np.min(boundingBoxes[:, 0])
    xmax = np.max(boundingBoxes[:, 0] + boundingBoxes[:, 2])
    ymin = np.min(boundingBoxes[:, 1])
    ymax = np.max(boundingBoxes[:, 1] + boundingBoxes[:, 3])
    boundingBox = [xmin, ymin, xmax, ymax]

    boundingBox = [ymin, xmin, ymax, xmax]

    # Sliding window: patch bounding boxes
    box_list = []
    for i in np.arange(xmin, xmax - patch_size, patch_size):
        for j in np.arange(ymin, ymax - patch_size, patch_size):
            # Note: cv2 uses (row, col) vs image x/y
            box = (j, i, j + patch_size, i + patch_size)
            box_list.append(box)

    # Count distinct cell types per patch
    CellTypeNum = []
    for box in box_list:
        # Here x/y map to vertical/horizontal image axes
        patch_xmin, patch_ymin, patch_xmax, patch_ymax = box
        patch = LabeledCellImage[patch_xmin: patch_xmax, patch_ymin: patch_ymax]
        CellTypeNum.append(len(np.unique(patch)) - 1)
        #CellTypeNum.append(np.sum(patch > 0))
    coords_list = pd.DataFrame(box_list)
    CellTypeNum = pd.DataFrame(CellTypeNum)
    coordinates = pd.concat([coords_list, CellTypeNum], axis=1)
    coordinates.columns = ['x_min', 'y_min', 'x_max', 'y_max', 'num']
    coordinates = coordinates.sort_values(by='num', ascending=False)

    coordinates = coordinates.iloc[: 5000, :4]

    # Take top patches only
    SelectedCords = coordinates.iloc[:32, :4]
    if SelectedCords.shape[0] < 32:
        print('Warning: The number of patches is less than 32')
        raise ValueError('The number of patches is less than 32')

    # coordinates = np.array(coordinates)
    # energy = []
    # for coordinate in tqdm(coordinates, desc='Windows loop'):
    #     try:
    #         img = np.array(WSIData.read_region((int(coordinate[1]) * 4, int(coordinate[0]) * 4), 2, (512, 512)))[:, :,
    #               :3]
    #         GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         _, b = cv2.threshold(GrayImage, 170, 255, cv2.THRESH_BINARY)
    #         if b.mean() > 150 or b.mean() < 20:
    #             eg = 0.0
    #             energy.append(eg)
    #             continue
    #         eg = img_energy(GrayImage)
    #
    #     except openslide.lowlevel.OpenSlideError as e:
    #         eg = 0.0
    #     energy.append(eg)
    #
    # coords_list = pd.DataFrame(coordinates)
    # energy_list = pd.DataFrame(energy)
    # coordinates = pd.concat([coords_list, energy_list], axis=1)
    # coordinates.columns = ['x_min', 'y_min', 'x_max', 'y_max', 'energy']
    # coordinates = coordinates.sort_values(by='energy', ascending=False)
    # SelectedCords = coordinates.iloc[: 128, :4]

    SelectedCords = np.array(SelectedCords)
    imgs = []
    # for coordinate in tqdm(SelectedCords, desc='Image'):
    #     # img = np.array(WSIData.read_region((int(coordinate[1] * 4), int(coordinate[0] * 4)), 2, (patch_size, patch_size)))[:, :, :3]
    #     img = np.array(WSIData.read_region((int(coordinate[1] * 4), int(coordinate[0] * 4)), 0, (patch_size * 4, patch_size * 4)))[:, :, :3]
    #     imgs.append(img)
    #     # plt.imshow(img)
    #     # plt.show()
    return SelectedCords, imgs



def get_origin_image(WSIData, level_dimensions):
    """
    Read RGB image at OpenSlide level `level_dimensions` as a NumPy array.
    :param WSIData: OpenSlide handle
    :param level_dimensions: Pyramid level index, typically [0-8]
    :return: NumPy RGB image
    """
    (m, n) = WSIData.level_dimensions[level_dimensions]
    OriginImage = np.array(WSIData.read_region((0, 0), level_dimensions, (m, n)))[:, :, :3]

    print('Finished get_origin_image')
    return OriginImage


def get_cell_coordinate_pixel(CellData, box):
    """
    Map CellData centroid coordinates to pixel coordinates in the patch.
    :param CellData: Cell table for the patch
    :param box: Patch box in micrometers
    :return: List of (x, y) pixel coordinates per cell
    """
    CoordinateUm = [list(CellData['Centroid X µm']), list(CellData['Centroid Y µm'])]
    # CoordinatePixelX = ((CoordinateUm[0] - box[1]) / (Ratio * 4)).astype('int')
    # CoordinatePixelY = ((CoordinateUm[1] - box[0]) / (Ratio * 4)).astype('int')
    CoordinatePixelX = (CoordinateUm[0])
    CoordinatePixelY = (CoordinateUm[1])
    CoordinatePixel = []
    for i in range(len(CellData)):
        CoordinatePixel.append((CoordinatePixelX[i], CoordinatePixelY[i]))
    CoordinatePixel = list(CoordinatePixel)
    return CoordinatePixel


# def generate_graph(CellData, Centroids, Features, PatchSize=1024, k=5, thresh=100):
#     """
#     Build graph with KNN edges.
#     :param Centroids: Node coordinates, list
#     :param Features: Node feature tensor
#     :param PatchSize: Patch size, int
#     :param k: K in KNN
#     :param thresh: No edge if distance exceeds this
#     :return: dgl.graph
#     """
#     graph = dgl.DGLGraph()
#     graph.add_nodes(len(Centroids))
#
#     image_size = (PatchSize, PatchSize)
#     # Node centroid positions
#     graph.ndata['centroid'] = torch.FloatTensor(Centroids) * 4
#     # Node features (includes normalized coordinates)
#     centroids = graph.ndata['centroid']
#     normalized_centroids = torch.empty_like(centroids)
#     normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
#     normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]
#
#     if Features.ndim == 3:
#         normalized_centroids = normalized_centroids \
#             .unsqueeze(dim=1) \
#             .repeat(1, Features.shape[1], 1)
#         concat_dim = 2
#     elif Features.ndim == 2:
#         concat_dim = 1
#
#     concat_features = torch.cat(
#         (
#             Features,
#             normalized_centroids
#         ),
#         dim=concat_dim,
#     )
#
#     graph.ndata['feat'] = concat_features[:, 2:]
#     name_tensor = []
#     for item in CellData['Class']:
#         Flag = 1
#         for i in range(len(CellTypes)):
#             if item == CellTypes[i]:
#                 name_tensor.append([i + 1])
#                 Flag = 0
#         if Flag:
#             name_tensor.append([7])  # Unlisted cell types -> Other (index 7)
#     name_tensor = torch.tensor(name_tensor)
#     graph.ndata['name'] = name_tensor
#     # KNN graph
#     if Features.shape[0] != 1:
#         k = min(Features.shape[0] - 1, k)
#         adj = kneighbors_graph(
#             centroids,
#             k,
#             mode="distance",
#             include_self=False,
#             metric="euclidean").toarray()
#
#         if thresh is not None:
#             adj[adj > thresh] = 0
#
#         edge_list = np.nonzero(adj)
#         graph.add_edges(list(edge_list[0]), list(edge_list[1]))
#
#     return graph

# import dgl
# import torch
# import numpy as np
# from scipy.spatial import Delaunay
#
# def generate_graph(CellData, Centroids, Features, PatchSize=1024, thresh=100):
#     """
#     Build graph with Delaunay triangulation.
#     :param Centroids: Node coordinates, list
#     :param Features: Node feature tensor
#     :param PatchSize: Patch size, int
#     :param thresh: No edge if distance exceeds this
#     :return: dgl.graph
#     """
#     graph = dgl.DGLGraph()
#     graph.add_nodes(len(Centroids))
#
#     image_size = (PatchSize, PatchSize)
#     # Node centroid positions
#     graph.ndata['centroid'] = torch.FloatTensor(Centroids) * 4
#     # Node features (includes normalized coordinates)
#     centroids = graph.ndata['centroid']
#     normalized_centroids = torch.empty_like(centroids)
#     normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
#     normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]
#
#     if Features.ndim == 3:
#         normalized_centroids = normalized_centroids \
#             .unsqueeze(dim=1) \
#             .repeat(1, Features.shape[1], 1)
#         concat_dim = 2
#     elif Features.ndim == 2:
#         concat_dim = 1
#
#     concat_features = torch.cat(
#         (
#             Features,
#             normalized_centroids
#         ),
#         dim=concat_dim,
#     )
#
#     graph.ndata['feat'] = concat_features[:, 2:]
#     name_tensor = []
#     for item in CellData['Class']:
#         Flag = 1
#         for i in range(len(CellTypes)):
#             if item == CellTypes[i]:
#                 name_tensor.append([i + 1])
#                 Flag = 0
#         if Flag:
#             name_tensor.append([7])  # Unlisted cell types -> Other (index 7)
#     name_tensor = torch.tensor(name_tensor)
#     graph.ndata['name'] = name_tensor
#
#     # Delaunay graph
#     tri = Delaunay(centroids.numpy())
#     for simplex in tri.simplices:
#         for i in range(3):
#             for j in range(i+1, 3):
#                 dist = np.linalg.norm(centroids[simplex[i]].numpy() - centroids[simplex[j]].numpy())
#                 if thresh is None or dist <= thresh:
#                     graph.add_edge(simplex[i], simplex[j])
#                     graph.add_edge(simplex[j], simplex[i])
#
#     return graph

import dgl
import torch
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

def generate_graph(CellData, Centroids, Features, PatchSize=1024):
    """
    Build graph using a minimum spanning tree (MST).
    :param Centroids: Node coordinates, list
    :param Features: Node feature tensor
    :param PatchSize: Patch size, int
    :return: dgl.graph
    """
    graph = dgl.DGLGraph()
    graph.add_nodes(len(Centroids))

    image_size = (PatchSize, PatchSize)
    # Node centroid positions
    graph.ndata['centroid'] = torch.FloatTensor(Centroids)
    # Node features (includes normalized coordinates)
    # centroids = graph.ndata['centroid']

    normalized_centroids = torch.empty_like(graph.ndata['centroid'])
    centroids = normalized_centroids
    centroids[:, 0] = Features[:, 0]
    centroids[:, 1] = Features[:, 1]
    normalized_centroids[:, 0] = Features[:, 0] / 256
    normalized_centroids[:, 1] = Features[:, 1] / 256

    if Features.ndim == 3:
        normalized_centroids = normalized_centroids \
            .unsqueeze(dim=1) \
            .repeat(1, Features.shape[1], 1)
        concat_dim = 2
    elif Features.ndim == 2:
        concat_dim = 1

    concat_features = torch.cat(
        (
            Features,
            normalized_centroids
        ),
        dim=concat_dim,
    )

    graph.ndata['feat'] = concat_features[:, 2:]
    name_tensor = []
    for item in CellData['Class']:
        Flag = 1
        for i in range(len(CellTypes)):
            if item == CellTypes[i]:
                name_tensor.append([i + 1])
                Flag = 0
        if Flag:
            name_tensor.append([7])  # Unlisted cell types -> Other (index 7)
    name_tensor = torch.tensor(name_tensor)
    graph.ndata['name'] = name_tensor

    # MST graph
    distances = cdist(centroids.numpy(), centroids.numpy(), metric='euclidean')
    mst = minimum_spanning_tree(distances).tocoo()
    graph.add_edges(mst.row, mst.col)

    return graph


# Append one-hot cell-type features
def concat_one_hot(Features, CellData):
    temp = []
    # print(Features.shape)
    # print(CellData.shape)
    for item in CellData['Class']:
        tmp = []
        for i in range(len(CellTypes)):
            if item == CellTypes[i]:
                tmp.append(1)
            else:
                tmp.append(0)
        temp.append(tmp)
    temp = torch.tensor(temp)
    Features = torch.cat((Features, temp), dim=1)
    return Features


def generate_and_save_cell_graphs(box, TXTData, Label, OutPath):
    """
    For each patch box, build one combined cell graph and save to disk.
    :param box: Patch bounds in micrometers
    :param TXTData: TXT table exported from pathologist annotations
    :param Label: Labels dict passed to save_graphs (e.g. survival)
    :param OutPath: Output directory for this patch (graph files)
    """
    # Cells inside the box
    CellINPatch = TXTData[(box[1] < TXTData['Centroid X µm']) & (TXTData['Centroid X µm'] < box[3]) &
                          (box[0] < TXTData['Centroid Y µm']) & (TXTData['Centroid Y µm'] < box[2])]

    # Optionally filter irrelevant cells here

    # CellINPatch=polymerization(CellINPatch,5000)
    # Cells in the box, grouped by class downstream
    print(CellINPatch.shape)
    CellData = CellINPatch
    # Proceed if there is at least one cell (original: require >5 nodes)
    if len(CellData) > 0:
        CoordinatePixel = get_cell_coordinate_pixel(CellData, box)
        # Morphology columns from TXT (exclude color-related columns)
        Features = pd.concat([CellData.iloc[:, 5: 13], CellData.iloc[:, 13:]], axis=1)
        Features['Centroid X µm'] = ((Features['Centroid X µm'] - box[1]) / Ratio / 4).astype('int')
        Features['Centroid Y µm'] = ((Features['Centroid Y µm'] - box[0]) / Ratio / 4).astype('int')

        # Features = Features.drop_duplicates(subset=['Centroid X µm', 'Centroid Y µm'], keep='first')
        Features = np.array(Features, dtype='float64')
        # NormalizedFeatures =
        Features = torch.from_numpy(Features)  # Features should be normalized
        # print("before one-hot", Features.shape, CellData.shape)
        # One-hot encode cell types
        Features = concat_one_hot(Features, CellData)
        # print("after one-hot", Features.shape, CellData.shape, len(CoordinatePixel))
        Graph = generate_graph(CellData, CoordinatePixel, Features)
        img = Image.open(OutPath + "/wsi.png")
        img = np.array(img)
        visualizer = OverlayGraphVisualization(
            node_style='fill',
            node_radius=3,
            edge_thickness=1,
            colormap='coolwarm',
            show_colormap=True,
            min_max_color_normalize=False
        )
        canvas = visualizer.process(img, Graph)
        canvas.save(
            OutPath + "/node_in_img.png",
            quality=100
        )
        print("hello")
        # Original note: require enough edges in the graph
        if Graph.num_edges() > -1:
            GraphName = 'AllCell.bin'
            GraphPath = os.path.join(OutPath, GraphName)
            save_graphs(GraphPath, Graph, Label)

    # print('Finished generate_graph')


def graph_visualize(box, WSIImage, graph):
    """
    Visualize graph overlay for inspection and hyperparameter tuning.
    :param box: Patch bounds in micrometers
    :param WSIImage: Downsampled WSI as NumPy array (e.g. level 4)
    :param graph: DGL graph to draw
    """
    boxpixel = (box / Ratio / 4).astype('int')
    image = WSIImage[boxpixel[1]: boxpixel[3], boxpixel[0]: boxpixel[2]]
    visualizer = OverlayGraphVisualization(node_radius=1,
                                           edge_thickness=1,
                                           )
    canvas = visualizer.process(image, graph)
    canvas.show()


def show_big_array(array):
    """
    Display a very large array via PIL (avoids matplotlib freezing on huge images).
    :param array: Image array
    """
    p = Image.fromarray(array)
    p.show()


def robust_read_csv(file_path, sep='\t'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    encoding_options = ['utf-8', 'latin1', 'iso-8859-1', 'utf-16']
    for encoding in encoding_options:
        try:
            data = pd.read_csv(file_path, sep=sep, engine='python', encoding=encoding)
            print(f"Successfully read the file with encoding: {encoding}")
            return data
        except Exception as e:
            print(f"Failed to read the file with encoding: {encoding}. Error: {e}")

    raise ValueError("Failed to read the file with any of the tried encodings.")


if __name__ == '__main__':
    opt = parse_args()
    print('\n')
    # Load paths
    LabelDataPath = opt.label_data_path  # Root of labeling projects
    WSIDataPath = opt.WSI_data_path  # Root of WSI files
    # FollowUpData = pd.read_csv(opt.follow_up_data, sep='\t', engine='python', encoding='utf-8')  # Follow-up table
    FollowUpData = robust_read_csv(opt.follow_up_data, sep='\t')
    FollowUpData.dropna(axis=0, how='all')
    # Patients = os.listdir(LabelDataPath)

    # Patients with region annotations
    Patients = os.listdir(LabelDataPath)
    Patients.sort()
    # FeaturesMinAndMax = np.loadtxt('MinAndMax.csv')
    result_dataset = '/data/yuanyz/datasets/32_patches_graphs'
    Patients = os.listdir('/data0/yuanyz/NewGraph/datasets/patientminimum_spanning_tree256412/test')
    for Patient in tqdm(Patients):
        if os.path.exists(os.path.join(result_dataset, Patient)) is False:
            os.makedirs(os.path.join(result_dataset, Patient))
            # TXTData = load_txt('./txtdata/PCA3', Patient)  # Load TXT table
            TXTData = load_txt(LabelDataPath, Patient)  # Load TXT table
            LabeledCellImage, LabeledNucleiImage = load_image(LabelDataPath, Patient)  # Load label images
            WSIData = load_wsi(WSIDataPath, Patient)  # Load WSI
            # WSIImage = get_origin_image(WSIData, level_dimensions=2)

            Ratio = float(WSIData.properties['openslide.mpp-x'])
            # Patches with most cell-type diversity
            SelectedPatchBoxes, imgs = find_patch_boxes_new(LabeledCellImage, ratio=Ratio)
            SelectedPatchBoxes = np.array(SelectedPatchBoxes)

            # Build graphs
            Ratio = float(WSIData.properties['openslide.mpp-x'])

            SelectedPatchBoxesUm = SelectedPatchBoxes * Ratio * 4  # Pixel boxes to physical units
            for idx, box in enumerate(SelectedPatchBoxesUm):
                BoxName = str(box[0]) + '_' + str(box[1])
                OutPath = os.path.join(result_dataset, Patient, BoxName)
                if not os.path.exists(OutPath):
                    os.makedirs(OutPath)
                # plt.imsave(OutPath + "/wsi.png", imgs[idx])
                PatientFollowUp = FollowUpData[FollowUpData['标本号'] == int(Patient)]
                if not PatientFollowUp.empty:
                    SurvLabel = {
                        'CoxLabel': torch.tensor([(float(PatientFollowUp['无瘤/月']), float(PatientFollowUp['复发']))]),
                    'SurvLabel': torch.tensor([(float(PatientFollowUp['生存/月']), float(PatientFollowUp['死亡']))])}
                    ImgSave = True
                    generate_and_save_cell_graphs(box, TXTData, SurvLabel, OutPath)
                else:
                    PatientFollowUp = FollowUpData[FollowUpData['标本号'] == str(Patient)]
                    if not PatientFollowUp.empty:
                        SurvLabel = {
                            'CoxLabel': torch.tensor(
                                [(float(PatientFollowUp['无瘤/月']), float(PatientFollowUp['复发']))]),
                        'SurvLabel': torch.tensor([(float(PatientFollowUp['生存/月']), float(PatientFollowUp['死亡']))])}
                        generate_and_save_cell_graphs(box, TXTData, SurvLabel, OutPath)
            print('Finished generate graph for ' + Patient)

        else:
            continue

    # Visualization (optional)
    # Graph = dgl.load_graphs(os.path.join(OutPath, 'Tumor cells.bin'))[0][0]
    # graph_visualize(box, WSIImage, Graph)
