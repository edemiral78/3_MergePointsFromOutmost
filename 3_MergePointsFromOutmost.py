import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import folium
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style
from geographiclib.geodesic import Geodesic
from geopy import distance
from scipy.spatial import cKDTree
from shapely.geometry import Point
from tqdm import tqdm


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def DefineFolders():
    InputFolder = 'Input'
    OutputFolder = 'Output'
    createFolder('{}'.format(OutputFolder))
    return InputFolder, OutputFolder

def ReadDataSets(FileName1, FileName2):
    #FileName1 = 'Points_DataSet'  # PointID, Lat, lon, Ağırlık: İçerdiği Nokta Sayısı, Birleşim Sayısı
    Points_DataSet = np.loadtxt('{}/{}.txt'.format(InputFolder, FileName1), usecols=(0, 1, 2, 3, 4), skiprows=0)
    # print(f"{Fore.GREEN}{Style.BRIGHT}{FileName} Dosyasından veriler okundu...{Style.RESET_ALL}")

    #FileName2 = 'Routes_DataSet'  # RouteNo, PointID, BearingToNext, BearingToPrev
    Routes_DataSet = np.loadtxt('{}/{}.txt'.format(InputFolder, FileName2), usecols=(0, 1, 2, 3), skiprows=0)
    print(f"{Fore.GREEN}{Style.BRIGHT}{FileName1} ve {FileName2} dosyalarından veriler okundu...{Style.RESET_ALL}")
    return Points_DataSet, Routes_DataSet

def PrintDataInformation(Routes_DataSet, Points_DataSet):
    route_numbers, route_counts = np.unique(Routes_DataSet[:, 0], return_counts=True)
    # En çok nokta içeren rotanın numarasını ve içerdiği nokta sayısını bul
    max_points_route_no = int(route_numbers[np.argmax(route_counts)])
    max_points_count = np.max(route_counts)
    print(f"{Fore.YELLOW}{Style.BRIGHT}Rota Sayısı: {len(route_numbers)}, Toplam Nokta sayısı: {len(Points_DataSet)}{Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}En çok noktayı içeren rotaNo: {max_points_route_no}, İçerdiği Nokta Sayısı: {max_points_count}{Style.RESET_ALL}")

def UpdateDatasetTables(Routes_DataSet, Points_DataSet, distance_threshold, angle_difference_threshold):
    # Routes_DataSet'ten unique PointID'leri çıkarın
    unique_point_ids = np.unique(Routes_DataSet[:, 1])

    # Points_DataSet'i filtreleyin
    filtered_points = np.array([point for point in Points_DataSet if point[0] in unique_point_ids])
    print('işlem öncesi Points_DataSet Nokta Sayısı:', len(Points_DataSet))
    Points_DataSet = filtered_points
    print('işlem sonrası Points_DataSet Nokta Sayısı:', len(Points_DataSet))

    # Eski PointID'leri yeni PointID'lerle eşleyen bir sözlük oluşturun
    pointID_mapping = {}
    for i in range(len(Points_DataSet)):
        old_pointID = Points_DataSet[i, 0]
        Points_DataSet[i, 0] = i
        pointID_mapping[old_pointID] = i

    # Routes_DataSet'teki PointID'leri yeni PointID'lerle güncelleyin
    for i in range(len(Routes_DataSet)):
        old_pointID = Routes_DataSet[i, 1]
        new_pointID = pointID_mapping[old_pointID]
        Routes_DataSet[i, 1] = new_pointID

    np.savetxt('{}/Points_DataSet_Dist{}_Bearing{}.txt'.format(OutputFolder, distance_threshold, angle_difference_threshold), Points_DataSet)
    np.savetxt('{}/Routes_DataSet_Dist{}_Bearing{}.txt'.format(OutputFolder, distance_threshold, angle_difference_threshold), Routes_DataSet)
    print('Yeni Dosyalar Oluşturuldu')
    return Routes_DataSet, Points_DataSet

def get_Distance(lat1, lon1, lat2, lon2):
    p1 = [lat1, lon1]
    p2 = [lat2, lon2]
    return distance.distance(p1, p2).m

def get_bearing(lat1, long1, lat2, long2):
    brng = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
    return brng

def getPoint2(lat1, lon1, azi1, s12):
    g = Geodesic.WGS84.Direct(lat1, lon1, azi1, s12)
    lat2 = g['lat2']
    long2 = g['lon2']
    p = Point(lat2,long2)
    return p

def get_start_points(routes_data):
    start_points = []

    # Routes_DataSet'i rota numaralarına göre gruplayarak her rotanın başlangıç noktasını bulma
    for route_no in np.unique(routes_data[:, 0]):
        route_points = routes_data[routes_data[:, 0] == route_no]

        # Başlangıç noktasını belirleme
        start_point_index = 0  # Varsayılan olarak ilk nokta
        start_point = route_points[start_point_index]
        start_points.append(start_point)
    return np.array(start_points)

def Sort_outermost_points(start_points):
    # Merkez noktasını belirleme (Örnek olarak orta noktayı kullanıyoruz)
    center_lat = np.mean(Points_DataSet[:, 1])
    center_lon = np.mean(Points_DataSet[:, 2])
    sorted_start_points = sorted(start_points, key=lambda x: get_Distance(center_lat, center_lon, Points_DataSet[int(x[1]), 1], Points_DataSet[int(x[1]), 2]), reverse=True)
    return sorted_start_points

def Sort_innermost_points(start_points):
    # Merkez noktasını belirleme (Örnek olarak orta noktayı kullanıyoruz)
    center_lat = np.mean(Points_DataSet[:, 1])
    center_lon = np.mean(Points_DataSet[:, 2])
    sorted_start_points = sorted(start_points, key=lambda x: get_Distance(center_lat, center_lon, Points_DataSet[int(x[1]), 1], Points_DataSet[int(x[1]), 2]), reverse=False)
    return sorted_start_points

# Function to run the merging process with given thresholds and collect data
def run_Process_with_thresholds(distance_threshold, angle_difference_threshold, degree_to_meter):
    global Points_DataSet, Routes_DataSet, Temp_Routes_DataSet

    # Load the initial datasets
    Points_DataSet, Routes_DataSet = ReadDataSets(FileName1='Points_DataSet', FileName2='Routes_DataSet_Dist20_Bearing12')
    Temp_Routes_DataSet = Routes_DataSet.copy()

    # Perform the merging process
    New_Points_DataSet, Routes_DataSet = Process(Routes_DataSet, Points_DataSet, Temp_Routes_DataSet, distance_threshold, angle_difference_threshold, degree_to_meter)

    print('Routes_DataSet güncelleniyor')

    # Bir başka rota tarafından içerilen rotalar silinir
    Routes_DataSet = remove_contained_routes(Routes_DataSet)

    # Save the new datasets (optional)
    np.savetxt('{}/New_Points_DataSet_Dist{}_Bearing{}.txt'.format(OutputFolder, distance_threshold, angle_difference_threshold), New_Points_DataSet)
    np.savetxt('{}/Routes_DataSet_Dist{}_Bearing{}.txt'.format(OutputFolder, distance_threshold, angle_difference_threshold), Routes_DataSet)

    # Collect the results
    remaining_points = len(New_Points_DataSet)
    remaining_routes = len(np.unique(Routes_DataSet[:, 0]))

    print('{}/New_Points_DataSet_Dist{}_Bearing{}.txt kaydedildi, Nokta Sayısı:{}'.format(OutputFolder, distance_threshold, angle_difference_threshold, remaining_points))
    print('{}/Routes_DataSet_Dist{}_Bearing{}.txt kaydedildi, Rota Sayısı:{}'.format(OutputFolder, distance_threshold, angle_difference_threshold, remaining_routes))

def Process(Routes_DataSet, Points_DataSet, Temp_Routes_DataSet, distance_threshold, angle_difference_threshold, degree_to_meter):
    New_Points_DataSet = []
    new_point_id = len(Points_DataSet)
    #print(new_point_id)

    pbar = tqdm(total=len(Temp_Routes_DataSet), desc=f"Noktalar Birleştiriliyor. Dist:{distance_threshold}, Angle:{angle_difference_threshold}", unit='line')
    while 0 < len(Temp_Routes_DataSet):
        current_group = SelectPointAndGetNeigbours_Optimized(Points_DataSet, Temp_Routes_DataSet, distance_threshold, angle_difference_threshold, degree_to_meter)
        New_Points_DataSet, Routes_DataSet, new_point_id = MergeGrupedPointsAndUpdateTables(current_group, Points_DataSet, New_Points_DataSet, Routes_DataSet, new_point_id)
        Temp_Routes_DataSet, Points_DataSet = Update_Temp_Routes_DataSet_and_Points_DataSet(current_group, Temp_Routes_DataSet, Points_DataSet)
        pbar.update(len(current_group))

    return New_Points_DataSet, Routes_DataSet

def SelectPointAndGetNeigbours_Optimized(Points_DataSet, Temp_Routes_DataSet, distance_threshold, angle_difference_threshold, degree_to_meter):
    # Başlangıç noktaları belirlenir
    start_points = get_start_points(Temp_Routes_DataSet)

    #Başlangıç Noktaları dıştan içe sıralanır
    sorted_start_points = Sort_innermost_points(start_points)

    current_point = sorted_start_points.pop(0)  # Başlangıç noktasını al

    current_point_id = int(current_point[1])
    mask = np.isin(Points_DataSet[:, 0], current_point_id)
    current_point_indices = np.where(mask)[0][0]

    # Mesafe kriterine göre seçili noktaya yakın noktalar gruplanır
    nearby_points_IDs_InPointDataSet = find_points_within_distance(current_point_indices, Points_DataSet, distance_threshold, degree_to_meter)
    current_group = [current_point_id]  # Başlangıç noktasını içeren grup

    # Açı kriterine göre seçili noktaya yakın noktalar gruplanır
    if len(nearby_points_IDs_InPointDataSet) > 0:
        grouped_nearby_points_withBearing_InPointDataSet = group_nearby_points(current_point_id, nearby_points_IDs_InPointDataSet, Points_DataSet,
                                                                           Temp_Routes_DataSet, angle_difference_threshold)
        current_group.extend(grouped_nearby_points_withBearing_InPointDataSet)

    # PointID ler PointIndices leri bulunur
    mask = np.isin(Points_DataSet[:, 0], current_group)
    current_group_indices = np.where(mask)[0]
    return current_group_indices

def find_points_within_distance(current_point_id, Points_DataSet, distance_threshold, degree_to_meter):
    # current_point'ın koordinatlarını al
    current_point_coords = Points_DataSet[current_point_id][1:3]  # Lat ve lon değerleri

    # KDTree oluştur
    tree = cKDTree(Points_DataSet[:, 1:3])  # Sadece Lat ve lon sütunları

    # Distance_meter'i dereceye dönüştürün
    distance_degree = distance_threshold / degree_to_meter

    # distance_threshold içindeki tüm noktaları bul
    nearby_points_indices = tree.query_ball_point(current_point_coords, distance_degree)

    # current_point'ı içeren indisleri kaldır
    nearby_points_indices.remove(current_point_id)

    nearby_points_IDs = Points_DataSet[nearby_points_indices, 0]
    return nearby_points_IDs

def group_nearby_points(current_point_id, nearby_points_IDs, Points_DataSet, Temp_Routes_DataSet, angle_difference_threshold):
    # Başlangıç noktasının bearing değerlerini al
    current_point_indices = np.where(Temp_Routes_DataSet[:, 1] == current_point_id)[0]
    current_point_bearings = Temp_Routes_DataSet[current_point_indices[0], 2:4]

    mask = (Temp_Routes_DataSet[:, 1] == current_point_id)
    #current_point_indices = np.where(mask)[0]
    current_point_bearings = Temp_Routes_DataSet[mask, 2:4][0]

    # Gruplanmış noktaları saklamak için boş bir liste oluştur
    grouped_nearby_points = []

    # Başlangıç noktası ile distance_threshold içinde olan tüm noktaları dolaş
    for point_id in nearby_points_IDs:
        # Noktanın Temp_Routes_DataSet içindeki indekslerini bul
        mask = (Temp_Routes_DataSet[:, 1] == point_id)
        #point_indices = np.where(mask)[0]
        point_bearings = Temp_Routes_DataSet[mask, 2:4][0]

        # Başlangıç noktasının bearing değeri ile farklarını hesapla
        bearing_diff_next = abs(current_point_bearings[0] - point_bearings[0])
        bearing_diff_prev = abs(current_point_bearings[1] - point_bearings[1])

        # Açı farklarını angle_difference_threshold ile karşılaştır ve koşulu sağlayanları grupla
        if np.any(bearing_diff_next < angle_difference_threshold) or np.any(bearing_diff_prev < angle_difference_threshold):
            #print(current_point_bearings, point_bearings, bearing_diff_next, bearing_diff_prev, np.any(bearing_diff_next < angle_difference_threshold), np.any(bearing_diff_prev < angle_difference_threshold))
            grouped_nearby_points.append(point_id)

    return grouped_nearby_points

def MergeGrupedPointsAndUpdateTables(current_group, Points_DataSet, New_Points_DataSet, Routes_DataSet, new_point_id):
    if len(current_group) > 1:
        # Birden fazla nokta birleştirilir ve yeni nokta oluşturulur
        total_lat = np.sum(Points_DataSet[current_group, 1] * Points_DataSet[current_group, 3])
        total_lon = np.sum(Points_DataSet[current_group, 2] * Points_DataSet[current_group, 3])
        total_weight = np.sum(Points_DataSet[current_group, 3])

        avg_lat = total_lat / total_weight
        avg_lon = total_lon / total_weight

        # Yeni noktayı Points_DataSet'e ekleyin
        new_point_id += 1
        new_point = [new_point_id, avg_lat, avg_lon, total_weight, 1]
        New_Points_DataSet.append(new_point)

    else:
        # Tekil Noktalar eğer istenirse ihmal edilebilir. Birleşim sayısı 0 olarak kaydedilir.
        point_id = int(current_group[0])
        avg_lat, avg_lon, total_weight = Points_DataSet[point_id, 1:4]
        new_point_id += 1

        # Tekil(birleşmemiş) Noktalar istenirse ihmal edilebilir. Birleşim sayısı 0 olarak kaydedilir.
        new_point = [new_point_id, avg_lat, avg_lon, total_weight, 0]
        New_Points_DataSet.append(new_point)

    # Routes_DataSet'i güncelle
    Routes_DataSet[np.isin(Routes_DataSet[:, 1], current_group), 1] = new_point_id
    return New_Points_DataSet, Routes_DataSet, new_point_id

def Update_Temp_Routes_DataSet_and_Points_DataSet(current_group, Temp_Routes_DataSet, Points_DataSet):
    points_IDs = Points_DataSet[current_group, 0]
    # İlgili satırları filtreleme kullanarak bul ve kaldır
    #print(len(Temp_Routes_DataSet), len(Points_DataSet), len(current_group), len(Temp_Routes_DataSet)-len(current_group))
    mask = np.isin(Temp_Routes_DataSet[:, 1], points_IDs, invert=True)
    Temp_Routes_DataSet = Temp_Routes_DataSet[mask]

    # Points_DataSet'ten ilgili satırları kaldırmak için boolean maske kullanın
    mask_points = np.ones(len(Points_DataSet), dtype=bool)
    mask_points[current_group] = False
    Points_DataSet = Points_DataSet[mask_points]
    #print(len(Temp_Routes_DataSet), len(Points_DataSet))

    return Temp_Routes_DataSet, Points_DataSet

def calculate_degree_to_meter(latitude, longitude):
    # Dünya yarıçapını (metre cinsinden) belirleyin
    R = 6371000  # Dünya yarıçapı yaklaşık olarak 6371 km

    # Lat ve long değerlerini derece cinsinden alın
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)

    # Yatayda (longitude) bir derecenin metre cinsinden uzunluğunu hesaplayın
    degree_to_meter_lon = R * np.cos(lat_rad) * np.pi / 180

    # Dikeyde (latitude) bir derecenin metre cinsinden uzunluğunu hesaplayın
    degree_to_meter_lat = R * np.pi / 180

    # Ortalama bir değer alarak bu iki değeri birleştirin
    degree_to_meter = (degree_to_meter_lon + degree_to_meter_lat) / 2

    return degree_to_meter

def get_center_lat_long(Points_DataSet):
    center_lat = np.mean(Points_DataSet[:, 1])
    center_lon = np.mean(Points_DataSet[:, 2])
    return center_lat, center_lon

def remove_contained_routes(Routes_DataSet):
    unique_routes = np.unique(Routes_DataSet[:, 0])
    route_points_dict = {}

    # Her rota için nokta kümesi oluştur
    for route_no in unique_routes:
        route_points = set(Routes_DataSet[Routes_DataSet[:, 0] == route_no][:, 1])
        route_points_dict[route_no] = route_points

    # Rotaları en çok nokta içerenden en az nokta içerene doğru sırala
    sorted_routes = sorted(route_points_dict.items(), key=lambda x: len(x[1]), reverse=False)

    to_remove = set()

    # Her bir rota için diğer rotalarla karşılaştırma yap
    for i, (route1, route1_points) in enumerate(tqdm(sorted_routes, desc="Processing Routes")):
        if route1 in to_remove:
            continue

        for route2, route2_points in sorted_routes[i + 1:]:
            if route2 in to_remove:
                continue

            if route1_points.issubset(route2_points):
                to_remove.add(route1)
                break
            elif route2_points.issubset(route1_points):
                to_remove.add(route2)

    mask = np.isin(Routes_DataSet[:, 0], list(to_remove), invert=True)
    Routes_DataSet = Routes_DataSet[mask]
    return Routes_DataSet

def create_result_file(distance_thresholds, angle_difference_thresholds, OutputFolder):
    results = []
    output_file = 'result.txt'

    for dist in distance_thresholds:
        for angle in angle_difference_thresholds:
            try:
                # Dosya adlarını oluşturma
                new_points_file = f"{OutputFolder}/New_Points_DataSet_Dist{dist}_Bearing{angle}.txt"
                routes_file = f"{OutputFolder}/Routes_DataSet_Dist{dist}_Bearing{angle}.txt"

                # Dosyaları okuma
                new_points_data = np.loadtxt(new_points_file)
                routes_data = np.loadtxt(routes_file)

                # Gerekli bilgileri elde etme
                num_points = new_points_data.shape[0]
                num_routes = len(np.unique(routes_data[:, 0]))

                # Sonuçları kaydetme
                results.append((dist, angle, num_points, num_routes))
            except Exception as e:
                print(f"Error processing dist {dist}, angle {angle}: {e}")

    # Sonuçları bir dosyaya kaydetme
    with open(output_file, 'w') as f:
        f.write("Distance_Threshold\tAngle_Difference_Threshold\tNum_Points\tNum_Routes\n")
        for result in results:
            f.write(f"{result[0]}\t{result[1]}\t{result[2]}\t{result[3]}\n")

    print(f"Results saved to {output_file}")
    return results

# Draw Graphics
def Graphic_1(OutputFolder):
    # Plot and save the scatter plot for remaining points and remaining_routes
    plt.figure(figsize=(6, 6))
    scatter1 = plt.scatter(results_array[:, 0], results_array[:, 1], c=results_array[:, 2], cmap='viridis')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Angle Difference Threshold')
    plt.title('Remaining Points')
    plt.colorbar(scatter1, label='Remaining Points')
    plt.savefig(f'{OutputFolder}/Graphics/remaining_points.png')
    plt.close()

    # Plot and save the scatter plot for remaining routes
    plt.figure(figsize=(6, 6))
    scatter2 = plt.scatter(results_array[:, 0], results_array[:, 1], c=results_array[:, 3], cmap='plasma')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Angle Difference Threshold')
    plt.title('Remaining Routes')
    plt.colorbar(scatter2, label='Remaining Routes')
    plt.savefig(f'{OutputFolder}/Graphics/remaining_routes.png')
    plt.close()
    print("remaining_points.png and remaining_routes.png saved")
    return

def separate_Graphics(OutputFolder, angle_difference_thresholds, distance_thresholds):
    # .............................. Açı sabitken mesafe artan grafikler AYRI AYRI gösterim
    num_angles = len(angle_difference_thresholds)
    fig, axs = plt.subplots(num_angles, 2, figsize=(12, 6 * num_angles))

    for i, angle in enumerate(angle_difference_thresholds):
        angle_results = results_array[results_array[:, 1] == angle]
        sorted_indices = np.argsort(angle_results[:, 0])
        angle_results = angle_results[sorted_indices]

        axs[i, 0].plot(angle_results[:, 0], angle_results[:, 2], marker='o', linestyle='-')
        axs[i, 0].set_title(f'Points for Angle {angle}')
        axs[i, 0].set_xlabel('Distance Threshold')
        axs[i, 0].set_ylabel('Remaining Points')

        axs[i, 1].plot(angle_results[:, 0], angle_results[:, 3], marker='o', linestyle='-')
        axs[i, 1].set_title(f'Routes for Angle {angle}')
        axs[i, 1].set_xlabel('Distance Threshold')
        axs[i, 1].set_ylabel('Remaining Routes')

    plt.tight_layout()
    plt.savefig(f'{OutputFolder}/Graphics/S_angle_fixed_distance_varying.png')
    plt.close()

    # Mesafe sabitken açı artan grafikler
    num_distances = len(distance_thresholds)
    fig, axs = plt.subplots(num_distances, 2, figsize=(12, 6 * num_distances))

    for i, dist in enumerate(distance_thresholds):
        dist_results = results_array[results_array[:, 0] == dist]
        sorted_indices = np.argsort(dist_results[:, 1])
        dist_results = dist_results[sorted_indices]

        axs[i, 0].plot(dist_results[:, 1], dist_results[:, 2], marker='o', linestyle='-')
        axs[i, 0].set_title(f'Points for Distance {dist}')
        axs[i, 0].set_xlabel('Angle Difference Threshold')
        axs[i, 0].set_ylabel('Remaining Points')

        axs[i, 1].plot(dist_results[:, 1], dist_results[:, 3], marker='o', linestyle='-')
        axs[i, 1].set_title(f'Routes for Distance {dist}')
        axs[i, 1].set_xlabel('Angle Difference Threshold')
        axs[i, 1].set_ylabel('Remaining Routes')

    plt.tight_layout()
    plt.savefig(f'{OutputFolder}/Graphics/S_distance_fixed_angle_varying.png')
    plt.close()
    print("angle_fixed_distance_varying.png and distance_fixed_angle_varying.png saved")
    return

def Combinated_Graphics(OutputFolder, angle_difference_thresholds, distance_thresholds):
    plt.figure(figsize=(12, 6))
    # .............................. Açı sabitken mesafe artan grafikler BİRARADA gösterim
    for angle in angle_difference_thresholds:
        angle_results = results_array[results_array[:, 1] == angle]
        sorted_indices = np.argsort(angle_results[:, 0])
        angle_results = angle_results[sorted_indices]

        plt.plot(angle_results[:, 0], angle_results[:, 2], marker='o', linestyle='-', label=f'Angle {angle}')
    plt.title('Remaining Points for Different Angles (Distance Varying)')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Remaining Points')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OutputFolder}/Graphics/combined_points_angle_fixed_distance_varying.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    for angle in angle_difference_thresholds:
        angle_results = results_array[results_array[:, 1] == angle]
        sorted_indices = np.argsort(angle_results[:, 0])
        angle_results = angle_results[sorted_indices]

        plt.plot(angle_results[:, 0], angle_results[:, 3], marker='o', linestyle='-', label=f'Angle {angle}')
    plt.title('Remaining Routes for Different Angles (Distance Varying)')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Remaining Routes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OutputFolder}/Graphics/combined_routes_angle_fixed_distance_varying.png')
    plt.close()

    # Mesafe sabitken açı artan grafikler
    plt.figure(figsize=(12, 6))
    for dist in distance_thresholds:
        dist_results = results_array[results_array[:, 0] == dist]
        sorted_indices = np.argsort(dist_results[:, 1])
        dist_results = dist_results[sorted_indices]

        plt.plot(dist_results[:, 1], dist_results[:, 2], marker='o', linestyle='-', label=f'Distance {dist}')
    plt.title('Remaining Points for Different Distances (Angle Varying)')
    plt.xlabel('Angle Difference Threshold')
    plt.ylabel('Remaining Points')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OutputFolder}/Graphics/combined_points_distance_fixed_angle_varying.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    for dist in distance_thresholds:
        dist_results = results_array[results_array[:, 0] == dist]
        sorted_indices = np.argsort(dist_results[:, 1])
        dist_results = dist_results[sorted_indices]

        plt.plot(dist_results[:, 1], dist_results[:, 3], marker='o', linestyle='-', label=f'Distance {dist}')
    plt.title('Remaining Routes for Different Distances (Angle Varying)')
    plt.xlabel('Angle Difference Threshold')
    plt.ylabel('Remaining Routes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OutputFolder}/Graphics/combined_routes_distance_fixed_angle_varying.png')
    plt.close()

    print("combined_routes and combined_points graphics saved")
    return

def Combinated_GraphicsSoftColored(OutputFolder, angle_difference_thresholds, distance_thresholds):
    # .............................. Açı sabitken mesafe artan grafikler RENK GEÇİŞLİ gösterim
    # Renk haritaları
    color_map_points = plt.get_cmap('viridis')
    color_map_routes = plt.get_cmap('plasma')

    # Açı sabitken mesafe artan grafikler
    plt.figure(figsize=(12, 6))
    for idx, angle in enumerate(angle_difference_thresholds):
        angle_results = results_array[results_array[:, 1] == angle]
        sorted_indices = np.argsort(angle_results[:, 0])
        angle_results = angle_results[sorted_indices]

        plt.plot(angle_results[:, 0], angle_results[:, 2], marker='o', linestyle='-', color=color_map_points(idx / len(angle_difference_thresholds)),
                 label=f'Angle {angle}')
    plt.title('Remaining Points for Different Angles (Distance Varying)')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Remaining Points')
    plt.legend()
    plt.text(0.45, 0.96, f'Total points before process: {Total_points_Before_Process}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')
    plt.tight_layout()
    plt.savefig(f'{OutputFolder}/Graphics/R_combined_points_angle_fixed_distance_varying.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    for idx, angle in enumerate(angle_difference_thresholds):
        angle_results = results_array[results_array[:, 1] == angle]
        sorted_indices = np.argsort(angle_results[:, 0])
        angle_results = angle_results[sorted_indices]

        plt.plot(angle_results[:, 0], angle_results[:, 3], marker='o', linestyle='-', color=color_map_routes(idx / len(angle_difference_thresholds)),
                 label=f'Angle {angle}')
    plt.title('Remaining Routes for Different Angles (Distance Varying)')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Remaining Routes')
    plt.legend()
    plt.text(0.60, 0.96, f'Total routes before process: {Total_routes_Before_Process}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')
    plt.tight_layout()
    plt.savefig(f'{OutputFolder}/Graphics/R_combined_routes_angle_fixed_distance_varying.png')
    plt.close()

    # Mesafe sabitken açı artan grafikler
    plt.figure(figsize=(12, 6))
    for idx, dist in enumerate(distance_thresholds):
        dist_results = results_array[results_array[:, 0] == dist]
        sorted_indices = np.argsort(dist_results[:, 1])
        dist_results = dist_results[sorted_indices]

        plt.plot(dist_results[:, 1], dist_results[:, 2], marker='o', linestyle='-', color=color_map_points(idx / len(distance_thresholds)),
                 label=f'Distance {dist}')
    plt.title('Remaining Points for Different Distances (Angle Varying)')
    plt.xlabel('Angle Difference Threshold')
    plt.ylabel('Remaining Points')
    plt.legend()
    plt.text(0.45, 0.96, f'Total points before process: {Total_points_Before_Process}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')
    plt.tight_layout()
    plt.savefig(f'{OutputFolder}/Graphics/R_combined_points_distance_fixed_angle_varying.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    for idx, dist in enumerate(distance_thresholds):
        dist_results = results_array[results_array[:, 0] == dist]
        sorted_indices = np.argsort(dist_results[:, 1])
        dist_results = dist_results[sorted_indices]

        plt.plot(dist_results[:, 1], dist_results[:, 3], marker='o', linestyle='-', color=color_map_routes(idx / len(distance_thresholds)),
                 label=f'Distance {dist}')
    plt.title('Remaining Routes for Different Distances (Angle Varying)')
    plt.xlabel('Angle Difference Threshold')
    plt.ylabel('Remaining Routes')
    plt.legend()
    plt.text(0.40, 0.96, f'Total routes before process: {Total_routes_Before_Process}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')
    plt.tight_layout()
    plt.savefig(f'{OutputFolder}/Graphics/R_combined_routes_distance_fixed_angle_varying.png')
    plt.close()

    print("R_combined_routes and R_combined_points graphics saved")
    return

# Plot Maps
def plot_AllRoutes_Folium(Routes_DataSet, Points_DataSet, OutputFolder, distance_threshold, angle_difference_threshold, title_text="Routes"):
    # Create a map centered at the average latitude and longitude
    avg_lat = np.mean(Points_DataSet[:, 1])
    avg_lon = np.mean(Points_DataSet[:, 2])
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

    # Define a colormap
    colors = plt.get_cmap("tab20", len(np.unique(Routes_DataSet[:, 0])))

    for idx, route_no in enumerate(np.unique(Routes_DataSet[:, 0])):
        route_points = Routes_DataSet[Routes_DataSet[:, 0] == route_no]
        point_ids = route_points[:, 1].astype(int)

        latitudes = Points_DataSet[Points_DataSet[:, 0].astype(int) == point_ids[0], 1]
        longitudes = Points_DataSet[Points_DataSet[:, 0].astype(int) == point_ids[0], 2]

        plt.scatter(longitudes, latitudes, color='Blue', marker='^', s=5, label=f"Route {int(route_no)} Start" if idx == 0 else "")

        # Initialize lists for route latitudes and longitudes
        route_latitudes = []
        route_longitudes = []

        for point_id in point_ids:
            point = Points_DataSet[Points_DataSet[:, 0].astype(int) == point_id]
            lat, lon, weight = point[0, 1:4]

            route_latitudes.append(lat)
            route_longitudes.append(lon)

            # Add markers for points
            if weight == 1:
                plt.scatter(lon, lat, color='red', marker='s', s=2)
            else:
                plt.scatter(lon, lat, color=colors(idx), marker='o', s=3)
        if route_no % 100 == 0:
            print(route_no)
            # Add the route to the map
            folium.PolyLine(locations=list(zip(route_latitudes, route_longitudes)), color=colors(idx), weight=3, opacity=0.8, popup=f"Route {route_no}").add_to(m)
            # Save the map to an HTML file
            output_file = f'{OutputFolder}/{title_text}_Dist{distance_threshold}_Bearing{angle_difference_threshold}.html'
            m.save(output_file)

    # Save the map to an HTML file
    output_file = f'{OutputFolder}/{title_text}_Dist{distance_threshold}_Bearing{angle_difference_threshold}.html'
    m.save(output_file)
    print(f"Map saved to {OutputFolder}")



# --------------------------------------------------------------------
InputFolder, OutputFolder = DefineFolders()

# ----------------------------------------------- Stage 1: Paralel process to create Files Points_DataSet ve Routes_DataSet with trashold values
if __name__ == '__main__':
    #Program Başlangıç saati
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    # Mesafe ve açı farkı eşik değerleri
    #distance_threshold = 20  # metre
    #angle_difference_threshold = 12  # derece

    # Define ranges for distance and angle thresholds
    distance_thresholds = [10, 20, 30, 40]
    angle_difference_thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]

    # Store results
    results = []
    Points_DataSet, Routes_DataSet = ReadDataSets(FileName1='Points_DataSet', FileName2='Routes_DataSet_Dist20_Bearing12')
    PrintDataInformation(Routes_DataSet, Points_DataSet)
    Temp_Routes_DataSet = Routes_DataSet

    # Total Point and Route Count Before Process
    Total_points_Before_Process = len(Points_DataSet)
    Total_routes_Before_Process = len(np.unique(Routes_DataSet[:, 0]))

    center_lat, center_lon = get_center_lat_long(Points_DataSet)
    degree_to_meter = calculate_degree_to_meter(center_lat, center_lon)

    # Paralel işlem için ProcessPoolExecutor kullanın
    with ProcessPoolExecutor() as executor:
        # Geleceklerin listesini oluşturun
        futures = [executor.submit(run_Process_with_thresholds, dist, angle, degree_to_meter)
                   for angle, dist in itertools.product(angle_difference_thresholds, distance_thresholds)]



    # ----------------------------------------------- Stage 2: create file containing results and draw graphics
    results = create_result_file(distance_thresholds, angle_difference_thresholds, OutputFolder)
    print(results)
    results_array = np.array(results)

    # ----------------------------------------------- Draw Graphics

    Graphic_1(OutputFolder)
    separate_Graphics(OutputFolder, angle_difference_thresholds, distance_thresholds)
    Combinated_Graphics(OutputFolder, angle_difference_thresholds, distance_thresholds)
    Combinated_GraphicsSoftColored(OutputFolder, angle_difference_thresholds, distance_thresholds)




    # .............................. SHOW Datasets on Maps

    # İlk deneme için bir eşik değeri ve açı farkı seçimi
    dist_threshold = distance_thresholds[3]
    angle_diff_threshold = angle_difference_thresholds[10]

    # Folium ile rotaları haritada görselleştirme
    plot_AllRoutes_Folium(Routes_DataSet, Points_DataSet, OutputFolder, dist_threshold, angle_diff_threshold)