"""
Creates the pytorch geometric dataset that is used in our publication.
To reuse the dataset, use the UBAGraph.get_dataset() routine.
"""

# general
import random
import pdb
import warnings
import pickle as pkl
import datetime

# data science
import numpy as np
import pandas as pd
import dask.dataframe as ddf

# pytorch
import torch
from torch_geometric.data import Data, InMemoryDataset

# sklearn
from sklearn.neighbors import NearestNeighbors

# own package
import settings
from retrieval_toar_db import StationData
from retrieval_toar_db import HourlyData
from preprocessing_utils import geo_to_cartesian, neighbor_index_filter, \
                                loc_weight, time_weight, get_one_hot

from multiprocessing import Pool

# oppress future warnings
warnings.filterwarnings('ignore')

"""
- missing_o3_mask (True if o3 is missing)
- train_mask (True if training sample)
- val_mask (True if val sample)
- test_mask (True if test sample)
n.b. all masks should have the same statistical properties as
missing_o3_mask

Also adds the new gaps to gaps.csv.
n.b. some parts of the data split are pretty hand crafted,
e.g. the masks for correlated gaps.
"""

def func0():
    print('preparing masks...')

    # read in data
    gap_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_korea/uba_graph_preproc/gap.csv", index_col=0)
    reg_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_korea/uba_graph_preproc/reg.csv", index_col=0)
    reg_df.datetime = pd.to_datetime(reg_df.datetime)
    y_df_flat = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_korea/uba_graph_preproc/y.csv", index_col=0)
    hd = HourlyData('o3')
    hd.read_from_file()
    y_df = hd.df
    y_df.index = pd.to_datetime(y_df.index) + settings.time_offset

    # random seed
    random.seed(settings.random_seed)

    # initialize mask df
    columns = ['missing_o3_mask', 'train_mask', 'val_mask',
               'test_mask']
    mask_df = pd.DataFrame(columns=columns, index=y_df_flat.index,
                           data=False)
    mask_df.index.name = 'node_index' ###missing_o3, train, val, test_mask를 column으로 임의의 df 생성

    # reset gap data frame
    gap_df = gap_df[gap_df.type=='missing_o3_mask'].copy()

    # missing o3 is simply where we do not have any measurement
    mask_df.loc[y_df_flat.y!=y_df_flat.y, 'missing_o3_mask'] = True ### [!=] 결측치를 True로 설정

    # print summary statistics of the missing values
    n_total = len(mask_df)
    n_missing = mask_df.missing_o3_mask.sum()
    print('\nmissing measurements')
    print(f'{n_missing} of {n_total} = {n_missing/n_total*100:.1f} %')

    # station and time step info
    station_id_list = [int(id_) for id_ in y_df.columns]
    n_stations = len(station_id_list)
    n_stations_without_data = y_df.isnull().all(axis=0).sum()
    n_stations_with_data = n_stations - n_stations_without_data
    n_timesteps = len(y_df)

    # analyze correlated gaps
    lens, counts = np.unique(gap_df[gap_df.correlated].len,
                             return_counts=True)
    print(f'\ncorrelated gaps (simultaneously at 273 stations)')
    n_samples_corr_gaps = gap_df[gap_df.correlated].len.sum()
    for idx, len_ in enumerate(lens):
        n_gaps = int(counts[idx]/n_stations_with_data)
        n_samples = counts[idx] * len_
        print(f'length {len_}\t# gaps {n_gaps}\t# samples {n_samples}')
    print(f'\t\tsum: \t{n_samples_corr_gaps} samples = ' +
          f'{n_samples_corr_gaps/n_total*100:.1f} %')

    # state shopping list for correlatd gaps
    # tuple = (length, number of gaps of this length)
    # n.b. for each, test and val set!
    shopping_list_corr = [(3, 45), (24, 2)]
    print('\nshopping list each for val and test set')
    n_samples_required = 0
    for len_, n_gaps_required in shopping_list_corr:
        n_samples = len_ * n_stations * n_gaps_required
        print(f'length {len_}\t# gaps {n_gaps_required}\t# samples {n_samples}')
        n_samples_required += n_samples
    n_samples_required *= 2  # bc we have test and val sets
    print(f'\t\tsum: \t{n_samples_required} samples = ' +
          f'{n_samples_required/n_total*100:.1f} %') ###list [(3hours, 45 gaps), (24hours, 2gaps)] 생성

    # analyze single gaps
    lens, counts = np.unique(gap_df[~gap_df.correlated].len,
                             return_counts=True)
    print(f'\nsingle gaps')
    n_samples_gaps = gap_df[~gap_df.correlated].len.sum()
    for idx, len_ in enumerate(lens):
        n_gaps = counts[idx]
        n_samples = counts[idx] * len_
        print(f'length {len_}\t# gaps {n_gaps}\t# samples {n_samples}')
    print(f'\t\tsum: \t{n_samples_gaps} samples = ' +
          f'{n_samples_gaps/n_total*100:.1f} %')

    # shopping list for single gaps
    shopping_list = []
    n_samples_required = 0
    print('\nshopping list each for val and test set')
    for idx, len_ in enumerate(lens):
        n_gaps_required = round(counts[idx] / 2)
        if n_gaps_required < 1:
            continue
        # split year long gaps in half
        if len_ == 8760:
            len_ = round(len_ / 2)
            n_gaps_required *= 2
        n_samples = n_gaps_required * len_
        n_samples_required += n_samples
        print(f'length {len_}\t# gaps {n_gaps_required}\t# ' +
              f'samples {n_samples}')
        shopping_list.append((len_, n_gaps_required))
    n_samples_required *= 2  # bc we have test and val sets
    print(f'\t\tsum: \t{n_samples_required} samples = ' +
          f'{n_samples_required/n_total*100:.1f} %') ###list 생성 후 코드를 이용해 대입, 논문 Table 2 확장

    # now flag the gaps
    shopping_list_corr.reverse()
    shopping_list.reverse()

    dis = 0  # tmp

    print('\nflagging correlated gaps...')
    correlated = True
    for len_, n_gaps_required in shopping_list_corr:  # shopping_list_corr에서 결측치가 있어야 하는 구간의 길이와 개수를 가져와 반복
        for mask in ['val_mask', 'test_mask']:  # 검증, 테스트 데이터의 마스크를 순환
            n_gaps_found = 0  # 찾은 결측치 개수 초기화
            while n_gaps_found < n_gaps_required:  # 찾아야 할 결측치 개수 만큼 반복
                start_datetime = random.choice(y_df.index)  # y_df의 인덱스 중에서 무작위로 시작 일시 선택
                end_datetime = start_datetime + pd.to_timedelta(len_, unit='h')  # 시작 일시에서 길이를 더한 것이 끝 일시
                if end_datetime > y_df.index[-1]:
                    print(f'{end_datetime} out of bound')
                    continue

                # check if less than 25 % is already flagged by
                # reshaping the mask array
                mask_any_flat = mask_df.any(axis=1).to_numpy()
                mask_any = mask_any_flat.reshape((n_stations,
                                                  n_timesteps)).T
                mask_any_df = pd.DataFrame(index=y_df.index,
                                           columns=y_df.columns,
                                           data=mask_any)
                candidates = mask_any_df.loc[start_datetime:end_datetime]
                n_nan = candidates.sum().sum()
                if n_nan/candidates.size > .25: ###논문 p.3에서 결측값의 최대 비용이 25%이하인 시계열에만 적용
                    print(f'{n_nan} of {candidates.size} missing') ###결측치 비율이 25%를 초과하면 과정을 생략
                    continue

                # flag as gap and append info to gap df
                for station_id in station_id_list:
                    type_ = mask
                    idx_filter = [(reg_df.datetime==start_datetime) &
                                  (reg_df.station_id==station_id)][0]  ###reg_df에서 시작 일시와 station_id가 같은 행을 필터링
                    start_index = reg_df.loc[idx_filter].index[0]  ###해당 행의 인덱스를 시작 인덱스로 설정
                    end_index = start_index + len_ -1  ###시작 인덱스에 길이를 더한 것이 끝 인덱스
                    filter_ = [(mask_df.index>=start_index) &
                               (mask_df.index<=end_index) &
                               (~mask_df.any(axis=1))][0]  ###mask_df에서 시작 인덱스와 끝 인덱스 사이에 해당 열에서 결측치가 없는 행을 필터링
                    n_samples = filter_.sum()  ###필터링된 행의 개수가 결측치로 채울 데이터 개수
                    mask_df.loc[filter_, mask] = True  ###해당하는 행의 mask 열의 값을 True로 설정하여 결측치로 채움
                    _ = [station_id, type_, correlated,  ###필요한 정보를 저장할 리스트
                         start_datetime, end_datetime, start_index,
                         len_, n_samples]
                    gap_df.loc[len(gap_df), :] = _  ###리스트를 gap_df에 추가
                n_gaps_found += 1  ###찾은 결측치 개수 증가
                print(f'found correlated {mask} gap of len {len_}' +
                      f' ({n_gaps_required-n_gaps_found} to go)') ###correlated는 전국의 모든 관측소에서 결측치가 나타나서 y_df를 사용하여 과정을 축소함

    print('\nflagging single gaps...')
    correlated = False
    for len_, n_gaps_required in shopping_list:
        for mask in ['val_mask', 'test_mask']:
            n_gaps_found = 0
            while n_gaps_found < n_gaps_required:
                # pick random start index of the gap
                start_idx = random.choice(mask_df.index) ###random choice에서 유효하지 않은 값 선별
                start_station = reg_df.at[start_idx, 'station_id']
                # add gap len to obtain end index
                end_idx = start_idx + len_ - 1
                # useless if it exceeds the original index
                if end_idx > mask_df.index[-1]:
                    print('exceeds df boundaries') ###mask_df에서 추출
                    continue
                end_station = reg_df.at[end_idx, 'station_id']
                if start_station != end_station:
                    print('different station')
                    continue
                # useless if any more than 10 % of the values True
                mask_df_filtered = mask_df.loc[start_idx:end_idx]
                missing_values = mask_df_filtered.sum().sum()
                if missing_values/len_ > .25 : ###논문 p.3에서 결측값의 최대 비용이 25%이하인 시계열에만 적용 ; "In order to ensure a certain level of robustness of the metric,"
                    print('too many missing values') ###결측치 비율이 25%를 초과하면 과정을 생략
                    continue
                # set those to true that are not true elsewhere
                index_list = mask_df_filtered[
                    (mask_df_filtered==False).all(axis=1)].index
                mask_df.loc[index_list, mask] = True
                n_samples = len(index_list)
                start_datetime = reg_df.at[start_idx, 'datetime']
                end_datetime = reg_df.at[end_idx, 'datetime']
                _ = [start_station, mask, correlated,
                     start_datetime, end_datetime, start_idx,
                     len_, n_samples]
                gap_df.loc[len(gap_df), :] = _
                n_gaps_found += 1 ###결측치가 있는 데이터의 다른 정보를 기록
                print(f'found single {mask} gap of len {len_}' +
                      f' ({n_gaps_required-n_gaps_found} to go)') ###reg_df에서 datetime을 위주로 랜덤으로 validation과 test를 선정함

    # train mask is where no other mask applies
    index_list = mask_df[(mask_df==False).all(axis=1)].index
    mask_df.loc[index_list, 'train_mask'] = True ###만약 다른 항목들이 모두 Flase라면 train_mask를 True로 입력

    # convert gap df columns back to int
    gap_df = gap_df.astype({'station_id':int, 'start_idx':int,
                            'len':int, 'n_samples':int})

    # print a summary of all masks
    print('\nmask summary')
    n_samples = len(mask_df)
    print(f'{n_samples} in total (100.0 %)')
    for col in mask_df.columns:
        n_samples_set = mask_df[col].sum().sum()
        percentage = n_samples_set / n_samples * 100
        print(f'{n_samples_set} in {col} ({percentage:.2f} %)')

    # mask sanity check
    assert mask_df.sum().sum() == len(mask_df)

    # save
    mask_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_korea/uba_graph_preproc/mask.csv")

    gap_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_korea/uba_graph_preproc/gap.csv")

iterable = ["p1"]

if __name__ == "__main__":
    pool = Pool(processes=12)
    pool.map(func0,iterable)
