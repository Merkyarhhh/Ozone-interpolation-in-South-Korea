"""
This file contains all classes necessary to retrieve data from
the TOAR database.
"""

# general
import pdb

# data science
import pandas as pd

# own
import settings
from retrieval_utils import query_db
from retrieval_utils import print_statistics


class ToarRetrieval:
    """
    A super-class for all retrievals from the database.
    """
    def __init__(self):
        """
        initialize the class.
        """
        pass

    def save_to_file(self):
        """
        This file goes to the resources.
        """
        self.df.to_csv(self.path)
        print(f'Saved to file: {self.path}')

    def read_from_file(self):
        """
        If the file already exists, read it!
        """
        self.df = pd.read_csv(self.path, index_col=0)


class StationData(ToarRetrieval):
    """
    Retrieving station data.
    """
    def __init__(self):
        """
        Initialize the class with.
        """
        ToarRetrieval.__init__(self)
        self.path = settings.resources_dir + 'uba_graph_raw/' + \
                                             'stations.csv'

    def retrieve_from_toar(self):
        """
        Query data from the database.
        """
        print('Retrieve station data...')

        query = """
                SELECT numid AS id,
                       station_lon as lon,
                       station_lat as lat,
                       station_alt as alt,
                       station_etopo_relative_alt as relative_alt,
                       station_nightlight_5km as nightlight,
                       station_population_density AS population_density,
                       TRIM(station_type) AS type,
                       TRIM(station_type_of_area) AS type_of_area
                FROM stations
                WHERE network_name = 'UBA'
                ORDER BY id;
                """
        df = query_db(query).set_index('id')
        self.df = df

    def drop_ids_without_data(self):
        """
        Every id in our list should have metadata, meteorological data,
        and at least one ozone measurement. We use a valid relative
        alt, and pbl height as indicators for this.
        """
        df = self.df

        # missing metadata
        df = df[df.relative_alt>-999.]

        # now check for the ids. This takes some time
        no_data_list = []
        for station_id in df.index:
            o3_query = f"""
                SELECT id
                FROM parameter_series
                WHERE station_numid={station_id}
                AND parameter_name='o3'
                AND data_start_date < '{settings.datetime_end}'
                AND data_end_date > '{settings.datetime_start}';
                """
            o3_df = query_db(o3_query)
            pbl_query = f"""
                SELECT id
                FROM parameter_series
                WHERE station_numid={station_id}
                AND parameter_name='pblheight';
                """
            pbl_df = query_db(pbl_query)
            if len(o3_df)==0 or len(pbl_df)==0:
                # print(station_id)
                no_data_list.append(station_id)

        self.df = df.drop(no_data_list)


class SeriesData(ToarRetrieval):
    """
    Retrieving series data.
    """
    def __init__(self, var):
        """
        Initialize the class with.
        """
        ToarRetrieval.__init__(self)
        self.var = var
        self.path = settings.resources_dir + 'uba_graph_raw/' + \
                                            f'series_{var}.csv'

    def retrieve_from_toar(self):
        """
        Query data from the database.
        """
        print(f'Retrieving series data for {self.var}...')

        # df with the station ids
        sd = StationData()
        sd.read_from_file()

        # initialile df for parameter series
        columns=['id', 'station_id', 'datetime_start', 'datetime_end']
        self.df = pd.DataFrame(columns=columns)

        measurement_method_part = ''
        if self.var != 'o3':
            measurement_method_part = """
                  AND parameter_measurement_method='model simulation'
                                      """

        # find the series properties
        for station_id in sd.df.index.to_list():
            query = f"""
                SELECT id,
                       station_numid AS station_id,
                       data_start_date AS datetime_start,
                       data_end_date AS datetime_end
                FROM parameter_series
                WHERE station_numid={station_id}
                AND parameter_name='{self.var}'
                {measurement_method_part}
                ORDER BY id;
                """
            rows = query_db(query)
            self.df = pd.concat([self.df, rows])

        # last, set the right index
        self.df = self.df.set_index('id')


class HourlyData(ToarRetrieval):
    """
    Retrieving hourly data.
    """
    def __init__(self, var):
        """
        Initialize the class with its variable and path.
        """
        ToarRetrieval.__init__(self)
        self.var = var
        self.path = settings.resources_dir + 'uba_graph_raw/' + \
                                            f'hourly_{var}.csv'
        self.datetime_start = settings.datetime_start
        self.datetime_end = settings.datetime_end

    def retrieve_from_toar(self):
        """
        Query data from the database.
        """
        print(f'Retrieving hourly data for {self.var}...')

        # all stations
        std = StationData()
        std.read_from_file()
        station_id_list = std.df.index.to_list()

        # we also need the series ids
        ser = SeriesData(self.var)
        ser.read_from_file()

        # initialize df
        date_range = pd.date_range(start=self.datetime_start,
                                   end=self.datetime_end,
                                   freq='H',
                                   closed=None)
        self.df = pd.DataFrame(index=date_range,
                               columns=station_id_list)

        # now retrieve the data
        for idx, row in ser.df.iterrows():
            series_id = idx
            station_id = row.station_id
            # print(station_id)
            query = f"""
                    SELECT datetime, value
                    FROM {self.var}_hourly
                    WHERE id={series_id}
                    AND datetime BETWEEN '{self.datetime_start}'
                                     AND '{self.datetime_end}';
                    """
            hourly = query_db(query).set_index('datetime')
            self.df.loc[hourly.index, station_id] = hourly.value.to_list()


def main_retrieval():
    """
    Retrieve the data from TOAR
    """
    std = StationData()
    std.retrieve_from_toar()
    std.drop_ids_without_data()
    std.save_to_file()

    var_list = ['o3', 'cloudcover', 'pblheight', 'relhum', 'temp', 'u',
                'v']

    for var in var_list:
        sed = SeriesData(var)
        sed.retrieve_from_toar()
        sed.save_to_file()

    for var in var_list:
        hd = HourlyData(var)
        hd.retrieve_from_toar()
        hd.save_to_file()


def main_sanity_checks():
    """
    Sanity checks for the data (TODO)
    """
    print('Starting sanity checks...')
    std = StationData()
    std.read_from_file()
    print(f'Checking {std.path}...')
    for col in std.df.columns:
        data = std.df[col]
        print_statistics(col, data)

    var_list = ['o3', 'cloudcover', 'pblheight', 'relhum', 'temp', 'u',
                'v']
    for var in var_list:
        hd = HourlyData(var)
        hd.read_from_file()
        print_statistics(var, hd.df)


if __name__ == '__main__':
    """
    Start the specified routine.
    """
    # main_retrieval()
    main_sanity_checks()

