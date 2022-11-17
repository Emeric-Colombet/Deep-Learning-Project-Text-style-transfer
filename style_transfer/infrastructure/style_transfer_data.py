"""This module loads subtitles data used in style transfer project """

from dataclasses import dataclass
import pandas as pd
import numpy as np
import pysrt
import datetime
import math
import os


@dataclass
class StyleTransferData:
    """Represent Spanish subtitles in two styles"""

    # TODO: remove them or keep them once VTT class/methods are available
    # raw_data_file_name: str
    # cleaned_data_file_name: str

    @staticmethod
    def load_clean_data() -> pd.DataFrame:
        """Collect properly arranged subtitles from SRT and VTT files

        :returns:
            df_all: DataFrame containing all subtitles present in SRT and VTT files

        Structure:
            index: DataFrame index
            start_time_range: Time range of subtitle dialogue (default is 5 seconds)
            text_latinamerica: Text of Latin-American Spanish
            text_spain: Text of European Spanish
            title: Title of show or movie
            episode: Episode number of show, "It's a movie" for films
        """

        df_srt = SubtitleDataSrt.load_raw_data_srt()
        df_vtt = SubtitleDataSrt.load_raw_data_srt()  # placeholder for load_raw_data_vtt()
        df_all = pd.concat([df_srt, df_vtt], ignore_index=True)

        return df_all

    @staticmethod
    def print_to_csv(df: pd.DataFrame, path: str, file_name: str) -> None:
        """
        Export DataFrame to a CSV file

        :parameters:
            df: DataFrame to export
            path: Path where file will be stored
            file_name: Name of file (including extension such as ".csv")
        """

        full_path = os.path.join(path, file_name)
        df.to_csv(full_path, index=False, encoding='utf-8-sig')

        print(f'Export done: {full_path}')


class SubtitleDataSrt:
    """Obtain content from SRT files"""

    def __init__(self, df: pd.DataFrame):
        """Constructor to initialize SubtitleDataSrt class

        :parameters:
            df: DataFrame with list of subtitle files
        """

        self.df = df

    @staticmethod
    def _get_srt_path() -> str:
        """
        Obtain path holding SRT files

        :returns:
            path: path holding SRT files
        """

        PATH_RELATIVE = '../../data/raw/srt'
        path = os.path.abspath(PATH_RELATIVE)

        return path

    @staticmethod
    def get_subtitle_files() -> pd.DataFrame:
        """
        Obtain dataframe with list of available subtitle files

        :returns:
            df: DataFrame with subtitle file names
        """

        path = SubtitleDataSrt._get_srt_path()
        df = pd.DataFrame(os.listdir(path), columns=['file'])

        df['region'] = np.where(df['file'].str.contains('(Latin America)'), 'latinamerica', 'spain')
        df['title'] = df['file'].str.split('.S0', expand=True)[0]
        df['episode'] = df.apply(
            lambda x: x['file'][x['file'].find('.S0') + 1:x['file'].find('.S0') + len('S00E00') + 1], axis=1)
        df['title_episode'] = df['title'] + '-' + df['episode']

        df.sort_values(by=['title', 'episode', 'region'], inplace=True, ignore_index=True)

        return df

    def _find_files_without_equivalent(self) -> (list, pd.DataFrame):
        """Find files that do not have an equivalent subtitle file

        :returns:
            files_to_remove:
                List of title-episodes that do not have an equivalent and must be removed
            summary_to_remove:
                DataFrame that shows number of equivalent files and order of region, for debugging purposes
        """

        df_count = self.df.groupby(['title_episode'], as_index=False).agg({'region': ['-'.join, 'count']})
        df_count.columns = df_count.columns.to_flat_index()

        summary_to_remove = df_count[
            (df_count[('region', 'count')] != 2) |
            (df_count[('region', 'join')] != 'latinamerica-spain')
            ]

        files_to_remove = list(summary_to_remove[('title_episode', '')])

        return files_to_remove, summary_to_remove

    def _remove_files_without_equivalent(self) -> pd.DataFrame:
        """Remove files that do not have an equivalent subtitle file

        :returns:
            self.df: Same DataFrame without rows that include a subtitle that does not have an equivalent
        """
        files_to_remove, _ = self._find_files_without_equivalent()
        indices_to_remove = list(self.df[self.df['title_episode'].isin(files_to_remove)].index)

        self.df.drop(indices_to_remove, inplace=True)

        return self.df

    def get_equivalent_files(self) -> pd.DataFrame:
        """Obtain equivalent subtitle files

        :returns:
            df_equivalent: DataFrame per title/episode showing equivalent subtitle files
        """

        SEPARATOR = '<><><>'  # Pattern that might not exist in file name already
        ORIGINAL_REGION = 'latinamerica'
        TARGET_REGION = 'spain'

        df = self._remove_files_without_equivalent()

        # OPTIMIZE: Only works when data is sorted and Latin America row is before Spain row
        df_equivalent = df.groupby(['title', 'episode'], as_index=False).agg({'file': SEPARATOR.join})

        df_equivalent['file_' + ORIGINAL_REGION] = df_equivalent['file'].str.split(SEPARATOR, expand=True)[0]
        df_equivalent['file_' + TARGET_REGION] = df_equivalent['file'].str.split(SEPARATOR, expand=True)[1]
        df_equivalent.drop(['file'], axis=1, inplace=True)

        return df_equivalent

    def _get_title_index(self, title: str, episode: str) -> int:
        """
        Get the index of title-episode

        :parameters:
            title: title of show or film
            episode: episode of show or "It's a movie" if film
        :returns:
            index: Index of title-episode in DataFrame
        """

        index = np.where((self.df['title'] == title) & (self.df['episode'] == episode))[0][0]

        return index

    def get_title_info(self, index: int) -> (str, str):
        """
        Get title and episode of a given index

        :parameters:
            index: Index of title-episode in DataFrame
        :returns:
            title: Title of show or film of a specific DataFrame index
            episode: Episode of show or film of a specific DataFrame index
        """

        title = self.df['title'][index]
        episode = self.df['episode'][index]

        return title, episode

    def _get_title_path(self, index: int, region: str) -> str:
        """
        Get path of subtitle file (for a specific title-episode and region)

        :parameters:
            index: Index of title-episode in DataFrame
            region: region of subtitle file
        :returns:
            full_path: Path to subtitle file (for a specific title-episode and region)
        """

        path = SubtitleDataSrt._get_srt_path()

        region_field = 'file_' + region
        file = self.df[region_field][index]
        full_path = os.path.join(path, file)

        return full_path

    @staticmethod
    def _get_subtitles(path: str) -> pd.DataFrame:
        """
        Get subtitles from an SRT file

        :parameters:
            path: Path to subtitle file
        :returns:
            df: Dataframe with subtitle content for a given title-episode-region

        :note:
            FutureWarning: The default value of regex will change from True to False in a future version.
            df['text'] = df['text'].str.replace('^- ', '').str.replace('\n- ', ' ').str.replace('\n', ' ')
        """

        subs_srt = pysrt.open(path)

        subs = []
        for sub in subs_srt:
            subs.append((sub.start, sub.end, sub.text))

        df = pd.DataFrame(subs, columns=['start', 'end', 'text'])

        # TODO: Can be done in domain pre-processing
        df['text'] = df['text']\
            .str.replace('^- ', '', regex=True)\
            .str.replace('\n- ', ' ', regex=True)\
            .str.replace('\n', ' ', regex=True)

        return df

    @staticmethod
    def _add_time_grouping_to_subtitles(df: pd.DataFrame, seconds_range=5) -> pd.DataFrame:
        """
        Adds a time range grouping to subtitle data

        :parameters:
            df: DataFrame with subtitle content for a given title-episode-region
            seconds_range: Time range in seconds, default is 5
        :returns:
            df_range: Dataframe with subtitles grouped by specific tranches of time
        """

        start_time = []
        start_time_range = []
        end_time = []

        for i in range(len(df)):
            s = round(math.floor(df['start'][i].seconds / seconds_range)) * seconds_range

            start_time_range.append(datetime.time(
                df['start'][i].hours,
                df['start'][i].minutes,
                s,
                0
            ))

            start_time.append(datetime.time(
                df['start'][i].hours,
                df['start'][i].minutes,
                df['start'][i].seconds,
                df['start'][i].milliseconds
            ))

            end_time.append(datetime.time(
                df['end'][i].hours,
                df['end'][i].minutes,
                df['end'][i].seconds,
                df['end'][i].milliseconds
            ))

        df['start_time_range'] = pd.Series(start_time_range)
        df['start_time'] = pd.Series(start_time)
        df['end_time'] = pd.Series(end_time)

        df_range = df.groupby(['start_time_range'], as_index=False).agg({'text': ' '.join})

        return df_range

    def _get_grouped_subtitles(self, index, region) -> pd.DataFrame:
        """
        Group subtitles within a time range

        :parameters:
            seconds_range: Time range in seconds, default is 5
        :returns:
            df_range: Dataframe with subtitles grouped by specific tranches of time
        """

        path = self._get_title_path(index, region)
        df = SubtitleDataSrt._get_subtitles(path)
        df_range = SubtitleDataSrt._add_time_grouping_to_subtitles(df, seconds_range=5)

        return df_range

    def get_subtitle_equivalents(self, index: int) -> pd.DataFrame:
        """
        Provides side-by-side comparison of regional subtitle content, for a given title-episode

        :parameters:
            index: Index of title-episode in DataFrame containing subtitle file equivalences
        :returns:
            df_comparison: Dataframe with subtitles comparison per specific tranches of time
        """

        ORIGINAL_REGION = 'latinamerica'
        TARGET_REGION = 'spain'

        df_original = self._get_grouped_subtitles(index, ORIGINAL_REGION)
        df_target = self._get_grouped_subtitles(index, TARGET_REGION)

        title, episode = self.get_title_info(index)

        df_comparison = pd.merge(
            df_original,
            df_target,
            on='start_time_range',
            suffixes=('_' + ORIGINAL_REGION, '_' + TARGET_REGION)
        )

        df_comparison['title'] = title
        df_comparison['episode'] = episode

        return df_comparison

    @staticmethod
    def load_raw_data_srt() -> pd.DataFrame:
        """
        Concatenate all subtitles from SRT files in a single DataFrame

        :returns:
            subtitles_all_titles: DataFrame with subtitles for various titles-episodes-regions
        """

        files = SubtitleDataSrt.get_subtitle_files()
        files_equivalent = SubtitleDataSrt(files).get_equivalent_files()

        nb_subtitle_pairs = len(files_equivalent)
        subtitles_all_titles = pd.DataFrame()
        encoding_errors = []

        for i in range(nb_subtitle_pairs):
            try:
                df = SubtitleDataSrt(files_equivalent).get_subtitle_equivalents(i)
                subtitles_all_titles = pd.concat([subtitles_all_titles, df], ignore_index=True)

            except UnicodeDecodeError:
                # Files that could not be accessed due to encoding issues:
                # Not be used but keep the info if necessary for debugging purposes
                title, episode = SubtitleDataSrt(files_equivalent).get_title_info(i)
                encoding_errors.append(title + '-' + episode)

        return subtitles_all_titles
