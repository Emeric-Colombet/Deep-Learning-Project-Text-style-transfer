"""This module loads subtitles data used in style transfer project """

from dataclasses import dataclass
import pandas as pd
import numpy as np
import pysrt
import webvtt
import datetime
import math
import git
import os


@dataclass
class MyRepo:
    repo: str = git.Repo('../domain', search_parent_directories=True).working_tree_dir

    @classmethod
    def find_path(cls, target_folder):
        path = os.path.join(cls.repo, target_folder)
        return path


class Subtitles(MyRepo):
    """Obtain content from subtitle files"""

    def __init__(self, df: pd.DataFrame):
        """Constructor to initialize Subtitles class

        :parameters:
            df: DataFrame with list of subtitle files
        """

        self.df = df

    @classmethod
    def _get_files_path(cls, file_extension: str) -> str:
        """
        Obtain path holding subtitle files

        :parameters:
            file_extension: extension of subtitle files ('srt' or 'vtt')
        :returns:
            path: path holding subtitle files of a given format
        """

        TARGET_FOLDER = 'data/raw'
        target_folder = os.path.join(TARGET_FOLDER, file_extension)
        path = cls.find_path(target_folder)

        return path

    @classmethod
    def get_subtitle_files(cls, file_extension: str) -> pd.DataFrame:
        """
        Obtain dataframe with list of available subtitle files

        :parameters:
            file_extension: extension of subtitle files ('srt' or 'vtt')
        :returns:
            df: DataFrame with subtitle file names
        """

        path = cls._get_files_path(file_extension)
        df = pd.DataFrame(os.listdir(path), columns=['file'])

        if file_extension == 'srt':
            # Currently handles files names for series as no movie with SRT subtitles is present
            # In the future, it could also take into account the movie name format for SRT files
            df['region'] = np.where(df['file'].str.contains('(Latin America)'), 'latinamerica', 'spain')
            df['title'] = df['file'].str.split('.S0', expand=True)[0]
            df['episode'] = df.apply(
                lambda x: x['file'][x['file'].find('.S0') + 1:x['file'].find('.S0') + len('S00E00') + 1], axis=1)
            df['title_episode'] = df['title'] + '-' + df['episode']

        else:
            # Handles name format of series and movies subtitle files
            df['region'] = np.where(df['file'].str.contains('es-ES'), 'spain', 'latinamerica')
            df['title'] = np.where(
                df['file'].str.contains('S0'),
                df['file'].str.split('.S0', expand=True)[0],
                df['file'].str.split('.WEBRip', expand=True)[0])
            df['episode'] = np.where(
                df['file'].str.contains(r"\bS0."),
                df.apply(
                    lambda x: x['file'][x['file'].find('.S0') + 1:x['file'].find('.S0') + len('S00E00') + 1], axis=1),
                'movie')
            df['title_episode'] = np.where(
                df['file'].str.contains(r"\bS0."),
                df['title'] + '-' + df['episode'],
                df['title'])

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
        """Obtains equivalent subtitle files

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
            episode: episode of show or 'film'
        :returns:
            index: Index of title-episode in DataFrame
        """

        index = np.where((self.df['title'] == title) & (self.df['episode'] == episode))[0][0]

        return index

    def _get_title_info(self, index: int) -> (str, str):
        """
        Get title and episode of a given index

        :parameters:
            index: Index of title-episode in DataFrame
        :returns:
            title: Title of show or film of a specific DataFrame index
            episode: Episode of show or film of a specific DataFrame index
            title_episode: Title and episode of show or film of a specific DataFrame index
        """

        title = self.df['title'][index]
        episode = self.df['episode'][index]

        return title, episode

    def _get_title_path(self, file_extension: str, index: int, region: str) -> str:
        """
        Get path of subtitle file (for a specific title-episode and region)

        :parameters:
            file_extension: extension of subtitle files ('srt' or 'vtt')
            index: Index of title-episode in DataFrame
            region: region of subtitle file
        :returns:
            full_path: Path to subtitle file (for a specific title-episode and region)
        """

        path = self._get_files_path(file_extension)

        region_field = 'file_' + region
        file = self.df[region_field][index]
        full_path = os.path.join(path, file)

        return full_path

    @classmethod
    def remove_formatting(cls, series: pd.Series) -> pd.Series:
        """
        Remove line break characters or HTML tags from subtitle content

        :parameters:
            series: Series with subtitle content
        :returns:
            series: Series with cleaned-up content
        """

        series = series\
            .str.replace(r'^- ', '', regex=True)\
            .str.replace(r'\n- ', ' ', regex=True)\
            .str.replace(r'\n', ' ', regex=True)\
            .str.replace(r'<[^<>]*>', '', regex=True)

        return series

    def remove_informational_phrases():
        """
        Remove rows with MY-SUB.co, Subtítulos por:, UNA SERIE ORIGINAL DE NETFLIX, etc
        """
        pass

    @classmethod
    def _get_subtitles(cls, path: str) -> pd.DataFrame:
        """
        Get subtitles from file

        :parameters:
            path: Path to subtitle file
        :returns:
            df: Dataframe with subtitle content for a given title-episode-region
        """

        if path.endswith('srt'):
            subs_raw = pysrt.open(path)
        else:
            subs_raw = webvtt.read(path)

        subs = [(caption.start, caption.end, caption.text) for caption in subs_raw]
        df = pd.DataFrame(subs, columns=['start', 'end', 'text'])

        if path.endswith('vtt'):
            df['start'] = [e[:-4] for e in df['start']]
            df['end'] = [e[:-4] for e in df['end']]

            df['start'] = [datetime.datetime.strptime(e, '%H:%M:%S').time() for e in df['start']]
            df['end'] = [datetime.datetime.strptime(e, '%H:%M:%S').time() for e in df['end']]

        df['text'] = cls.remove_formatting(df['text'])

        return df

    @classmethod
    def _add_time_grouping_to_subtitles(cls, file_extension: str, df: pd.DataFrame, seconds_range=5) -> pd.DataFrame:
        """
        Adds a time range grouping to subtitle data

        :parameters:
            file_extension: extension of subtitle files ('srt' or 'vtt')
            df: DataFrame with subtitle content for a given title-episode-region
            seconds_range: Time range in seconds, default is 5
        :returns:
            df_range: Dataframe with subtitles grouped by specific tranches of time
        """

        if file_extension == 'srt':
            start_time_range = [
                datetime.time(
                    # SubRipTime object with different syntax from regular datetime (hours vs hour, etc)
                    e.hours,
                    e.minutes,
                    round(math.floor(e.seconds / seconds_range)) * seconds_range,
                    0)
                for e in df['start']]
        else:
            start_time_range = [
                datetime.time(
                    e.hour,
                    e.minute,
                    round(math.floor(e.second / seconds_range)) * seconds_range,
                    0)
                for e in df['start']]

        df['start_time_range'] = pd.Series(start_time_range)

        df_range = df.groupby(['start_time_range'], as_index=False).agg({'text': ' '.join})

        return df_range

    def _get_grouped_subtitles(self, file_extension: str, index: int, region: str) -> pd.DataFrame:
        """
        Group subtitles within a time range

        :parameters:
            file_extension: extension of subtitle files ('srt' or 'vtt')
            index: Index of title-episode in DataFrame
            region: region of subtitle file
        :returns:
            df_range: Dataframe with subtitles grouped by specific tranches of time
        """

        path = self._get_title_path(file_extension, index, region)
        df = self._get_subtitles(path)
        df_range = self._add_time_grouping_to_subtitles(file_extension, df, seconds_range=5)

        return df_range

    def get_subtitle_equivalents(self, file_extension: str, index: int) -> pd.DataFrame:
        """
        Provides side-by-side comparison of regional subtitle content, for a given title-episode

        :parameters:
            file_extension: extension of subtitle files ('srt' or 'vtt')
            index: Index of title-episode in DataFrame containing subtitle file equivalences
        :returns:
            df_comparison: Dataframe with subtitles comparison per specific tranches of time
        """

        ORIGINAL_REGION = 'latinamerica'
        TARGET_REGION = 'spain'

        df_original = self._get_grouped_subtitles(file_extension, index, ORIGINAL_REGION)
        df_target = self._get_grouped_subtitles(file_extension, index, TARGET_REGION)

        title, episode = self._get_title_info(index)

        df_comparison = pd.merge(
            df_original,
            df_target,
            on='start_time_range',
            suffixes=('_' + ORIGINAL_REGION, '_' + TARGET_REGION)
        )

        df_comparison['title'] = title
        df_comparison['episode'] = episode

        return df_comparison

    @classmethod
    def load_raw_data_per_type(cls, file_extension: str) -> pd.DataFrame:
        """
        Concatenate all subtitles from SRT or VTT files in a single DataFrame

        :parameters:
            file_extension: extension of subtitle files ('srt' or 'vtt')
        :returns:
            subtitles_all_titles: DataFrame with subtitles for various titles-episodes-regions
        """

        files = cls.get_subtitle_files(file_extension)
        files_equivalent = cls(files).get_equivalent_files()

        nb_subtitle_pairs = len(files_equivalent)
        df = pd.DataFrame()
        encoding_errors = []

        for i in range(nb_subtitle_pairs):
            try:
                df_tmp = cls(files_equivalent).get_subtitle_equivalents(file_extension, i)
                df = pd.concat([df, df_tmp], ignore_index=True)

            except UnicodeDecodeError:
                # Files that could not be accessed due to encoding issues:
                # Not to be used but keep the info if necessary for debugging purposes
                title, episode = cls(files_equivalent)._get_title_info(i)
                encoding_errors.append(title + '-' + episode)

        return df

    @classmethod
    def print_to_csv(cls, df: pd.DataFrame, path: str, file_name: str, index=False) -> None:
        """
        Export DataFrame to a CSV file

        :parameters:
            df: DataFrame to export
            path: Path where file will be stored
            file_name: Name of file (including extension such as '.csv')
        """

        full_path = os.path.join(path, file_name)
        df.to_csv(full_path, index=index, encoding='utf-8-sig')

        print(f'Export done: {full_path}')

    @classmethod
    def aggregate_subtitle_types(cls) -> pd.DataFrame:
        """Collect properly arranged subtitles from SRT and VTT files

        :returns:
            df_all: DataFrame containing all subtitles present in SRT and VTT files

        Structure:
            index: DataFrame index
            start_time_range: Time range of subtitle dialogue (default is 5 seconds)
            text_latinamerica: Text of Latin-American Spanish
            text_spain: Text of European Spanish
            title: Title of show or film
            episode: Episode number of show or 'film'
        """
        df_srt = Subtitles.load_raw_data_per_type('srt')
        df_vtt = Subtitles.load_raw_data_per_type('vtt')
        df_all = pd.concat([df_srt, df_vtt], ignore_index=True)

        TARGET_FOLDER = 'data'
        path = cls.find_path(TARGET_FOLDER)
        cls.print_to_csv(df_all, path, 'all_subtitles.csv')

        return df_all

    @classmethod
    def load_aggregated_data(cls) -> pd.DataFrame:
        """
        Concatenate all subtitles from SRT or VTT files in a single DataFrame

        :returns:
            subtitles_all_titles: DataFrame with subtitles for various titles-episodes-regions
        """
        TARGET_FOLDER = 'data/all_subtitles.csv'
        path = cls.find_path(TARGET_FOLDER)

        df = pd.read_csv(path)

        return df


@dataclass
class StyleTransferData(Subtitles):
    """Class with base data for pre-processing"""

    data: pd.DataFrame = Subtitles.load_aggregated_data()
