"""
Print aggregated data from individual SRT and VTT files.

Only needs to be run once in order to have one CSV that
regroups all subtitle data in just one file and that new file
will be used as the main "raw" data source going forward.
"""

from style_transfer.infrastructure.style_transfer_data import Subtitles

Subtitles.aggregate_subtitle_types()
