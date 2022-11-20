from style_transfer.infrastructure.style_transfer_data import Subtitles

# Print aggregated data from individual SRT and VTT files
# Only needs to be run once in order to have one CSV that regroups all data together in just on file
# which will be used as the main data source going forward
Subtitles.aggregate_subtitle_types()
