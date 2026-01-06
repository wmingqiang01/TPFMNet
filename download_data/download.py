"""
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-13 15:10:31
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-12-14 16:57:36
FilePath: \OSI-450-a\download.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""

import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests


def download_data(start_date, end_date, output_directory):
    base_url = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/conc_450a_files/monthly"

    current_date = start_date
    while current_date <= end_date:
        file_date = current_date.strftime("%Y%m")
        file_url = f"{base_url}/{current_date.year}/ice_conc_nh_ease2-250_cdr-v3p0_{file_date}.nc"
        output_file = os.path.join(
            output_directory, f"ice_conc_nh_ease2-250_cdr-v3p0_{file_date}.nc"
        )

        try:
            # Download file
            response = requests.get(file_url)
            response.raise_for_status()  # Raise HTTPError for bad responses

            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {output_file}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {file_url}: {e}")

        # Move to the next day
        current_date += relativedelta(months=1)


if __name__ == "__main__":
    start_date = datetime(1988, 1, 1)
    end_date = datetime(2020, 12, 1)
    output_directory = "."

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    download_data(start_date, end_date, output_directory)
